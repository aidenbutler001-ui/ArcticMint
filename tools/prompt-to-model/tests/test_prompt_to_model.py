"""Tests for prompt_to_model with a fake ComfyUI and a fake engine (no GPU, no models):  python -m unittest discover tools/prompt-to-model/tests"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import comfy_client  # noqa: E402
import prompt_to_model as p2m  # noqa: E402
from fake_comfy import FakeComfy  # noqa: E402

ENGINE = "%s %s" % (sys.executable, os.path.join(HERE, "fake_engine.py"))


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.comfy = FakeComfy(0, speed=0.01).start()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.addCleanup(self.comfy.stop)

    def run_main(self, *args):
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = p2m.main(list(args))
        return rc, err.getvalue()

    def base(self, *extra, model="klein"):
        return ["a wooden chest with iron bands", "-o", self.tmp, "--comfy-url", self.comfy.url, "--cli", ENGINE, "-m", "models",
                "--image-model", model] + list(extra)


class TestPipeline(Run):
    def test_prompt_to_glb(self):
        rc, _ = self.run_main(*self.base("--seed", "7", "--gpu", "1", "--engine-args", "--res 512"))
        self.assertEqual(rc, 0)
        glb = os.path.join(self.tmp, "a_wooden_chest_with_iron_bands_s7.glb")
        self.assertTrue(os.path.isfile(glb))
        self.assertIn(b"--seed 7", open(glb, "rb").read())
        self.assertIn(b"--gpu 1", open(glb, "rb").read())
        self.assertIn(b"--res 512", open(glb, "rb").read())
        self.assertIn(b"--require-gpu", open(glb, "rb").read())
        rec = json.load(open(os.path.join(self.tmp, "a_wooden_chest_with_iron_bands_s7.json")))
        self.assertEqual(rec["engine_exit"], 0)
        self.assertTrue(rec["final_prompt"].endswith(p2m.FRAMING))
        self.assertTrue(os.path.isfile(rec["pictures"][0]))
        self.assertGreaterEqual(self.comfy.freed, 1)

    def test_zimage_and_style_none(self):
        rc, _ = self.run_main(*self.base("--style", "none", model="zimage"))
        self.assertEqual(rc, 0)
        rec = json.load(open(os.path.join(self.tmp, "a_wooden_chest_with_iron_bands_s1.json")))
        self.assertEqual(rec["final_prompt"], "a wooden chest with iron bands")

    def test_picture_only_count(self):
        rc, _ = self.run_main(*self.base("--picture-only", "--count", "3", "--seed", "10", "--name", "chest"))
        self.assertEqual(rc, 0)
        for s in (10, 11, 12):
            self.assertTrue(os.path.isfile(os.path.join(self.tmp, "chest_s%d.png" % s)))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "chest_s10.glb")))

    def test_image_after_picture_only_keeps_prompt(self):
        self.run_main(*self.base("--picture-only", "--count", "2", "--seed", "4", "--name", "c"))
        rc, err = self.run_main("--image", os.path.join(self.tmp, "c_s5.png"), "--seed", "5", "-o", self.tmp, "--cli", ENGINE, "-m", "m")
        self.assertEqual(rc, 0, err)                                     # no --force needed for the same run
        rec = json.load(open(os.path.join(self.tmp, "c_s5.json")))
        self.assertEqual(rec["prompt"], "a wooden chest with iron bands")
        self.assertTrue(rec["final_prompt"].endswith(p2m.FRAMING))
        self.assertEqual(rec["engine_exit"], 0)
        self.assertEqual(rec["seed"], 5)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "c_s5.glb")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "c_s4.glb")))
        again, err = self.run_main("--image", os.path.join(self.tmp, "c_s5.png"), "--seed", "5", "-o", self.tmp, "--cli", ENGINE, "-m", "m")
        self.assertEqual(again, 1)                                       # a finished model is not silently replaced
        self.assertIn("would overwrite", err)

    def test_prompt_file(self):
        pf = os.path.join(self.tmp, "prompt.txt")
        with open(pf, "w", encoding="utf-8-sig") as f:               # PowerShell's Out-File writes a BOM
            f.write('a brass "diving"\n  helmet\n')
        out = os.path.join(self.tmp, "pf")
        rc, err = self.run_main("--prompt-file", pf, "-o", out, "--comfy-url", self.comfy.url, "--picture-only", "--name", "h")
        self.assertEqual(rc, 0, err)
        rec = json.load(open(os.path.join(out, "h_s1.json")))
        self.assertEqual(rec["prompt"], 'a brass "diving" helmet')

    def test_image_skips_comfy(self):
        self.run_main(*self.base("--picture-only", "--name", "c"))
        pic = os.path.join(self.tmp, "c_s1.png")
        out2 = os.path.join(self.tmp, "second")
        rc, _ = self.run_main("--image", pic, "-o", out2, "--cli", ENGINE, "-m", "models", "--comfy-url", "http://127.0.0.1:9")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(os.path.join(out2, "c_s1.glb")))      # the picture's _s1 is not doubled


class TestSafety(Run):
    def test_second_seed_keeps_first_run(self):
        self.assertEqual(self.run_main(*self.base("--name", "c", "--seed", "1"))[0], 0)
        self.assertEqual(self.run_main(*self.base("--name", "c", "--seed", "2"))[0], 0)
        for ext in (".glb", ".json", ".png"):
            for sd in (1, 2):
                self.assertTrue(os.path.isfile(os.path.join(self.tmp, "c_s%d%s" % (sd, ext))))

    def test_same_seed_refused_until_force(self):
        self.assertEqual(self.run_main(*self.base("--name", "c"))[0], 0)
        glb = os.path.join(self.tmp, "c_s1.glb")
        before = os.stat(glb).st_mtime_ns
        rc, err = self.run_main(*self.base("--name", "c"))
        self.assertEqual(rc, 1)
        self.assertIn("would overwrite", err)
        self.assertEqual(os.stat(glb).st_mtime_ns, before)
        self.assertEqual(self.run_main(*self.base("--name", "c", "--force"))[0], 0)

    def test_force_failed_run_leaves_no_old_glb(self):
        self.run_main(*self.base("--name", "c"))
        os.environ["FAKE_ENGINE_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_ENGINE_FAIL", None)
        rc, _ = self.run_main(*self.base("--name", "c", "--force"))
        self.assertEqual(rc, 1)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "c_s1.glb")))
        self.assertIsNone(json.load(open(os.path.join(self.tmp, "c_s1.json")))["glb"])

    def test_name_cannot_escape_out_dir(self):
        out = os.path.join(self.tmp, "out")
        rc, _ = self.run_main("a chest", "-o", out, "--comfy-url", self.comfy.url, "--cli", ENGINE, "-m", "m", "--name", "../escape")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(os.path.join(out, "escape_s1.glb")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "escape_s1.glb")))

    def test_cli_path_with_spaces(self):
        d = os.path.join(self.tmp, "Program Files", "arctic mint")
        os.makedirs(d)
        exe = os.path.join(d, "trellis cli.py")
        shutil.copy(os.path.join(HERE, "fake_engine.py"), exe)
        os.chmod(exe, 0o755)
        rc, _ = self.run_main("x", "-o", os.path.join(self.tmp, "o"), "--comfy-url", self.comfy.url, "--cli", exe, "-m", "m")
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "o", "x_s1.glb")))

    def test_size_multiple_of_16(self):
        with self.assertRaises(SystemExit):
            self.run_main(*self.base("--size", "1000"))


class TestFailures(Run):
    def test_prompt_and_prompt_file_conflict(self):
        pf = os.path.join(self.tmp, "p.txt")
        open(pf, "w").write("x")
        with self.assertRaises(SystemExit):
            self.run_main("x", "--prompt-file", pf, "-o", self.tmp)
        with self.assertRaises(SystemExit):
            self.run_main("--prompt-file", os.path.join(self.tmp, "missing.txt"), "-o", self.tmp)

    def test_picture_timeout_cancels_job(self):
        slow = FakeComfy(0, speed=1000).start()
        self.addCleanup(slow.stop)
        rc, err = self.run_main("x", "-o", self.tmp, "--comfy-url", slow.url, "--cli", ENGINE, "-m", "m", "--picture-timeout", "0.3")
        self.assertEqual(rc, 1)
        self.assertIn("took longer", err)
        self.assertGreaterEqual(slow.interrupted, 1)
        self.assertEqual(len(slow.deleted), 1)

    def test_engine_timeout_is_reported_as_timeout(self):
        os.environ["FAKE_ENGINE_SLEEP"] = "5"
        self.addCleanup(os.environ.pop, "FAKE_ENGINE_SLEEP", None)
        rc, err = self.run_main(*self.base("--engine-timeout", "0.5"))
        self.assertEqual(rc, 1)
        self.assertIn("ran longer than 0.5 s", err)

    def test_engine_failure(self):
        os.environ["FAKE_ENGINE_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_ENGINE_FAIL", None)
        rc, err = self.run_main(*self.base())
        self.assertEqual(rc, 1)
        self.assertIn("3D step failed", err)

    def test_comfy_down(self):
        rc, err = self.run_main("x", "-o", self.tmp, "--comfy-url", "http://127.0.0.1:9", "--cli", ENGINE, "-m", "m")
        self.assertEqual(rc, 1)
        self.assertIn("not reachable", err)

    def test_missing_model_file(self):
        os.environ["FAKE_COMFY_MISSING"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_COMFY_MISSING", None)
        rc, err = self.run_main(*self.base())
        self.assertEqual(rc, 1)
        self.assertIn("flux-2-klein-4b-fp8.safetensors", err)

    def test_execution_failure(self):
        os.environ["FAKE_COMFY_EXEC_FAIL"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_COMFY_EXEC_FAIL", None)
        rc, err = self.run_main(*self.base())
        self.assertEqual(rc, 1)
        self.assertIn("out of memory", err)

    def test_count_needs_picture_only(self):
        with self.assertRaises(SystemExit):
            self.run_main(*self.base("--count", "2"))

    def test_models_required(self):
        with self.assertRaises(SystemExit):
            self.run_main("x", "-o", self.tmp)


class TestUnits(unittest.TestCase):
    def test_slug(self):
        self.assertEqual(p2m.slug("A Wooden Chest, iron-bound!"), "a_wooden_chest_iron_bound")
        self.assertTrue(p2m.slug("???").startswith("model_"))
        self.assertNotEqual(p2m.slug("???"), p2m.slug("!!!"))
        self.assertEqual(p2m.slug("con"), "con_")

    def test_workflow_nodes(self):
        for key in comfy_client.IMAGE_MODELS:
            wf = comfy_client.build_workflow(key, "hello", 5, "t", 768, 512)
            spec = comfy_client.IMAGE_MODELS[key]
            self.assertEqual(wf[spec["prompt"][0]]["inputs"][spec["prompt"][1]], "hello")
            self.assertEqual(wf[spec["seed"][0]]["inputs"][spec["seed"][1]], 5)
            for nid, w, h in spec["size"]:
                self.assertEqual((wf[nid]["inputs"][w], wf[nid]["inputs"][h]), (768, 512))

    def test_split_cmd_quotes(self):
        self.assertEqual(p2m.split_cmd("python 'a b.py'"), ["python", "a b.py"])


if __name__ == "__main__":
    unittest.main()
