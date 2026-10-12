"""TRELLIS_REMESH_MEM_GIB changes only how many workers build the remesh candidate bitset, never the result (cloud, Linux).

Needs numpy, and a post-replay built from tools/arctic-post (ARCTIC_POST_BUILD=<build dir>, as for test_phase_lock.py).
Makes a small synthetic dump with gen_dump.py, runs post-replay up to the end of the front chain with several budgets that
force 4, 2 and 1 workers, and requires identical stage hashes (TRELLIS_POST_HASH=1). python3 -m unittest tools/arctic-post/test_remesh_budget.py
"""
import os, re, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""


class RemeshBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b = os.environ.get("ARCTIC_POST_BUILD")
        cls.exe = os.path.join(b, "post-replay" + EXE) if b else None
        if not cls.exe or not os.path.isfile(cls.exe):
            raise unittest.SkipTest("set ARCTIC_POST_BUILD to a build dir containing post-replay")
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("numpy needed for gen_dump.py")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dump = os.path.join(cls.tmp.name, "syn.bin")
        subprocess.run([sys.executable, os.path.join(HERE, "gen_dump.py"), cls.dump, "--N", "128", "--pbr-res", "128"],
                       check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def front(self, budget):
        env = dict(os.environ, TRELLIS_POST_THREADS="4", TRELLIS_POST_HASH="1")
        env.pop("TRELLIS_REMESH_MEM_GIB", None)
        if budget is not None:
            env["TRELLIS_REMESH_MEM_GIB"] = str(budget)
        r = subprocess.run([self.exe, self.dump, os.path.join(self.tmp.name, "o.glb"), "--geometry-res", "512", "--band", "1",
                            "--stop-before-qem", "--no-audit"], env=env, capture_output=True, text=True, check=True)
        hashes = re.findall(r"\[post-hash\]\s+(remesh|clean_mesh|drop_components)\s+(.*)", r.stdout)
        workers = re.search(r"workers (\d+)->(\d+)", r.stdout)
        return hashes, (int(workers.group(2)) if workers else 4)

    def test_hashes_do_not_depend_on_the_budget(self):
        ref, w = self.front(None)
        self.assertEqual(w, 4); self.assertEqual(len(ref), 3)
        for budget, expect in ((0.04, 2), (0.02, 1), (0.005, 1), (8, 4)):  # 16 MiB bitset per worker at res 512
            h, w = self.front(budget)
            self.assertEqual(w, expect, budget); self.assertEqual(h, ref, "budget %s changed the remesh" % budget)

    def test_invalid_value_means_the_default(self):
        ref, _ = self.front(None)
        h, w = self.front("abc")
        self.assertEqual((h, w), (ref, 4))


if __name__ == "__main__":
    unittest.main()
