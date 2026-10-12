import csv, json, os, stat, sys, tempfile, time, unittest
from unittest import mock
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import arctic_batch as ab

FAKE = os.path.join(HERE, "fake_engine.py")
CLI = '"%s" "%s" cli' % (sys.executable, FAKE)
REP = '"%s" "%s" replay' % (sys.executable, FAKE)
FIN = '"%s" "%s" finish {glb} {name} {out}' % (sys.executable, FAKE)
SMI = '"%s" "%s"' % (sys.executable, os.path.join(HERE, "fake_nvidia_smi.py"))


def make_imgs(d, names):
    for n in names:
        open(os.path.join(d, n + ".png"), "wb").write(n.encode())
    return d


def run(imgs, out, *extra):
    return ab.main(["run", imgs, "-o", out, "-m", "m", "--cli", CLI, "--replay", REP, "--no-preflight", *extra])


def rows(out):
    return list(csv.DictReader(open(os.path.join(out, "batch.csv"))))


def gen_args(out, name):
    """The ARGS line the fake trellis-cli printed into its log: {'argv': [...], 'env': {...}}."""
    for line in open(os.path.join(out, "logs", name + ".gen.log"), errors="replace"):
        if line.startswith("ARGS "):
            return json.loads(line[5:])
    raise AssertionError("no ARGS line in the gen log of " + name)


class T(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = self.t.name
        self.imgs = make_imgs(os.path.join(self.d, "i") if os.makedirs(os.path.join(self.d, "i")) is None else "",
                              ["a", "b", "c", "d"])

    def tearDown(self):
        self.t.cleanup()

    def test_overlap_faster_and_identical(self):
        os.environ["FAKE_GEN_S"] = os.environ["FAKE_POST_S"] = "0.5"
        so, ov = os.path.join(self.d, "s"), os.path.join(self.d, "o")
        self.assertEqual(run(self.imgs, so, "--serial"), 0)
        self.assertEqual(run(self.imgs, ov), 0)
        ws = float(rows(so)[-1]["asset_total_s"]); wo = float(rows(ov)[-1]["asset_total_s"])
        self.assertLess(wo, ws * 0.8)
        self.assertEqual(ab.cmd_compare(type("A", (), {"dir_a": so, "dir_b": ov})), 0)
        r = rows(ov)
        self.assertEqual([x["status"] for x in r[:4]], ["ok"] * 4)
        self.assertIn("assets_per_hour", r[-1]["error"])
        self.assertFalse(os.listdir(os.path.join(ov, "dumps")) and
                         [f for f in os.listdir(os.path.join(ov, "dumps")) if f.endswith(".bin")])

    def test_env_threads_and_gpu(self):
        out = os.path.join(self.d, "o")
        run(self.imgs, out, "--total-threads", "24", "--post-workers", "3")
        e = json.load(open(os.path.join(out, "a.part.glb.env.json")))
        self.assertEqual({k: e[k] for k in ("CUDA_VISIBLE_DEVICES", "TRELLIS_POST_THREADS", "CUDA_DEVICE_ORDER")},
                         {"CUDA_VISIBLE_DEVICES": "0", "TRELLIS_POST_THREADS": "8", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        self.assertIsNone(e["TRELLIS_REMESH_MEM_GIB"])  # several post workers share the RAM: binary default
        out2 = os.path.join(self.d, "o2")
        run(self.imgs, out2, "--serial", "--qem", "cpu")
        e = json.load(open(os.path.join(out2, "a.part.glb.env.json")))
        self.assertEqual((e["CUDA_VISIBLE_DEVICES"], e["TRELLIS_POST_THREADS"]), ("-1", "24"))

    def test_failure_isolated(self):
        make_imgs(self.imgs, ["bad"])
        out = os.path.join(self.d, "o")
        self.assertEqual(run(self.imgs, out), 1)
        st = {x["asset"]: x["status"] for x in rows(out)}
        self.assertEqual(st["bad"], "gen_failed"); self.assertEqual(st["a"], "ok"); self.assertEqual(st["d"], "ok")

    def test_compare_detects_diff(self):
        a, b = os.path.join(self.d, "a"), os.path.join(self.d, "b")
        os.makedirs(a); os.makedirs(b)
        open(os.path.join(a, "x.glb"), "wb").write(b'{"generated":"1"}AA')
        open(os.path.join(b, "x.glb"), "wb").write(b'{"generated":"2"}AA')
        A = type("A", (), {"dir_a": a, "dir_b": b})
        self.assertEqual(ab.cmd_compare(A), 0)
        open(os.path.join(b, "x.glb"), "wb").write(b'{"generated":"2"}AB')
        self.assertEqual(ab.cmd_compare(A), 1)


class Parallel(unittest.TestCase):
    """Raw dump mode, instrumentation, report, preflight, GPU locks, phase lock, finish retry (cloud: fake engine)."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = self.t.name
        os.makedirs(os.path.join(self.d, "i"))
        self.imgs = make_imgs(os.path.join(self.d, "i"), ["a", "b", "c"])
        self.keys = {k: os.environ.pop(k, None) for k in ("FAKE_CPU_QEM", "FAKE_FINISH_OOM_ONCE", "FAKE_BAD_DUMP_HASH",
                                                          "FAKE_POST_FAIL", "FAKE_SMI", "GPULOCK_DIR")}
        os.environ["FAKE_GEN_S"] = os.environ["FAKE_POST_S"] = "0.1"

    def tearDown(self):
        for k, v in self.keys.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        self.t.cleanup()

    def out(self, n="o"):
        return os.path.join(self.d, n)

    def test_defaults(self):
        a = ab.build_parser().parse_args(["run", "x", "-o", "o"])
        self.assertEqual((a.post_workers, a.dump_mode, a.preflight, a.hash), (1, "cleaned", True, False))

    def test_cleaned_mode_unchanged(self):
        out = self.out(); self.assertEqual(run(self.imgs, out, "--keep-dumps"), 0)
        g = gen_args(out, "a")
        self.assertIn("--dump-post", g["argv"])
        self.assertIsNone(g["env"]["TRELLIS_STOP_AFTER_POST_DUMP"]); self.assertIsNone(g["env"]["TRELLIS_DUMP_POST"])

    def test_raw_mode_env_pair_and_no_dump_post_flag(self):
        out = self.out(); self.assertEqual(run(self.imgs, out, "--dump-mode", "raw", "--keep-dumps"), 0)
        g = gen_args(out, "b")
        self.assertNotIn("--dump-post", g["argv"])
        self.assertEqual(g["env"]["TRELLIS_STOP_AFTER_POST_DUMP"], "1")
        self.assertEqual(g["env"]["TRELLIS_DUMP_POST"], os.path.join(out, "dumps", "b.bin"))
        self.assertEqual({r["dump_mode"] for r in rows(out)[:3]}, {"raw"})

    def test_raw_mode_refuses_partial_chain(self):
        for bad in ("--no-remesh", "--fill-holes", "--atlas 4096 --no-weld"):
            with self.assertRaises(SystemExit):
                run(self.imgs, self.out(), "--dump-mode", "raw", "--replay-extra=" + bad)
        self.assertEqual(run(self.imgs, self.out(), "--replay-extra=--no-remesh"), 0)  # cleaned mode may

    def test_hash_columns(self):
        out = self.out(); self.assertEqual(run(self.imgs, out, "--dump-mode", "raw", "--hash"), 0)
        r = rows(out)[0]
        self.assertRegex(r["front_hash"], r"^[0-9a-f]{16}:[0-9a-f]{16}$"); self.assertEqual(r["dump_ok"], "yes")
        os.environ["FAKE_BAD_DUMP_HASH"] = "1"
        out2 = self.out("o2"); run(self.imgs, out2, "--dump-mode", "raw", "--hash")
        self.assertEqual(rows(out2)[0]["dump_ok"], "NO")
        self.assertIn("dump_ok NO", open(os.path.join(out2, "batch_report.md")).read())
        out3 = self.out("o3"); run(self.imgs, out3)  # no --hash: columns stay empty
        self.assertEqual((rows(out3)[0]["front_hash"], rows(out3)[0]["dump_ok"]), ("", ""))

    def test_timestamps_ordered(self):
        out = self.out(); self.assertEqual(run(self.imgs, out, "--finish-cmd", FIN), 0)
        for r in rows(out)[:3]:
            ts = [r[k] for k in ("gen_start", "gen_end", "post_start", "post_end", "finish_start", "finish_end")]
            self.assertTrue(all(ts), ts); self.assertEqual(ts, sorted(ts)); self.assertRegex(ts[0], r"^\d{4}-\d\d-\d\dT[\d:]+\.\d{3}Z$")
        self.assertEqual(rows(out)[-1]["asset"], "WALL_TOTAL")

    def test_run_order_is_argument_order(self):
        files = [os.path.join(self.imgs, n + ".png") for n in ("c", "a", "b")]
        out = self.out(); self.assertEqual(run(files[0], out, *[]) if False else ab.main(
            ["run", *files, "-o", out, "-m", "m", "--cli", CLI, "--replay", REP, "--no-preflight"]), 0)
        r = {x["asset"]: x["gen_start"] for x in rows(out)[:3]}
        self.assertTrue(r["c"] < r["a"] < r["b"], r)

    def test_cpu_qem_is_a_warning_or_a_failure(self):
        os.environ["FAKE_CPU_QEM"] = "1"
        out = self.out(); self.assertEqual(run(self.imgs, out), 0)
        self.assertEqual({r["warnings"] for r in rows(out)[:3]}, {"post:cpu_qem"})
        self.assertEqual({r["status"] for r in rows(out)[:3]}, {"ok"})
        rep = open(os.path.join(out, "batch_report.md")).read()
        self.assertIn("**Verdict: CHECK", rep); self.assertIn("post:cpu_qem", rep)
        out2 = self.out("o2"); self.assertEqual(run(self.imgs, out2, "--fail-on-cpu-qem"), 1)
        self.assertEqual({r["status"] for r in rows(out2)[:3]}, {"post_failed"})

    def test_report_ok_and_slow_post_hint(self):
        os.environ["FAKE_GEN_S"] = "0.8"; os.environ["FAKE_POST_S"] = "0.1"
        out = self.out(); run(self.imgs, out)
        rep = open(os.path.join(out, "batch_report.md")).read()
        self.assertIn("**Verdict: OK**", rep); self.assertIn("assets/hour", rep); self.assertIn("one post worker keeps up", rep)
        os.environ["FAKE_GEN_S"] = "0.1"; os.environ["FAKE_POST_S"] = "0.6"
        out2 = self.out("o2"); run(self.imgs, out2, "--max-queue", "3")
        self.assertIn("second post worker may help", open(os.path.join(out2, "batch_report.md")).read())

    def test_remesh_budget_auto(self):
        out = self.out(); run(self.imgs, out)
        e = json.load(open(os.path.join(out, "a.part.glb.env.json"))); self.assertEqual(e["TRELLIS_REMESH_MEM_GIB"], "4")
        out2 = self.out("o2"); run(self.imgs, out2, "--remesh-mem-gib", "2.5")
        self.assertEqual(json.load(open(os.path.join(out2, "a.part.glb.env.json")))["TRELLIS_REMESH_MEM_GIB"], "2.5")
        out3 = self.out("o3"); run(self.imgs, out3, "--remesh-mem-gib", "0")
        self.assertIsNone(json.load(open(os.path.join(out3, "a.part.glb.env.json")))["TRELLIS_REMESH_MEM_GIB"])

    def test_finish_oom_retry(self):
        os.environ["FAKE_FINISH_OOM_ONCE"] = os.path.join(self.d, "marker")
        out = self.out(); self.assertEqual(run(self.imgs, out, "--finish-cmd", FIN, "--finish-retry-wait", "0.1"), 0)
        w = [r["warnings"] for r in rows(out)[:3]]
        self.assertEqual(sum("finish:oom_retried" in x for x in w), 1)
        self.assertTrue(os.path.isfile(os.path.join(out, "logs", "a.finish.log.try1")) or
                        any(f.endswith(".try1") for f in os.listdir(os.path.join(out, "logs"))))
        out2 = self.out("o2"); os.remove(os.environ["FAKE_FINISH_OOM_ONCE"])
        self.assertEqual(run(self.imgs, out2, "--finish-cmd", FIN, "--finish-retries", "0"), 1)
        self.assertIn("finish_failed", {r["status"] for r in rows(out2)[:3]})

    def test_preflight(self):
        def go(smi, *extra):
            os.environ["FAKE_SMI"] = smi
            return ab.main(["run", self.imgs, "-o", self.out(), "-m", "m", "--cli", CLI, "--replay", REP,
                            "--nvidia-smi", SMI, *extra])
        with self.assertRaises(SystemExit) as c:
            go("0, 7300\n1, 900")           # ComfyUI resident on the 3060 Ti
        self.assertEqual(c.exception.code, 4)
        with self.assertRaises(SystemExit) as c:
            go("0, 300\n1, 5000")           # something big on the generation card
        self.assertEqual(c.exception.code, 4)
        self.assertFalse(os.path.exists(os.path.join(self.out(), "batch.csv")))
        self.assertEqual(go("0, 300\n1, 900"), 0)
        self.assertEqual(go("0, 7300\n1, 900", "--no-preflight"), 0)
        self.assertEqual(go("0, 7300\n1, 900", "--qem", "cpu"), 0)  # no GPU work on the post card, so not checked
        self.assertEqual(ab.main(["run", self.imgs, "-o", self.out("m"), "-m", "m", "--cli", CLI, "--replay", REP,
                                  "--nvidia-smi", "definitely-not-a-binary"]), 0)  # skipped, not failed

    def gl(self, name="fake_gpulock.py"):
        os.environ["GPULOCK_DIR"] = os.path.join(self.d, "locks")
        return os.path.join(HERE, name) if name == "fake_gpulock.py" else name

    def test_gpulock_hold_and_release(self):
        mod = ab.load_gpulock(self.gl()); out = self.out()
        os.environ["FAKE_GEN_S"] = "0.5"
        seen = {}

        def peek():
            time.sleep(0.25)
            seen["locks"] = sorted(g for g in (0, 1) if mod.read_json(mod.lock_path(g)))
            seen["owner"] = mod.read_json(mod.lock_path(1))["owner"]
            seen["pid"] = mod.read_json(mod.lock_path(1))["pid"]
        import threading
        th = threading.Thread(target=peek); th.start()
        self.assertEqual(run(self.imgs, out, "--gpulock", self.gl(), "--finish-cmd", FIN), 0)
        th.join()
        self.assertEqual(seen["locks"], [0, 1]); self.assertEqual(seen["owner"], "expiramental3d"); self.assertEqual(seen["pid"], os.getpid())
        self.assertEqual([g for g in (0, 1) if mod.read_json(mod.lock_path(g)) or mod.read_json(mod.res_path(g))], [])
        e = json.load(open(os.path.join(out, "finished", "a", "env.json")))
        self.assertEqual(e["ARCTIC_GPU_PHASE_LOCK"], os.path.join(self.d, "locks", "gpu0.phase"))
        self.assertIsNone(gen_args(out, "a")["env"]["ARCTIC_GPU_PHASE_LOCK"])  # never exported to the generation process
        self.assertEqual(json.load(open(os.path.join(out, "a.part.glb.env.json")))["ARCTIC_GPU_PHASE_LOCK"], e["ARCTIC_GPU_PHASE_LOCK"])

    def test_gpulock_refuses_when_busy_and_starts_nothing(self):
        mod = ab.load_gpulock(self.gl()); mod.acquire(0, "space-gothic")
        out = self.out()
        with self.assertRaises(SystemExit) as c:
            run(self.imgs, out, "--gpulock", self.gl())
        self.assertEqual(c.exception.code, 3)
        self.assertFalse(os.path.exists(os.path.join(out, "batch.csv")))
        self.assertIsNone(mod.read_json(mod.lock_path(1)))          # the GPU it did get is given back
        self.assertEqual(mod.read_json(mod.lock_path(0))["owner"], "space-gothic")  # the other tenant's lock untouched

    def test_gpulock_assume_held(self):
        mod = ab.load_gpulock(self.gl())
        with self.assertRaises(SystemExit):
            run(self.imgs, self.out(), "--gpulock", self.gl(), "--gpulock-assume-held")
        for g in (0, 1):
            mod.acquire(g, "expiramental3d")
        self.assertEqual(run(self.imgs, self.out("o2"), "--gpulock", self.gl(), "--gpulock-assume-held"), 0)
        self.assertEqual(sorted(g for g in (0, 1) if mod.read_json(mod.lock_path(g))), [0, 1])  # not released: not ours

    def test_explicit_phase_lock_without_gpulock(self):
        out = self.out(); pl = os.path.join(self.d, "x.phase")
        self.assertEqual(run(self.imgs, out, "--phase-lock", pl), 0)
        self.assertEqual(json.load(open(os.path.join(out, "a.part.glb.env.json")))["ARCTIC_GPU_PHASE_LOCK"], pl)

    @unittest.skipUnless(os.environ.get("GPULOCK_PY") and os.path.isfile(os.environ.get("GPULOCK_PY", "")),
                         "set GPULOCK_PY to Expiramental3d tools/gpulock.py to check the real module")
    def test_real_gpulock(self):
        os.environ["GPULOCK_DIR"] = os.path.join(self.d, "rl"); out = self.out()
        self.assertEqual(run(self.imgs, out, "--gpulock", os.environ["GPULOCK_PY"]), 0)
        mod = ab.load_gpulock(os.environ["GPULOCK_PY"])
        self.assertEqual([g for g in (0, 1) if mod.lock_path(g).exists() or mod.res_path(g).exists()], [])
        mod.acquire(1, "space-gothic")
        with self.assertRaises(SystemExit) as c:
            run(self.imgs, self.out("o2"), "--gpulock", os.environ["GPULOCK_PY"])
        self.assertEqual(c.exception.code, 3)


class Quoting(unittest.TestCase):
    def test_windows_quotes_are_stripped(self):
        self.assertEqual(ab.split_cmd(r'"C:\Program Files\Blender\blender.exe" -b -P "x y.py"', posix=False),
                         [r"C:\Program Files\Blender\blender.exe", "-b", "-P", "x y.py"])
        self.assertEqual(ab.split_cmd('python C:\\a\\b.py {glb}', posix=False), ["python", "C:\\a\\b.py", "{glb}"])

    def test_iso_and_hash_parse(self):
        self.assertEqual(ab.iso(0), ""); self.assertTrue(ab.iso(1.5).endswith(".500Z"))
        h = ab.parse_hashes("  [post-hash] drop_components    V=1 F=2 verts=00ff faces=0a0b area=1\nx")
        self.assertEqual(h["drop_components"], (1, 2, "00ff", "0a0b"))


if __name__ == "__main__":
    unittest.main()
