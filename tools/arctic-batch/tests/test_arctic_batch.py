import json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import arctic_batch as ab

FAKE = os.path.join(HERE, "fake_engine.py")
CLI = '"%s" "%s" cli' % (sys.executable, FAKE)
REP = '"%s" "%s" replay' % (sys.executable, FAKE)


def make_imgs(d, names):
    for n in names:
        open(os.path.join(d, n + ".png"), "wb").write(n.encode())
    return d


def run(imgs, out, *extra):
    return ab.main(["run", imgs, "-o", out, "-m", "m", "--cli", CLI, "--replay", REP, *extra])


def rows(out):
    import csv
    return list(csv.DictReader(open(os.path.join(out, "batch.csv"))))


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
        self.assertEqual(e, {"CUDA_VISIBLE_DEVICES": "0", "TRELLIS_POST_THREADS": "8", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
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


if __name__ == "__main__":
    unittest.main()
