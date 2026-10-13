"""--gen-lanes 2 and dispatcher.py: scheduling rules, VRAM gate, runner behaviour with the fake engine."""
import csv, json, os, sys, tempfile, threading, time, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, HERE)
import dispatcher as dsp
import test_arctic_batch as tb

# item 11 (2026-10-12) measured, seconds on the 4070: name, gen, post, finish
ITEM11 = [("Winch", 378.65, 99.85, 109.53), ("Crystals", 602.46, 109.12, 172.93), ("Bunk", 637.35, 109.42, 188.63),
          ("Chest", 558.54, 117.34, 88.76), ("Statue", 138.92, 84.35, 79.98)]
JOBS = [dict(name=n, gen_s=g, post_s=p, finish_s=f) for n, g, p, f in ITEM11]


def lanes(speed=0.5, cap_b=8.0):
    return [dsp.Lane("A", 1, 1.0, 12.0, 1.4, primary=True), dsp.Lane("B", 0, speed, cap_b, 0.0)]


class Choose(unittest.TestCase):
    def test_primary_takes_queue_order_and_priority(self):
        d = dsp.Dispatcher(lanes())
        pend = [dsp.Job("x", 100), dsp.Job("y", 200, priority=5), dsp.Job("z", 50)]
        self.assertEqual(d.choose(0, d.primary, pend).name, "y")

    def test_secondary_takes_the_job_that_lowers_the_makespan(self):
        d = dsp.Dispatcher(lanes(0.5)); a, b = d.lanes
        a.job, a.est_end = dsp.Job("run", 600), 600.0
        pend = [dsp.Job("big", 600), dsp.Job("small", 140)]
        # small on B: 280 s; A then has 600 + 600; taking big on B: 1200 -> no better than A alone
        self.assertEqual(d.choose(0, b, pend).name, "small")

    def test_tail_guard_no_big_job_on_the_slow_lane_at_the_end(self):
        d = dsp.Dispatcher(lanes(0.4)); a, b = d.lanes
        a.job, a.est_end = dsp.Job("run", 100), 100.0
        self.assertIsNone(d.choose(0, b, [dsp.Job("last", 600)]))   # B: 1500 s vs A: 700 s

    def test_vram_fit_and_pin(self):
        d = dsp.Dispatcher(lanes(0.5)); a, b = d.lanes
        a.job, a.est_end = dsp.Job("run", 900), 900.0
        cabin = dsp.Job("cabin", 300, vram_gb=8.9)
        self.assertIsNone(d.choose(0, b, [cabin]))                       # 8.9 GB does not fit an 8 GB card
        self.assertEqual(d.choose(0, d.primary, [cabin]).name, "cabin")
        pinned = dsp.Job("p", None, pin="B")
        self.assertEqual(d.choose(0, b, [dsp.Job("q", 100), pinned]).name, "p")
        self.assertIsNone(d.choose(0, d.primary, [pinned]))              # pinned to B: A leaves it alone

    def test_unknown_sizes_split_the_queue(self):
        d = dsp.Dispatcher(lanes(0.5), min_gain_s=0)
        pend = [dsp.Job("a"), dsp.Job("b"), dsp.Job("c")]
        self.assertIsNotNone(d.choose(0, d.lanes[1], pend))

    def test_observe_learns_the_speed(self):
        d = dsp.Dispatcher(lanes(0.5)); b = d.lanes[1]
        d.observe(b, dsp.Job("x", 300), 1000.0)
        self.assertAlmostEqual(b.speed, 0.3)


class LongestFirst(unittest.TestCase):
    def test_order_slowest_first_unknown_is_median_and_stable(self):
        items = [("pa", "a"), ("pb", "b"), ("pc", "c"), ("pd", "d"), ("pe", "e")]
        got = tb.ab.longest_first(items, {"a": 100, "b": 600, "d": 300, "e": 600})
        self.assertEqual([n for _, n in got], ["b", "e", "c", "d", "a"])   # c unknown = median 450; b/e tie keeps queue order
        self.assertEqual(tb.ab.longest_first(items, {}), items)

    def test_runner_order_is_opt_in_and_outputs_are_hash_equal(self):
        import contextlib, io
        with tempfile.TemporaryDirectory() as d:
            imgs = tb.make_imgs(os.makedirs(os.path.join(d, "i")) or os.path.join(d, "i"), list("abcd"))
            os.environ["FAKE_GEN_S"] = "0.1"; os.environ["FAKE_POST_S"] = "0.1"
            base, lf = os.path.join(d, "base"), os.path.join(d, "lf")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(tb.run(imgs, base), 0)
            self.assertNotIn("order: longest first", buf.getvalue())
            est = os.path.join(d, "est.csv")
            with open(est, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["asset", "status", "gen_s"])
                for n, g in (("a", 10), ("b", 20), ("c", 40), ("d", 30)):
                    w.writerow([n, "ok", g])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(tb.run(imgs, lf, "--order", "longest-first", "--est-from", est), 0)
            self.assertIn("order: longest first -> c, d, b, a", buf.getvalue())
            self.assertEqual([x["asset"] for x in tb.rows(lf)[:-1]], list("abcd"))   # csv keeps the given order
            self.assertEqual(tb.ab.cmd_compare(type("A", (), {"dir_a": base, "dir_b": lf})), 0)


class PriorityOrder(unittest.TestCase):
    def test_priority_then_arrival_or_longest_first(self):
        pend = [dsp.Job("a", 100), dsp.Job("b", 300), dsp.Job("c", 200, priority=100), dsp.Job("d", 50, priority=100), dsp.Job("e")]
        d = dsp.Dispatcher(lanes())
        self.assertEqual([j.name for j in d.order(pend)], ["c", "d", "a", "b", "e"])
        d = dsp.Dispatcher(lanes(), longest_first=True)
        self.assertEqual([j.name for j in d.order(pend)], ["c", "d", "b", "e", "a"])   # e: no estimate = median 150 -> between b and a


class LiveQueueTests(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = self.t.name
        self.q = os.path.join(self.d, "q")
        os.environ["FAKE_GEN_S"] = "0.6"; os.environ["FAKE_POST_S"] = "0.2"

    def tearDown(self):
        self.t.cleanup()

    def test_poll_claims_once_and_rejects_bad_jobs(self):
        import live_queue as lq
        img = tb.make_imgs(os.makedirs(os.path.join(self.d, "i")) or os.path.join(self.d, "i"), ["x", "y"])
        lq.submit(self.q, os.path.join(img, "x.png"), priority=7, pin="B")
        lq.submit(self.q, os.path.join(img, "y.png"), name="X")                    # duplicate of x (case-insensitive)
        lq.submit(self.q, os.path.join(img, "nope.png"))                             # missing image
        open(os.path.join(self.q, "0-bad.job.json"), "w").write("{not json")
        open(os.path.join(self.q, "partial.job.json.tmp"), "w").write("{}")          # a half-written file is ignored
        got = lq.LiveQueue(self.q, lambda m: None).poll([])
        self.assertEqual([(g["name"], g["priority"], g["pin"]) for g in got], [("x", 7, "B")])
        self.assertEqual(len(os.listdir(os.path.join(self.q, "rejected"))), 6)     # 3 files + 3 .why.txt
        self.assertEqual(lq.LiveQueue(self.q, lambda m: None).poll(["x"]), [])      # nothing is taken twice
        self.assertFalse(lq.LiveQueue(self.q, lambda m: None).is_closed()); lq.close(self.q)
        self.assertTrue(lq.LiveQueue(self.q, lambda m: None).is_closed())

    def test_jobs_added_while_running_urgent_jumps_the_queue_and_output_is_hash_equal(self):
        import live_queue as lq
        base_in = tb.make_imgs(os.makedirs(os.path.join(self.d, "all")) or os.path.join(self.d, "all"), list("abcdeu"))
        first = os.path.join(self.d, "first"); os.makedirs(first)
        for n in "abcd":
            os.replace(os.path.join(base_in, n + ".png"), os.path.join(first, n + ".png"))
        os.replace(os.path.join(first, "a.png"), os.path.join(base_in, "a.png"))   # all/ = a,e,u ; first/ = b,c,d
        # reference: the plain batch of the same six pictures
        ref_in = tb.make_imgs(os.makedirs(os.path.join(self.d, "ref")) or os.path.join(self.d, "ref"), list("abcdeu"))
        ref, live = os.path.join(self.d, "ref_out"), os.path.join(self.d, "live_out")
        self.assertEqual(tb.run(ref_in, ref, "--gen-lanes", "2", "--lane2-speed", "1.0", "--min-gain-s", "0"), 0)
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("rc", tb.run(first, live, "--gen-lanes", "2", "--lane2-speed", "1.0",
                                                                        "--min-gain-s", "0", "--queue-dir", self.q)))
        th.start(); time.sleep(0.4)   # b and c are generating, d is waiting
        lq.submit(self.q, os.path.join(base_in, "e.png"), priority=0)
        lq.submit(self.q, os.path.join(base_in, "u.png"), priority=100)
        lq.submit(self.q, os.path.join(base_in, "a.png"), priority=0)
        time.sleep(0.3); lq.close(self.q)
        th.join(120); self.assertFalse(th.is_alive())
        self.assertEqual(res["rc"], 0)
        rows = {x["asset"]: x for x in tb.rows(live)[:-1]}
        self.assertEqual(sorted(rows), list("abcdeu")); self.assertTrue(all(r["status"] == "ok" for r in rows.values()))
        start = {n: r["gen_start"] for n, r in rows.items()}
        self.assertLess(start["u"], start["e"]); self.assertLess(start["u"], start["a"])   # urgent beat the earlier normal jobs
        self.assertEqual(tb.ab.cmd_compare(type("A", (), {"dir_a": ref, "dir_b": live})), 0)

    def test_per_job_options_reach_only_that_job(self):
        import live_queue as lq
        img = tb.make_imgs(os.makedirs(os.path.join(self.d, "i")) or os.path.join(self.d, "i"), ["plain", "draft"])
        out = os.path.join(self.d, "o")
        lq.submit(self.q, os.path.join(img, "draft.png"), priority=0, opts={"res": 512, "seed": 9, "faces": 150000,
                                                                          "cli_extra": "--steps 12 --steps-ss 24"})
        first = os.path.join(self.d, "first"); os.makedirs(first)
        os.replace(os.path.join(img, "plain.png"), os.path.join(first, "plain.png"))
        lq.close(self.q)
        self.assertEqual(tb.run(first, out, "--gen-lanes", "2", "--lane2-speed", "1.0", "--min-gain-s", "0",
                                "--queue-dir", self.q, "--res", "1024", "--seed", "1", "--faces", "400000"), 0)
        def argv(name, kind):
            for line in open(os.path.join(out, "logs", "%s.%s.log" % (name, kind)), errors="replace"):
                if line.startswith("ARGS "):
                    j = json.loads(line[5:]); return j["argv"] if isinstance(j, dict) else j
        g, pl = argv("draft", "gen"), argv("plain", "gen")
        self.assertEqual((g[g.index("--res") + 1], g[g.index("--seed") + 1], g[-4:]), ("512", "9", ["--steps", "12", "--steps-ss", "24"]))
        self.assertEqual((pl[pl.index("--res") + 1], pl[pl.index("--seed") + 1], "--steps" in pl), ("1024", "1", False))
        pd, pp = argv("draft", "post"), argv("plain", "post")
        self.assertEqual((pd[pd.index("--geometry-res") + 1], pd[pd.index("--faces") + 1]), ("512", "150000"))
        self.assertEqual((pp[pp.index("--geometry-res") + 1], pp[pp.index("--faces") + 1]), ("1024", "400000"))

    def test_batch_pin_applies_to_live_jobs(self):
        import live_queue as lq
        img = tb.make_imgs(os.makedirs(os.path.join(self.d, "i")) or os.path.join(self.d, "i"), ["a", "b", "c"])
        first = os.path.join(self.d, "first"); os.makedirs(first)
        os.replace(os.path.join(img, "a.png"), os.path.join(first, "a.png"))
        lq.submit(self.q, os.path.join(img, "b.png")); lq.submit(self.q, os.path.join(img, "c.png")); lq.close(self.q)
        out = os.path.join(self.d, "o")
        self.assertEqual(tb.run(first, out, "--gen-lanes", "2", "--lane2-speed", "1.0", "--min-gain-s", "0", "--queue-dir", self.q,
                                "--pin", "b=A", "--pin", "c=A"), 0)
        r = {x["asset"]: x for x in tb.rows(out)[:-1]}
        self.assertEqual((r["b"]["gen_lane"], r["c"]["gen_lane"]), ("A", "A"))

    def test_bad_job_options_are_refused(self):
        import live_queue as lq
        img = tb.make_imgs(os.makedirs(os.path.join(self.d, "i")) or os.path.join(self.d, "i"), ["x"])
        for bad in ({"ress": 512}, {"res": "big"}, {"res": 5}, {"cli_extra": 3}):
            with self.assertRaises(ValueError):
                lq.submit(self.q, os.path.join(img, "x.png"), opts=bad)

    def test_idle_exit_and_validation(self):
        import contextlib, io
        os.makedirs(self.q)
        t0 = time.time()
        self.assertEqual(tb.run(self._one(), os.path.join(self.d, "o"), "--gen-lanes", "2",
                                "--lane2-speed", "1.0", "--queue-dir", self.q, "--idle-exit", "3"), 0)
        self.assertGreaterEqual(time.time() - t0, 3.0)
        with self.assertRaises(SystemExit):
            tb.run(self._one(), os.path.join(self.d, "o2"), "--queue-dir", self.q)   # needs --gen-lanes 2

    def _one(self):
        d = os.path.join(self.d, "one"); os.makedirs(d, exist_ok=True)
        return tb.make_imgs(d, ["z"])


class Gate(unittest.TestCase):
    def test_admission_and_release(self):
        g = dsp.VramGate({0: 8.0}, {0: 0.0}, margin_gb=0.4)
        self.assertTrue(g.admit(0, "gen", 4.5))
        self.assertTrue(g.can_admit(0, 3.0))          # 4.5 + 3.0 + 0.4 = 7.9 <= 8
        self.assertFalse(g.can_admit(0, 3.2))
        g.release(0, "gen")
        self.assertTrue(g.can_admit(0, 7.5))

    def test_shared_slot_counts_once(self):
        g = dsp.VramGate({0: 8.0}, margin_gb=0.4)
        self.assertTrue(g.acquire_shared(0, "phase", 3.0))
        self.assertTrue(g.acquire_shared(0, "phase", 3.0))      # a second holder joins; no second 3 GB
        self.assertTrue(g.can_admit(0, 4.5))
        g.release_shared(0, "phase"); self.assertFalse(g.can_admit(0, 5.0))
        g.release_shared(0, "phase"); self.assertTrue(g.can_admit(0, 5.0))

    def test_wait_admit_unblocks_on_release(self):
        g = dsp.VramGate({0: 8.0}, margin_gb=0.4)
        g.admit(0, "a", 6.0); got = []
        t = threading.Thread(target=lambda: got.append(g.wait_admit(0, "b", 4.5, timeout=10, poll=0.1))); t.start()
        time.sleep(0.3); self.assertEqual(got, []); g.release(0, "a"); t.join(5)
        self.assertEqual(got, [True])
        stop = threading.Event(); stop.set()
        self.assertFalse(g.wait_admit(0, "c", 4.5, stop=stop))


class Sim(unittest.TestCase):
    def test_one_lane_reproduces_item11(self):
        w, _ = dsp.simulate(JOBS, 1)
        self.assertAlmostEqual(w, 2480.0, delta=2)          # measured wall 2,480 s

    def test_two_lanes_gain_and_strict_costs(self):
        base, _ = dsp.simulate(JOBS, 1)
        w_vram, _ = dsp.simulate(JOBS, 2, 0.5, placement="vram")
        w_strict, _ = dsp.simulate(JOBS, 2, 0.5, placement="strict")
        self.assertLess(w_vram, base * 0.85)
        self.assertLess(w_vram, w_strict)
        self.assertLessEqual(w_strict, base)

    def test_threshold(self):
        base, th = dsp.threshold(JOBS, 0.10)
        self.assertLessEqual(th["vram"], 0.35); self.assertGreater(th["strict"], th["vram"])


class Runner(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory(); self.d = self.t.name
        os.makedirs(os.path.join(self.d, "i")); self.imgs = tb.make_imgs(os.path.join(self.d, "i"), list("abcdef"))
        os.environ["FAKE_GEN_S"] = "0.6"; os.environ["FAKE_POST_S"] = "0.2"

    def tearDown(self):
        os.environ.pop("FAKE_LANE_B_OOM", None); self.t.cleanup()

    def lanes_run(self, out, *extra):
        return tb.run(self.imgs, out, "--gen-lanes", "2", "--lane2-speed", "1.0", "--min-gain-s", "0", *extra)

    def test_two_lanes_are_faster_and_lane_b_is_hidden_from_the_4070(self):
        one, two = os.path.join(self.d, "one"), os.path.join(self.d, "two")
        self.assertEqual(tb.run(self.imgs, one), 0)
        self.assertEqual(self.lanes_run(two), 0)
        r1, r2 = tb.rows(one), tb.rows(two)
        self.assertLess(float(r2[-1]["asset_total_s"]), float(r1[-1]["asset_total_s"]) * 0.8)
        lanes = {x["asset"]: x["gen_lane"] for x in r2[:-1]}
        self.assertEqual(set(lanes.values()), {"A", "B"})
        for n, ln in lanes.items():
            g = tb.gen_args(two, n)
            if ln == "B":
                self.assertEqual(g["env"]["CUDA_VISIBLE_DEVICES"], "0"); self.assertEqual(g["argv"][g["argv"].index("--gpu") + 1], "0")
            else:
                self.assertIsNone(g["env"]["CUDA_VISIBLE_DEVICES"]); self.assertEqual(g["argv"][g["argv"].index("--gpu") + 1], "1")
        # same output whichever lane generated it (the fake engine is deterministic)
        self.assertEqual(tb.ab.cmd_compare(type("A", (), {"dir_a": one, "dir_b": two})), 0)

    def test_lane2_env_set_and_unset_only_on_lane_b(self):
        os.environ["TRELLIS_CONVNEXT_STREAM_ALL"] = "1"
        try:
            out = os.path.join(self.d, "env")
            self.assertEqual(self.lanes_run(out, "--lane2-unset", "TRELLIS_CONVNEXT_STREAM_ALL", "--lane2-env", "X_TEST=1"), 0)
            seen = {x["asset"]: x["gen_lane"] for x in tb.rows(out)[:-1]}
            self.assertEqual(set(seen.values()), {"A", "B"})
            for n, ln in seen.items():   # the fake prints only its ENV_KEYS; the cmd line proves the lane
                env = tb.gen_args(out, n)["env"]
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0" if ln == "B" else None)
                self.assertEqual(env["TRELLIS_CONVNEXT_STREAM_ALL"], None if ln == "B" else "1")
                self.assertEqual(env["X_TEST"], "1" if ln == "B" else None)
        finally:
            os.environ.pop("TRELLIS_CONVNEXT_STREAM_ALL", None)

    def test_default_is_unchanged(self):
        out = os.path.join(self.d, "o"); self.assertEqual(tb.run(self.imgs, out), 0)
        self.assertEqual({x["gen_lane"] for x in tb.rows(out)[:-1]}, {""})

    def test_oom_on_lane_b_requeues_on_lane_a(self):
        os.environ["FAKE_LANE_B_OOM"] = "1"
        out = os.path.join(self.d, "oom"); self.assertEqual(self.lanes_run(out), 0)
        r = tb.rows(out)[:-1]
        self.assertEqual([x["status"] for x in r], ["ok"] * 6)
        self.assertEqual({x["gen_lane"] for x in r}, {"A"})

    def test_pin_and_vram_keep_a_big_job_off_the_3060_ti(self):
        out = os.path.join(self.d, "pin")
        self.assertEqual(self.lanes_run(out, "--job-vram", "a=8.9", "--job-vram", "b=8.9", "--job-vram", "c=8.9"), 0)
        lanes = {x["asset"]: x["gen_lane"] for x in tb.rows(out)[:-1]}
        self.assertTrue(all(lanes[n] == "A" for n in "abc"))

    def test_post_waits_for_vram_beside_a_lane_b_generation(self):
        out = os.path.join(self.d, "gate")
        # 7.0 GB leaves room for a generation (4.5) or a post (3.0) on the 3060 Ti but not both: post/finish wait, nothing fails
        self.assertEqual(self.lanes_run(out, "--gpu-cap-gb", "0=7.0,1=12", "--vram-margin-gb", "0.0"), 0)
        r = tb.rows(out)[:-1]
        self.assertEqual([x["status"] for x in r], ["ok"] * 6)
        self.assertTrue(any(float(x["gate_wait_s"] or 0) > 0.1 for x in r))
        self.assertTrue(all(x["phase_gpu"] == "0" for x in r))

    def test_lane_flag_validation(self):
        with self.assertRaises(SystemExit):
            tb.run(self.imgs, os.path.join(self.d, "x"), "--gen-lanes", "2", "--lane2-gpu", "1")


if __name__ == "__main__":
    unittest.main()


class DraftRefine(unittest.TestCase):
    """tools/arctic-batch/draft_refine.py with the fake engine and fake judge: drafts carry draft flags, only keepers are refined,
    refine output is hash-equal to a plain batch of the same pictures."""
    def test_funnel(self):
        import draft_refine as dr
        t = tempfile.TemporaryDirectory(); d = t.name
        os.environ["FAKE_GEN_S"] = "0.3"; os.environ["FAKE_POST_S"] = "0.1"
        imgs = tb.make_imgs(os.makedirs(os.path.join(d, "i")) or os.path.join(d, "i"), ["c1", "c2", "c3", "v1", "v2"])
        man = os.path.join(d, "props.json")
        json.dump({"chest": [os.path.join(imgs, "c1.png"), os.path.join(imgs, "c2.png"), {"image": os.path.join(imgs, "c3.png"), "seed": 5}],
                   "veto": [os.path.join(imgs, "v1.png"), os.path.join(imgs, "v2.png")]}, open(man, "w"))
        seed = os.path.join(d, "seed"); os.makedirs(seed)
        os.replace(os.path.join(imgs, "c1.png"), os.path.join(seed, "seedimg.png"))
        import shutil; shutil.copy(os.path.join(seed, "seedimg.png"), os.path.join(imgs, "c1.png"))
        q, out = os.path.join(d, "q"), os.path.join(d, "out")
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("rc", tb.run(seed, out, "--gen-lanes", "2", "--lane2-speed", "1.0",
                                                                         "--min-gain-s", "0", "--queue-dir", q, "--res", "1024",
                                                                         "--finish-cmd", "%s %s {glb} {name} {out}" % (sys.executable, os.path.join(HERE, "fake_engine.py") + " finish"))))
        th.start(); time.sleep(0.3)
        judge = "%s %s {glb}" % (sys.executable, os.path.join(HERE, "fake_judge.py"))
        args = type("A", (), dict(queue_dir=q, out=out, manifest=man, judge_cmd=judge, keep=1, draft_res=None, draft_cli_extra=None,
                                  draft_est_s=1, draft_vram_gb=None, refine_est_s=1, draft_timeout=60, close=True, plain=False))
        rep = dr.run(args, log=lambda m: None)
        th.join(120); self.assertFalse(th.is_alive()); self.assertEqual(res["rc"], 0)
        self.assertEqual(rep["props"]["chest"]["keepers"], ["chest_3"])         # highest score, one keeper
        self.assertEqual(rep["props"]["veto"]["keepers"], [])                    # the judge vetoed both drafts
        names = {r["asset"] for r in tb.rows(out)[:-1]}
        self.assertEqual(names, {"seedimg", "chest_1_d", "chest_2_d", "chest_3_d", "veto_1_d", "veto_2_d", "chest_3"})
        fin = sorted(os.listdir(os.path.join(out, "finished")))
        self.assertEqual(fin, ["chest_3", "seedimg"])                           # drafts are never finished
        def gen_argv(n):
            for l in open(os.path.join(out, "logs", n + ".gen.log"), errors="replace"):
                if l.startswith("ARGS "): return json.loads(l[5:])["argv"]
        dv, rv = gen_argv("chest_3_d"), gen_argv("chest_3")
        self.assertEqual(dv[dv.index("--res") + 1], "512"); self.assertIn("--steps-ss", dv)
        self.assertEqual(rv[rv.index("--res") + 1], "1024"); self.assertNotIn("--steps-ss", rv)
        self.assertEqual((dv[dv.index("--seed") + 1], rv[rv.index("--seed") + 1]), ("5", "5"))   # the candidate's own seed, draft and refine
        # refine == plain run of the same picture
        plain_in = os.path.join(d, "plain"); os.makedirs(plain_in); shutil.copy(os.path.join(imgs, "c3.png"), os.path.join(plain_in, "chest_3.png"))
        plain = os.path.join(d, "plain_out")
        self.assertEqual(tb.run(plain_in, plain, "--res", "1024", "--seed", "5"), 0)
        self.assertEqual(tb.ab.masked_glb_hash(os.path.join(out, "chest_3.glb")), tb.ab.masked_glb_hash(os.path.join(plain, "chest_3.glb")))
        t.cleanup()

    def test_plain_arm_submits_everything_as_full_jobs(self):
        import draft_refine as dr
        t = tempfile.TemporaryDirectory(); d = t.name
        imgs = tb.make_imgs(os.makedirs(os.path.join(d, "i")) or os.path.join(d, "i"), ["a", "b"])
        man = os.path.join(d, "props.json")
        json.dump({"p": [os.path.join(imgs, "a.png"), {"image": os.path.join(imgs, "b.png"), "seed": 3}]}, open(man, "w"))
        q = os.path.join(d, "q")
        args = type("A", (), dict(queue_dir=q, out=os.path.join(d, "o"), manifest=man, judge_cmd="", keep=1, draft_res=None,
                                  draft_cli_extra=None, draft_est_s=1, draft_vram_gb=None, refine_est_s=1, draft_timeout=1, close=True, plain=True))
        rep = dr.run(args, log=lambda m: None)
        self.assertEqual(rep["props"]["p"]["keepers"], ["p_1", "p_2"])
        got = {g["name"]: g for g in __import__("live_queue").LiveQueue(q, lambda m: None).poll([])}
        self.assertEqual(sorted(got), ["p_1", "p_2"]); self.assertEqual(got["p_2"]["opts"], {"seed": 3}); self.assertEqual(got["p_1"]["opts"], {})
        t.cleanup()


class SizeJudge(unittest.TestCase):
    def _glb(self, path, mn, mx):
        js = json.dumps({"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
                         "accessors": [{"min": mn, "max": mx}]}).encode()
        js += b" " * (-len(js) % 4)
        import struct
        open(path, "wb").write(b"glTF" + struct.pack("<II", 12 + 8 + len(js), 2) + struct.pack("<II", len(js), 0x4E4F534A) + js)

    def test_score_and_veto(self):
        import size_judge as sj, contextlib, io
        t = tempfile.TemporaryDirectory(); g = os.path.join(t.name, "a.glb")
        self._glb(g, [0, 0, 0], [1.3, 0.85, 0.8])                        # chest, axes in another order
        self.assertAlmostEqual(sj.deviation(sj.glb_bounds(g), [1.3, 0.8, 0.85]), 0.0, places=6)
        self._glb(g, [0, 0, 0], [1.3, 0.4, 0.85])                        # half as deep
        with contextlib.redirect_stdout(io.StringIO()) as o:
            rc = sj.main([g, "--spec", "1.3,0.8,0.85"])
        self.assertEqual(rc, 4); self.assertNotIn("-0", o.getvalue().splitlines()[-1][:2])
        t.cleanup()
