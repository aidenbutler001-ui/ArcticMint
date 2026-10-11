#!/usr/bin/env python3
"""arctic-batch: overlap GPU generation with CPU post-processing (stdlib only, Windows ok).

GPU worker : trellis-cli ... --dump-post <dump>   (back to back, on the generation GPU)
Post pool  : post-replay <dump> <glb> ...          (N workers, TRELLIS_POST_THREADS = total/N each)
--serial   : same commands, one asset at a time, post gets all threads (today's behaviour).

Subcommands: run (default), compare (masked-GLB hash compare of two output dirs).
"""
import argparse, csv, hashlib, os, queue, re, shlex, subprocess, sys, threading, time

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
CSV_FIELDS = ["asset", "status", "gen_s", "queue_wait_s", "post_s", "asset_total_s",
              "dump_mb", "glb_sha256_masked", "error"]


def split_cmd(s):
    return shlex.split(s, posix=(os.name != "nt")) if isinstance(s, str) else list(s)


def masked_glb_hash(path):
    """SHA-256 of a GLB with the volatile "generated" timestamp masked (same rule as pc-verify.ps1)."""
    with open(path, "rb") as f:
        data = f.read()
    data = re.sub(rb'("generated"\s*:\s*")[^"]*(")', lambda m: m.group(1) + m.group(2), data)
    return hashlib.sha256(data).hexdigest()


def collect_images(items):
    out = []
    for it in items:
        if os.path.isdir(it):
            out += sorted(os.path.join(it, n) for n in os.listdir(it)
                          if os.path.splitext(n)[1].lower() in IMG_EXT)
        elif os.path.isfile(it):
            out.append(it)
        else:
            raise SystemExit("image not found: " + it)
    if not out:
        raise SystemExit("no images")
    return out


class Batch:
    def __init__(self, a):
        self.a = a
        self.out = a.out
        self.dumps = os.path.join(self.out, "dumps")
        self.logs = os.path.join(self.out, "logs")
        self.cli = split_cmd(a.cli)
        self.replay = split_cmd(a.replay)
        for d in (self.out, self.dumps, self.logs):
            os.makedirs(d, exist_ok=True)
        self.rows = {}
        self.lock = threading.Lock()

    # -- environments -------------------------------------------------------
    def gen_env(self):
        e = dict(os.environ)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        return e

    def post_env(self, threads):
        e = dict(os.environ)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        e["TRELLIS_POST_THREADS"] = str(threads)
        # post-replay never runs ggml: without this its GPU QEM lands on CUDA device 0.
        # "-1" = no visible device -> decimate_qem falls back to the CPU path.
        e["CUDA_VISIBLE_DEVICES"] = "-1" if self.a.qem == "cpu" else str(self.a.post_gpu)
        return e

    # -- commands -----------------------------------------------------------
    def gen_cmd(self, img, name, dump):
        a = self.a
        cmd = self.cli + ["-i", img, "-o", os.path.join(self.dumps, name + ".unused.glb"),
                          "-m", a.models, "--res", str(a.res), "--gpu", str(a.gen_gpu),
                          "--seed", str(a.seed), "--dump-post", dump]
        return cmd + split_cmd(a.cli_extra)

    def post_cmd(self, dump, glb):
        a = self.a
        band = max(1, a.res // 512)
        return self.replay + [dump, glb, "--geometry-res", str(a.res), "--band", str(band),
                              "--faces", str(a.faces), "--no-audit"] + split_cmd(a.replay_extra)

    def run_logged(self, cmd, env, log):
        with open(log, "wb") as lf:
            lf.write((" ".join(cmd) + "\n").encode())
            lf.flush()
            return subprocess.call(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    # -- stages ---------------------------------------------------------------
    def do_gen(self, img, name):
        dump = os.path.join(self.dumps, name + ".bin")
        t0 = time.time()
        rc = self.run_logged(self.gen_cmd(img, name, dump), self.gen_env(),
                             os.path.join(self.logs, name + ".gen.log"))
        dt = time.time() - t0
        if rc != 0 or not os.path.isfile(dump):
            self.fail(name, "gen", dt, "trellis-cli exit %s" % rc)
            return None
        self.rows[name] = {"asset": name, "gen_s": dt, "t_gen_end": time.time(),
                           "dump_mb": os.path.getsize(dump) / 1048576.0}
        return dump

    def do_post(self, name, dump, threads):
        row = self.rows[name]
        t_start = time.time()
        row["queue_wait_s"] = t_start - row["t_gen_end"]
        glb = os.path.join(self.out, name + ".glb")
        rc = self.run_logged(self.post_cmd(dump, glb), self.post_env(threads),
                             os.path.join(self.logs, name + ".post.log"))
        row["post_s"] = time.time() - t_start
        if rc != 0 or not os.path.isfile(glb):
            row["status"], row["error"] = "post_failed", "post-replay exit %s" % rc
        else:
            row["status"] = "ok"
            row["glb_sha256_masked"] = masked_glb_hash(glb)
        row["asset_total_s"] = row["gen_s"] + row["queue_wait_s"] + row["post_s"]
        if not self.a.keep_dumps and row["status"] == "ok":
            try:
                os.remove(dump)
            except OSError:
                pass
        self.log("%s: gen %.1fs, wait %.1fs, post %.1fs -> %s" %
                 (name, row["gen_s"], row["queue_wait_s"], row["post_s"], row["status"]))

    def fail(self, name, stage, dt, err):
        self.rows[name] = {"asset": name, "status": stage + "_failed", "gen_s": dt, "error": err}
        self.log("%s: %s FAILED (%s)" % (name, stage, err))

    def log(self, msg):
        with self.lock:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    # -- drivers --------------------------------------------------------------
    def run_serial(self, items):
        for img, name in items:
            dump = self.do_gen(img, name)
            if dump:
                self.do_post(name, dump, self.a.total_threads)

    def run_overlap(self, items):
        q = queue.Queue(maxsize=self.a.max_queue)  # bounds dumps on disk (0.3-0.6 GB each)
        n, threads = self.a.post_workers, max(1, self.a.total_threads // self.a.post_workers)

        def worker():
            while True:
                job = q.get()
                if job is None:
                    return
                try:
                    self.do_post(job[0], job[1], threads)
                except Exception as e:  # keep the pool alive
                    self.rows[job[0]].update(status="post_failed", error=repr(e))
                    self.log("%s: post error %r" % (job[0], e))

        ws = [threading.Thread(target=worker) for _ in range(n)]
        for w in ws:
            w.start()
        for img, name in items:
            dump = self.do_gen(img, name)
            if dump:
                self.rows[name]["t_gen_end"] = time.time()
                q.put((name, dump))  # blocks when the post pool is max_queue behind
        for _ in ws:
            q.put(None)
        for w in ws:
            w.join()

    def write_csv(self, order, wall):
        path = os.path.join(self.out, "batch.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, CSV_FIELDS)
            w.writeheader()
            for name in order:
                r = self.rows.get(name, {"asset": name, "status": "not_run"})
                w.writerow({k: ("%.2f" % r[k] if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in CSV_FIELDS})
            ok = sum(1 for r in self.rows.values() if r.get("status") == "ok")
            w.writerow({"asset": "WALL_TOTAL", "status": "%d/%d ok" % (ok, len(order)),
                        "asset_total_s": "%.2f" % wall,
                        "error": "mode=%s assets_per_hour=%.2f" %
                                 ("serial" if self.a.serial else "overlap", ok * 3600.0 / wall if wall else 0)})
        return path


def cmd_run(a):
    imgs = collect_images(a.images)
    names, items = set(), []
    for p in imgs:
        n = os.path.splitext(os.path.basename(p))[0]
        if n in names:
            raise SystemExit("duplicate asset name: " + n)
        names.add(n)
        items.append((p, n))
    b = Batch(a)
    t0 = time.time()
    (b.run_serial if a.serial else b.run_overlap)(items)
    wall = time.time() - t0
    path = b.write_csv([n for _, n in items], wall)
    b.log("wall %.1fs; csv %s" % (wall, path))
    return 0 if all(r.get("status") == "ok" for r in b.rows.values()) and len(b.rows) == len(items) else 1


def cmd_compare(a):
    bad = 0
    for name in sorted(n for n in os.listdir(a.dir_a) if n.endswith(".glb") and not n.endswith(".unused.glb")):
        pb = os.path.join(a.dir_b, name)
        if not os.path.isfile(pb):
            print("MISSING in B: " + name); bad += 1; continue
        ha, hb = masked_glb_hash(os.path.join(a.dir_a, name)), masked_glb_hash(pb)
        print("%s %s %s" % ("PASS" if ha == hb else "FAIL", name, ha[:16] if ha == hb else ha[:16] + " != " + hb[:16]))
        bad += ha != hb
    return 1 if bad else 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    r = sub.add_parser("run", help="run a batch")
    r.add_argument("images", nargs="+", help="image files and/or directories")
    r.add_argument("-o", "--out", required=True)
    r.add_argument("-m", "--models", default=r"C:\AI\trellis\models")
    r.add_argument("--cli", default="trellis-cli", help="trellis-cli command (path or 'python fake.py')")
    r.add_argument("--replay", default="post-replay")
    r.add_argument("--cli-extra", default="", help="extra args for trellis-cli")
    r.add_argument("--replay-extra", default="", help="extra args for post-replay")
    r.add_argument("--res", type=int, default=1024)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--faces", type=int, default=300000)
    r.add_argument("--gen-gpu", type=int, default=1, help="PCI index of the generation GPU (4070 SUPER = 1)")
    r.add_argument("--post-gpu", type=int, default=0, help="PCI index for post GPU QEM (3060 Ti = 0)")
    r.add_argument("--qem", choices=["gpu", "cpu"], default="gpu",
                   help="gpu = QEM on --post-gpu; cpu = hide all GPUs from post-replay (CPU QEM, slower)")
    r.add_argument("--post-workers", type=int, default=2)
    r.add_argument("--total-threads", type=int, default=24, help="threads/worker = total/workers")
    r.add_argument("--max-queue", type=int, default=2, help="finished dumps allowed to wait for post")
    r.add_argument("--serial", action="store_true", help="gen then post, one asset at a time (A/B baseline)")
    r.add_argument("--keep-dumps", action="store_true")
    c = sub.add_parser("compare", help="compare GLBs of two output dirs (timestamp masked)")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv.insert(0, "run")
    a = build_parser().parse_args(argv)
    if a.cmd == "compare":
        return cmd_compare(a)
    if a.cmd != "run":
        build_parser().print_help()
        return 2
    if a.post_workers < 1 or a.max_queue < 1:
        raise SystemExit("--post-workers and --max-queue must be >= 1")
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
