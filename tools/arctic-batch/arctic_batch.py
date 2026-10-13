#!/usr/bin/env python3
"""arctic-batch: overlap GPU generation with CPU post-processing (stdlib only, Windows ok).

GPU worker : trellis-cli ... --dump-post <dump>   (back to back, on the generation GPU)
             --dump-mode raw: TRELLIS_DUMP_POST=<dump> + TRELLIS_STOP_AFTER_POST_DUMP=1 instead (exits right after the
             texture decode; post-replay then runs the serial CLI's exact weld..drop chain)
Post pool  : post-replay <dump> <glb> ...          (N workers, TRELLIS_POST_THREADS = total/N each)
--serial   : same commands, one asset at a time, post gets all threads (today's behaviour).
--gpulock  : hold the cross-project GPU locks (tools/gpulock.py of Expiramental3d) for the whole batch.
--phase-lock / ARCTIC_GPU_PHASE_LOCK : serialise the 3060 Ti's QEM (post-replay) and Cycles bake (finish.py).
--gen-lanes 2 : OPT-IN second generation lane on the post GPU (3060 Ti) with a VRAM-aware dispatcher (dispatcher.py): jobs go to
             the lane that lowers the projected makespan, post/finish are admitted by declared VRAM. Default 1 = unchanged.

Subcommands: run (default), compare (masked-GLB hash compare of two output dirs).
"""
import argparse, csv, hashlib, importlib.util, os, queue, re, shlex, subprocess, sys, threading, time

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}
CSV_FIELDS = ["asset", "status", "gen_s", "queue_wait_s", "post_s", "finish_wait_s", "finish_s", "asset_total_s",
              "dump_mb", "glb_sha256_masked", "error",
              "gen_start", "gen_end", "post_start", "post_end", "finish_start", "finish_end",
              "dump_mode", "front_hash", "dump_ok", "warnings", "gen_lane", "gen_gpu", "phase_gpu", "gate_wait_s"]
TIME_KEYS = {"gen_start": "t_gen_start", "gen_end": "t_gen_end", "post_start": "t_post_start", "post_end": "t_post_end",
             "finish_start": "t_finish_start", "finish_end": "t_finish_end"}
PHASE_LOCK_ENV = "ARCTIC_GPU_PHASE_LOCK"
HERE = os.path.dirname(os.path.abspath(__file__))
HASH_RE = re.compile(r"\[post-hash\]\s+(\S+)\s+V=(\d+)\s+F=(\d+)\s+verts=([0-9a-f]+)\s+faces=([0-9a-f]+)")
# log patterns -> warning tokens; "GPU decimation unavailable" is decimate_qem.cpp's CPU-fallback message (15-25 min per prop)
WARN_PATTERNS = [("cpu_qem", re.compile(r"GPU decimation unavailable")),
                 ("oom", re.compile(r"out of memory|OUT_OF_MEMORY|MemoryAllocation|bad_alloc", re.I))]


def split_cmd(s, posix=None):
    """shlex split. On Windows (posix=False) shlex keeps the quote characters in the tokens, which Popen would then
    escape again, so surrounding quotes are stripped."""
    if not isinstance(s, str):
        return list(s)
    posix = (os.name != "nt") if posix is None else posix
    toks = shlex.split(s, posix=posix)
    if not posix:
        toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]
    return toks


def iso(t):
    return "" if not t else time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + ".%03dZ" % int((t % 1) * 1000)


def parse_hashes(text):
    """stage -> (V, F, verts_hash, faces_hash), last occurrence wins (TRELLIS_POST_HASH=1 output)."""
    out = {}
    for m in HASH_RE.finditer(text):
        out[m.group(1)] = (int(m.group(2)), int(m.group(3)), m.group(4), m.group(5))
    return out


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def scan_warnings(path):
    text = read_text(path)
    return {tok for tok, rx in WARN_PATTERNS if rx.search(text)}


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
        self.procs = set()          # live children, killed on Ctrl+C / fatal error
        self.stop = threading.Event()
        self.aborted = None
        self.finish = split_cmd(a.finish_cmd) if a.finish_cmd else None
        self.gate = None            # dispatcher.VramGate, set by run_lanes (--gen-lanes 2)
        self.lane_b_busy = 0

    # -- environments -------------------------------------------------------
    def gen_env(self, dump=None, lane2=False):
        e = dict(os.environ)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if lane2:   # hide the other card completely, as the item 12 sheet did; --gpu is then 0 in the remapped view
            e["CUDA_VISIBLE_DEVICES"] = str(self.a.lane2_gpu)
            for kv in self.a.lane2_env:
                k, _, v = kv.partition("=")
                e[k] = v
            for k in self.a.lane2_unset:   # TRELLIS_CONVNEXT_STREAM_ALL is tested for presence (getenv != nullptr): "=0" does not turn it off
                e.pop(k, None)
        if self.a.hash:
            e["TRELLIS_POST_HASH"] = "1"
        if self.a.dump_mode == "raw":
            # trellis_cli.cpp: TRELLIS_DUMP_POST writes the pre-weld mesh after fill_holes, TRELLIS_STOP_AFTER_POST_DUMP
            # then returns 0, so the generation process never runs weld / remesh / clean / drop.
            e["TRELLIS_DUMP_POST"] = dump
            e["TRELLIS_STOP_AFTER_POST_DUMP"] = "1"
        return e

    def remesh_gib(self):
        """TRELLIS_REMESH_MEM_GIB for post-replay: auto = 4 GiB with one post worker (24 remesh workers instead of 8),
        unset with several (they share the RAM); a number forces it; 0 or less leaves the binary's default."""
        g = self.a.remesh_mem_gib
        if g is None:
            return 4.0 if self.a.post_workers == 1 else None
        return g if g > 0 else None

    def phase_lock_for(self, gpu):
        """One phase-lock file per card: QEM and bake are serialised on the card they run on."""
        pl = self.a.phase_lock
        if not pl or gpu == self.a.post_gpu:
            return pl
        tag = "gpu%d.phase" % self.a.post_gpu
        return pl[:-len(tag)] + "gpu%d.phase" % gpu if pl.endswith(tag) else "%s.gpu%d" % (pl, gpu)

    def post_env(self, threads, gpu=None):
        gpu = self.a.post_gpu if gpu is None else gpu
        e = dict(os.environ)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        e["TRELLIS_POST_THREADS"] = str(threads)
        if self.a.hash:
            e["TRELLIS_POST_HASH"] = "1"
        if self.remesh_gib() is not None:
            e["TRELLIS_REMESH_MEM_GIB"] = "%g" % self.remesh_gib()
        if self.a.phase_lock:
            e[PHASE_LOCK_ENV] = self.phase_lock_for(gpu)
        # post-replay never runs ggml: without this its GPU QEM lands on CUDA device 0.
        # "-1" = no visible device -> decimate_qem falls back to the CPU path.
        e["CUDA_VISIBLE_DEVICES"] = "-1" if self.a.qem == "cpu" else str(gpu)
        return e

    # -- commands -----------------------------------------------------------
    def gen_cmd(self, img, name, dump, lane2=False):
        a = self.a
        o = getattr(self, "job_opts", {}).get(name, {})   # per-job overrides from the live queue (OPT-IN; empty = unchanged)
        cmd = self.cli + ["-i", img, "-o", os.path.join(self.dumps, name + ".unused.glb"),
                          "-m", a.models, "--res", str(o.get("res", a.res)), "--gpu", "0" if lane2 else str(a.gen_gpu),
                          "--seed", str(o.get("seed", a.seed))]
        if a.dump_mode == "cleaned":
            cmd += ["--dump-post", dump]
        return cmd + split_cmd(a.cli_extra) + split_cmd(o.get("cli_extra", ""))

    def post_cmd(self, dump, glb, name=None):
        a = self.a
        o = getattr(self, "job_opts", {}).get(name, {})
        res = o.get("res", a.res)
        band = max(1, res // 512)
        return self.replay + [dump, glb, "--geometry-res", str(res), "--band", str(band),
                              "--faces", str(o.get("faces", a.faces)), "--no-audit"] + split_cmd(a.replay_extra)

    def run_logged(self, cmd, env, log, timeout=None, low_priority=False):
        """Run a child with its output in a log. Returns the exit code, or -9 on timeout / -15 on abort.
        low_priority: post and finish yield CPU to the generation process (its CPU feed thread must not starve)."""
        kw = {}
        if low_priority and self.a.low_priority:
            if os.name == "nt":
                kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
            else:
                kw["preexec_fn"] = lambda: os.nice(10)
        if self.stop.is_set():
            return -15
        with open(log, "wb") as lf:
            lf.write((" ".join(cmd) + "\n").encode())
            lf.flush()
            p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, **kw)
            self.procs.add(p)
            try:
                return p.wait(timeout=timeout or None)
            except subprocess.TimeoutExpired:
                self.kill(p)
                lf.write(b"\n[arctic-batch] killed after %ss timeout\n" % str(timeout).encode())
                return -9
            finally:
                self.procs.discard(p)
                if self.stop.is_set() and p.poll() is None:
                    self.kill(p)

    @staticmethod
    def kill(p):
        try:
            if os.name == "nt":  # kill the tree: blender / trellis-cli may have children
                subprocess.call(["taskkill", "/F", "/T", "/PID", str(p.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                p.kill()
            p.wait(timeout=30)
        except Exception:
            pass

    def abort(self):
        self.stop.set()
        for p in list(self.procs):
            self.kill(p)

    # -- stages ---------------------------------------------------------------
    def do_gen(self, img, name, lane2=False):
        dump = os.path.join(self.dumps, name + ".bin")
        self.rm(dump)  # a stale dump from an earlier run must never pass for this run's output
        glog = os.path.join(self.logs, name + ".gen.log")
        lane = ({"gen_lane": "B" if lane2 else "A", "gen_gpu": self.a.lane2_gpu if lane2 else self.a.gen_gpu}
                if self.a.gen_lanes == 2 else {})
        if lane2 and self.gate is not None:   # admission by declared VRAM: wait while post/finish hold the 3060 Ti
            w0 = time.time()
            if not self.gate.wait_admit(self.a.lane2_gpu, "gen:" + name, self.a.gen_need_gb, stop=self.stop):
                self.fail(name, "gen", 0.0, "aborted while waiting for VRAM on the 3060 Ti", time.time())
                return None
            lane["gate_wait_s"] = time.time() - w0
        t0 = time.time()
        try:
            rc = self.run_logged(self.gen_cmd(img, name, dump, lane2), self.gen_env(dump, lane2), glog, timeout=self.a.gen_timeout)
        finally:
            if lane2 and self.gate is not None:
                self.gate.release(self.a.lane2_gpu, "gen:" + name)
        dt = time.time() - t0
        if rc != 0 or not os.path.isfile(dump):
            self.fail(name, "gen", dt, "trellis-cli exit %s" % rc, t0)
            self.rows[name].update(lane)
            self.rows[name]["_warn"] |= {"gen:" + w for w in scan_warnings(glog)}
            return None
        warn = {"gen:" + w for w in scan_warnings(glog)}
        self.rows[name] = {"asset": name, "gen_s": dt, "t_gen_start": t0, "t_gen_end": time.time(),
                           "dump_mb": os.path.getsize(dump) / 1048576.0, "dump_mode": self.a.dump_mode,
                           "_warn": warn, **lane}
        return dump

    @staticmethod
    def rm(path):
        try:
            os.remove(path)
        except OSError:
            pass

    def do_post(self, name, dump, threads):
        row = self.rows[name]
        t_start = time.time()
        row["t_post_start"] = t_start
        row["queue_wait_s"] = t_start - row["t_gen_end"]
        glb = os.path.join(self.out, name + ".glb")
        tmp = os.path.join(self.out, name + ".part.glb")  # written here, renamed only on success
        self.rm(glb); self.rm(tmp)
        plog = os.path.join(self.logs, name + ".post.log")
        gpu = self.place_phase(name, "post", self.a.post_need_gb, row)
        try:
            rc = self.run_logged(self.post_cmd(dump, tmp, name), self.post_env(threads, gpu), plog,
                                 timeout=self.a.post_timeout, low_priority=True)
        finally:
            self.unplace_phase(gpu)
        row["post_s"] = time.time() - t_start
        self.read_hashes(name, plog)
        pw = scan_warnings(plog)
        row["_warn"] |= {"post:" + w for w in pw}
        if rc != 0 or not os.path.isfile(tmp):
            row["status"], row["error"] = "post_failed", "post-replay exit %s" % rc
            self.rm(tmp)
        elif "cpu_qem" in pw and self.a.fail_on_cpu_qem:
            row["status"], row["error"] = "post_failed", "GPU QEM fell back to the CPU (--fail-on-cpu-qem)"
            self.rm(tmp)
        else:
            os.replace(tmp, glb)
            row["status"] = "ok"
            row["glb_sha256_masked"] = masked_glb_hash(glb)
            if "cpu_qem" in pw:
                self.log("%s: WARNING GPU QEM fell back to the CPU (VRAM contention?)" % name)
        row["asset_total_s"] = row["gen_s"] + row["queue_wait_s"] + row["post_s"]
        row["t_post_end"] = time.time()
        if not self.a.keep_dumps and row["status"] == "ok":
            try:
                os.remove(dump)
            except OSError:
                pass
        self.log("%s: gen %.1fs, wait %.1fs, post %.1fs -> %s" %
                 (name, row["gen_s"], row["queue_wait_s"], row["post_s"], row["status"]))
        return glb if row["status"] == "ok" else None

    def do_finish(self, name, glb):
        """Optional Blender finish. --finish-cmd is a template with {glb} {name} {out}. Runs with only the post GPU
        (the 3060 Ti) visible so a Cycles bake cannot land on the generation card."""
        row = self.rows[name]
        t_start = time.time()
        row["t_finish_start"] = t_start
        row["finish_wait_s"] = t_start - row["t_post_end"]
        fdir = os.path.join(self.out, "finished", name)
        os.makedirs(fdir, exist_ok=True)
        cmd = [c.format(glb=glb, name=name, out=fdir) for c in self.finish]
        e = dict(os.environ)
        e["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu = self.place_phase(name, "finish", self.a.finish_need_gb, row)
        e["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if self.a.phase_lock:
            e[PHASE_LOCK_ENV] = self.phase_lock_for(gpu)
        flog = os.path.join(self.logs, name + ".finish.log")
        try:
            rc = self.run_logged(cmd, e, flog, timeout=self.a.finish_timeout, low_priority=True)
            fw = scan_warnings(flog)
            for attempt in range(self.a.finish_retries):  # an out-of-memory finish is a bake collision: wait, then try once more
                if rc == 0 or "oom" not in fw or self.stop.is_set():
                    break
                self.log("%s: finish hit an out-of-memory; retrying in %.0fs" % (name, self.a.finish_retry_wait))
                row["_warn"].add("finish:oom_retried")
                time.sleep(self.a.finish_retry_wait)
                os.replace(flog, flog + ".try%d" % (attempt + 1))
                rc = self.run_logged(cmd, e, flog, timeout=self.a.finish_timeout, low_priority=True)
                fw = scan_warnings(flog)
        finally:
            self.unplace_phase(gpu)
        row["_warn"] |= {"finish:" + w for w in fw}
        row["t_finish_end"] = time.time()
        row["finish_s"] = row["t_finish_end"] - t_start
        if rc != 0:
            row["status"], row["error"] = "finish_failed", "finish exit %s" % rc
        row["asset_total_s"] = row["gen_s"] + row["queue_wait_s"] + row["post_s"] + row["finish_wait_s"] + row["finish_s"]
        self.log("%s: finish %.1fs (wait %.1fs) -> %s" % (name, row["finish_s"], row["finish_wait_s"], row["status"]))

    def read_hashes(self, name, plog):
        """front_hash = the post chain's drop_components buffers; dump_ok = the generation's fill_holes hash equals
        post-replay's input hash (raw mode: proves the dump round trip). Both need --hash."""
        row = self.rows[name]
        ph = parse_hashes(read_text(plog))
        if "drop_components" in ph:
            row["front_hash"] = "%s:%s" % (ph["drop_components"][2], ph["drop_components"][3])
        gh = parse_hashes(read_text(os.path.join(self.logs, name + ".gen.log")))
        if "fill_holes" in gh and "input" in ph:
            row["dump_ok"] = "yes" if gh["fill_holes"] == ph["input"] else "NO"

    def fail(self, name, stage, dt, err, t0=None):
        self.rows[name] = {"asset": name, "status": stage + "_failed", "gen_s": dt, "error": err,
                           "dump_mode": self.a.dump_mode, "_warn": set()}
        if t0:
            self.rows[name].update(t_gen_start=t0, t_gen_end=t0 + dt)
        self.log("%s: %s FAILED (%s)" % (name, stage, err))

    def log(self, msg):
        with self.lock:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    # -- VRAM placement (--gen-lanes 2 only; gate is None otherwise and everything runs on --post-gpu as before) ----------
    def place_phase(self, name, kind, need, row):
        """GPU for a post-replay / finish child. With no gate: --post-gpu, no waiting. With the gate: wait until the card has
        the declared VRAM free (QEM and bake share ONE 'phase' slot because the phase lock serialises them). Placement
        'flexible' may use the generation card when the 3060 Ti is full; after --gate-timeout the child starts anyway
        (warning gate_timeout) so a wrong estimate can never deadlock the batch."""
        if self.gate is None or self.a.qem == "cpu":
            return self.a.post_gpu
        cands = [self.a.post_gpu] + ([self.a.gen_gpu] if self.a.phase_placement == "flexible" else [])
        need = max(self.a.post_need_gb, self.a.finish_need_gb)   # one slot for QEM and bake
        t0 = time.time()
        while not self.stop.is_set():
            for g in cands:
                if self.gate.acquire_shared(g, "phase", need):
                    row["phase_gpu"] = g
                    row["gate_wait_s"] = row.get("gate_wait_s", 0.0) + time.time() - t0
                    return g
            if time.time() - t0 > self.a.gate_timeout:
                row["_warn"].add(kind + ":gate_timeout")
                self.log("%s: %s waited %.0fs for VRAM on GPU %s; starting anyway" % (name, kind, time.time() - t0, cands))
                row["phase_gpu"] = self.a.post_gpu
                self.gate.acquire_shared(self.a.post_gpu, "phase", need, force=True)
                return self.a.post_gpu
            with self.gate.cond:
                self.gate.cond.wait(2.0)
        return self.a.post_gpu

    def unplace_phase(self, gpu):
        if self.gate is not None and self.a.qem != "cpu":
            self.gate.release_shared(gpu, "phase")

    def run_lanes(self, items, baseline=None):
        """--gen-lanes 2: lane A (--gen-gpu) and lane B (--lane2-gpu) each pull from one pending list; Dispatcher.choose says
        who takes what. A job whose generation runs out of memory on lane B is re-queued once, pinned to lane A."""
        import dispatcher as dsp
        a = self.a
        est = load_est(a.est_from) if a.est_from else {}
        vram = dict(kv.split("=", 1) for kv in a.job_vram)
        pins = dict(kv.split("=", 1) for kv in a.pin)
        caps = dict((int(k), float(v)) for k, v in (kv.split("=", 1) for kv in a.gpu_cap_gb.split(",")))
        base = dict(baseline or {})
        base.setdefault(a.gen_gpu, 1.5)
        base.setdefault(a.lane2_gpu, 0.0)
        la = dsp.Lane("A", a.gen_gpu, 1.0, caps.get(a.gen_gpu, 12.0), base[a.gen_gpu], primary=True)
        lb = dsp.Lane("B", a.lane2_gpu, a.lane2_speed, caps.get(a.lane2_gpu, 8.0), base[a.lane2_gpu])
        disp = dsp.Dispatcher([la, lb], a.vram_margin_gb, a.min_gain_s, longest_first=(a.order == "longest-first"))
        live = None
        if a.queue_dir:
            import live_queue
            live = live_queue.LiveQueue(a.queue_dir, self.log)
            self.live_names = []
            self.job_opts = {}
        idle = {"since": None}
        self.gate = dsp.VramGate({g: caps.get(g, 12.0) for g in (a.gen_gpu, a.lane2_gpu)}, base, a.vram_margin_gb)
        img_of = dict((n, p) for p, n in items)
        pend = [dsp.Job(n, est.get(n), float(vram.get(n, a.gen_need_gb)), pin=pins.get(n)) for _, n in items]
        q = queue.Queue(maxsize=a.max_queue)
        fq = queue.Queue(maxsize=a.finish_queue)
        cond = threading.Condition()
        npost, threads = a.post_workers, max(1, a.total_threads // a.post_workers)
        retried = set()

        def ingest():   # under cond: take the new files of the live queue (OPT-IN: --queue-dir)
            if live is None:
                return
            for jd in live.poll(list(img_of)):
                img_of[jd["name"]] = jd["image"]
                self.live_names.append(jd["name"])
                if jd.get("opts"):
                    self.job_opts[jd["name"]] = jd["opts"]
                pend.append(dsp.Job(jd["name"], jd["est_s"] if jd["est_s"] is not None else est.get(jd["name"]),
                                    float(jd["vram_gb"] if jd["vram_gb"] is not None else a.gen_need_gb), jd["priority"], jd["pin"]))
                self.log("%s: queued live (priority %d%s)" % (jd["name"], jd["priority"], ", pin " + jd["pin"] if jd["pin"] else ""))

        def pworker():
            while True:
                job = q.get()
                if job is None:
                    return
                try:
                    glb = self.do_post(job[0], job[1], threads)
                    if glb and self.finish:
                        fq.put((job[0], glb))
                except BaseException as e:
                    self.rows[job[0]].update(status="post_failed", error=repr(e))
                    self.log("%s: post error %r" % (job[0], e))

        def fworker():
            while True:
                job = fq.get()
                if job is None:
                    return
                try:
                    self.do_finish(*job)
                except BaseException as e:
                    self.rows[job[0]].update(status="finish_failed", error=repr(e))
                    self.log("%s: finish error %r" % (job[0], e))

        def lane_worker(lane):
            while True:
                with cond:
                    while True:
                        if self.stop.is_set():
                            return
                        ingest()
                        j = disp.choose(time.time(), lane, pend)
                        if j is not None:
                            pend.remove(j)
                            lane.job, lane.started = j, time.time()
                            lane.est_end = lane.started + (j.est_s / lane.speed if j.est_s else 0.0)
                            break
                        if not pend and all(l.job is None for l in disp.lanes if l is not lane):
                            if live is None:
                                return
                            if live.is_closed():
                                ingest()   # a job written before CLOSE is visible now
                                if not pend:
                                    return
                            elif a.idle_exit > 0:
                                idle["since"] = idle["since"] or time.time()
                                if time.time() - idle["since"] >= a.idle_exit:
                                    return
                        else:
                            idle["since"] = None
                        cond.wait(2.0)
                self.log("%s: generation -> lane %s (GPU %d)" % (j.name, lane.name, lane.gpu))
                dump = self.do_gen(img_of[j.name], j.name, lane2=not lane.primary)
                with cond:
                    lane.job = None
                    row = self.rows.get(j.name, {})
                    if dump:
                        disp.observe(lane, j, row.get("gen_s", 0.0))
                    elif not lane.primary and j.name not in retried and "gen:oom" in row.get("_warn", ()):
                        retried.add(j.name)   # does not fit the 3060 Ti: the 4070 takes it, nothing is lost
                        pend.insert(0, dsp.Job(j.name, j.est_s, j.vram_gb, pin=la.name))
                        self.log("%s: out of memory on lane B; re-queued on lane A" % j.name)
                    cond.notify_all()
                if dump:
                    self.rows[j.name]["t_gen_end"] = time.time()
                    q.put((j.name, dump))

        ws = [threading.Thread(target=pworker, daemon=True) for _ in range(npost)]
        fs = [threading.Thread(target=fworker, daemon=True) for _ in range(a.finish_workers if self.finish else 0)]
        ls = [threading.Thread(target=lane_worker, args=(l,), daemon=True) for l in (la, lb)]
        for w in ws + fs + ls:
            w.start()
        try:
            for w in ls:
                while w.is_alive():
                    w.join(0.5)
        except BaseException as e:
            self.log("ABORT: %r; stopping children" % (e,))
            self.abort()
            self.aborted = e
            with cond:
                cond.notify_all()
        finally:
            for w in ls:
                w.join(60)
            for _ in ws:
                q.put(None)
            for w in ws:
                w.join()
            for _ in fs:
                fq.put(None)
            for w in fs:
                w.join()

    # -- drivers --------------------------------------------------------------
    def run_serial(self, items):
        for img, name in items:
            if self.stop.is_set():
                break
            dump = self.do_gen(img, name)
            if dump:
                glb = self.do_post(name, dump, self.a.total_threads)
                if glb and self.finish:
                    self.rows[name]["t_post_end"] = time.time()
                    self.do_finish(name, glb)

    def run_overlap(self, items):
        q = queue.Queue(maxsize=self.a.max_queue)  # bounds dumps on disk (0.3-0.6 GB each)
        fq = queue.Queue(maxsize=self.a.finish_queue)  # post workers (not gen) block when finish is behind
        n, threads = self.a.post_workers, max(1, self.a.total_threads // self.a.post_workers)

        def pworker():
            while True:
                job = q.get()
                if job is None:
                    return
                try:
                    glb = self.do_post(job[0], job[1], threads)
                    if glb and self.finish:
                        fq.put((job[0], glb))
                except BaseException as e:  # keep the pool alive; never leave a row without a status
                    self.rows[job[0]].update(status="post_failed", error=repr(e))
                    self.log("%s: post error %r" % (job[0], e))

        def fworker():
            while True:
                job = fq.get()
                if job is None:
                    return
                try:
                    self.do_finish(*job)
                except BaseException as e:
                    self.rows[job[0]].update(status="finish_failed", error=repr(e))
                    self.log("%s: finish error %r" % (job[0], e))

        ws = [threading.Thread(target=pworker, daemon=True) for _ in range(n)]
        fs = [threading.Thread(target=fworker, daemon=True) for _ in range(self.a.finish_workers if self.finish else 0)]
        for w in ws + fs:
            w.start()
        try:
            for img, name in items:
                if self.stop.is_set():
                    break
                dump = self.do_gen(img, name)
                if dump:
                    self.rows[name]["t_gen_end"] = time.time()
                    q.put((name, dump))  # blocks when the post pool is max_queue behind
        except BaseException as e:  # Ctrl+C, or an unexpected error in the gen loop (e.g. exe not found)
            self.log("ABORT: %r; stopping children" % (e,))
            self.abort()
            self.aborted = e
        finally:
            for _ in ws:  # sentinels always go in, so workers cannot block forever on an empty queue
                q.put(None)
            for w in ws:
                w.join()
            for _ in fs:
                fq.put(None)
            for w in fs:
                w.join()

    def write_csv(self, order, wall):
        path = os.path.join(self.out, "batch.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, CSV_FIELDS)
            w.writeheader()
            for name in order:
                r = dict(self.rows.get(name, {"asset": name, "status": "not_run"}))
                for col, key in TIME_KEYS.items():
                    r[col] = iso(r.get(key))
                r["warnings"] = ";".join(sorted(r.get("_warn", ())))
                w.writerow({k: ("%.2f" % r[k] if isinstance(r.get(k), float) else r.get(k, ""))
                            for k in CSV_FIELDS})
            ok = sum(1 for r in self.rows.values() if r.get("status") == "ok")
            w.writerow({"asset": "WALL_TOTAL", "status": "%d/%d ok" % (ok, len(order)),
                        "asset_total_s": "%.2f" % wall,
                        "error": "mode=%s assets_per_hour=%.2f" %
                                 ("serial" if self.a.serial else "overlap", ok * 3600.0 / wall if wall else 0)})
        return path

    def write_report(self, order, wall):
        """batch_report.md: one page to read instead of the logs (wall, busy %, per-prop table, warnings, verdict)."""
        rows = [self.rows.get(n, {"asset": n, "status": "not_run"}) for n in order]
        ok = [r for r in rows if r.get("status") == "ok"]
        gen_sum = sum(r.get("gen_s", 0.0) for r in rows if isinstance(r.get("gen_s"), float))
        mode = "serial" if self.a.serial else "overlap"
        problems = ["%s: %s" % (r["asset"], r.get("status")) for r in rows if r.get("status") != "ok"]
        for r in rows:
            for wtok in sorted(r.get("_warn", ())):
                if wtok.split(":")[1] in ("cpu_qem", "oom", "oom_retried"):
                    problems.append("%s: %s" % (r["asset"], wtok))
            if r.get("dump_ok") == "NO":
                problems.append("%s: dump_ok NO (generation fill_holes hash != post-replay input hash)" % r["asset"])
        lines = ["# arctic-batch report", "",
                 "- mode %s, dump mode %s, post workers %d x %d threads, finish workers %d" %
                 (mode, self.a.dump_mode, self.a.post_workers, max(1, self.a.total_threads // self.a.post_workers),
                  self.a.finish_workers if self.finish else 0),
                 "- wall %.0f s, %d/%d ok, %.2f assets/hour" % (wall, len(ok), len(order), len(ok) * 3600.0 / wall if wall else 0),
                 "- generation GPU busy %.0f %% of the wall (sum of gen_s %.0f s)" % (100.0 * gen_sum / wall if wall else 0, gen_sum),
                 "- phase lock: %s; remesh budget: %s" % (self.a.phase_lock or "off",
                                                          "%g GiB" % self.remesh_gib() if self.remesh_gib() else "binary default"), ""]
        timed = sorted((r for r in rows if r.get("t_gen_start")), key=lambda r: r["t_gen_start"])
        if mode == "overlap" and len(timed) > 1:
            gaps = [max(0.0, b["t_gen_start"] - a["t_gen_end"]) for a, b in zip(timed, timed[1:]) if a.get("t_gen_end")]
            lines.append("- generation GPU idle between props: %.0f s total (post queue full or a failed prop)" % sum(gaps))
            slow = sum(1 for a, b in zip(timed, timed[1:])
                       if isinstance(a.get("post_s"), float) and isinstance(b.get("gen_s"), float) and a["post_s"] > b["gen_s"])
            lines.append("- post slower than the next generation on %d prop(s): %s" %
                         (slow, "a second post worker may help (T3)" if slow >= 2 else "one post worker keeps up"))
            lines.append("")
        lines += ["| asset | status | gen_s | queue_wait_s | post_s | finish_wait_s | finish_s | warnings |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            f = lambda k: ("%.0f" % r[k]) if isinstance(r.get(k), float) else ""
            lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["asset"], r.get("status", ""), f("gen_s"), f("queue_wait_s"), f("post_s"), f("finish_wait_s"),
                f("finish_s"), ";".join(sorted(r.get("_warn", ()))) or ""))
        lines += ["", "**Verdict: %s**" % ("OK" if not problems else "CHECK - " + "; ".join(problems)), ""]
        path = os.path.join(self.out, "batch_report.md")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines))
        return path


def load_gpulock(path):
    if not os.path.isfile(path):
        raise SystemExit("--gpulock: %s not found (Expiramental3d tools\\gpulock.py)" % path)
    spec = importlib.util.spec_from_file_location("gpulock", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GpuLocks:
    """Hold the cross-project GPU locks (tools/gpulock.py protocol) for the whole batch. The lock records THIS
    process's pid, so it goes stale (and Space Gothic may remove it) only if the runner dies. Reserve first so a
    lower-priority owner cannot slip in between the acquires; release and unreserve in __exit__.
    assume_held: the runner was started under `gpulock run` (or the Hub), which already holds the locks."""

    def __init__(self, a, log=print):
        self.a, self.log, self.mod = a, log, None
        self.taken, self.reserved = [], []
        self.phase_path = None
        self.gpus = sorted({a.gen_gpu, a.post_gpu})

    def __enter__(self):
        a = self.a
        if not a.gpulock:
            return self
        self.mod = m = load_gpulock(a.gpulock)
        self.phase_path = os.path.join(str(m.lock_dir()), "gpu%d.phase" % a.post_gpu)
        if a.gpulock_assume_held:
            for g in self.gpus:
                info = m.read_json(m.lock_path(g))
                if not info or str(info.get("owner", "")).lower() != a.lock_owner.lower():
                    self.fail("--gpulock-assume-held but GPU %d is not locked by %s" % (g, a.lock_owner))
            return self
        try:
            if a.lock_hours > 0:
                for g in self.gpus:
                    ok, d = m.reserve(g, a.lock_owner, a.lock_hours, "arctic-batch " + a.out)
                    if not ok:
                        self.fail(d)
                    self.reserved.append(g)
            for g in self.gpus:
                if not m.acquire(g, a.lock_owner, "arctic-batch " + a.out, wait=a.lock_wait, log=self.log):
                    self.fail("GPU %d is busy" % g)
                self.taken.append(g)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def fail(self, msg):
        print("arctic-batch: cannot lock the GPUs: %s" % msg, file=sys.stderr)
        raise SystemExit(3)

    def __exit__(self, *exc):
        for g in self.taken:
            self.mod.release(g, self.a.lock_owner, os.getpid())
        for g in self.reserved:
            self.mod.unreserve(g, self.a.lock_owner)
        self.taken, self.reserved = [], []
        return False


def preflight(a, log=print):
    """Returns {gpu: GB in use now} (the dispatcher's baseline), or None when nvidia-smi is unavailable."""
    """Refuse to start while something else holds VRAM we need: ComfyUI / llama-server on the 3060 Ti would push GPU
    QEM into its 15-25 min CPU fallback, a browser or Blender on the 4070 eats the generation's thin margin.
    nvidia-smi missing or failing = skipped with a line, never a failure."""
    cmd = split_cmd(a.nvidia_smi) + ["--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError) as e:
        log("preflight skipped (nvidia-smi unavailable: %s)" % e)
        return
    used = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                used[int(parts[0])] = float(parts[1])
            except ValueError:
                pass
    if not used:
        log("preflight skipped (no nvidia-smi output)")
        return
    limits = {a.gen_gpu: a.gen_gpu_max_gb}
    if a.finish_cmd or a.qem == "gpu":
        limits[a.post_gpu] = a.post_gpu_max_gb
    bad = ["GPU %d holds %.1f GB (limit %.1f GB)" % (g, used[g] / 1024.0, lim)
           for g, lim in sorted(limits.items()) if g in used and used[g] > lim * 1024.0]
    if bad:
        print("arctic-batch preflight: %s. Close ComfyUI / llama-server / Blender / browsers, or pass --no-preflight." %
              "; ".join(bad), file=sys.stderr)
        raise SystemExit(4)
    log("preflight ok: " + ", ".join("GPU %d %.1f GB used" % (g, used[g] / 1024.0) for g in sorted(used)))
    return {g: v / 1024.0 for g, v in used.items()}


def load_est(path):
    """gen_s per asset from an earlier batch.csv (status ok): the job-size estimates."""
    with open(path, encoding="utf-8-sig") as f:
        return {r["asset"]: float(r["gen_s"]) for r in csv.DictReader(f) if r.get("status") == "ok" and r.get("gen_s")}


def longest_first(items, est):
    """Queue order with the slowest jobs first (stable). A job with no estimate counts as the median known one, so it
    keeps its place among the known jobs instead of dropping to the end. Order only: each job's output is unchanged."""
    import statistics
    known = [est[n] for _, n in items if n in est]
    if not known:
        return list(items)
    mid = statistics.median(known)
    return sorted(items, key=lambda it: -est.get(it[1], mid))


def cmd_run(a):
    imgs = collect_images(a.images) if a.images else []
    names, items = set(), []
    for p in imgs:
        n = os.path.splitext(os.path.basename(p))[0]
        if n.lower() in names:  # NTFS is case-insensitive: A.png and a.png would share a dump and a GLB
            raise SystemExit("duplicate asset name: " + n)
        names.add(n.lower())
        items.append((p, n))
    with GpuLocks(a) as locks:  # SystemExit(3) when a GPU is busy or reserved; nothing has been started yet
        if not a.phase_lock and locks.phase_path:
            a.phase_lock = locks.phase_path
        if a.phase_lock:  # include/phase_lock.h opens the file but does not create directories
            os.makedirs(os.path.dirname(os.path.abspath(a.phase_lock)), exist_ok=True)
        used = preflight(a) if a.preflight else None
        b = Batch(a)
        t0 = time.time()
        todo = longest_first(items, load_est(a.est_from)) if a.order == "longest-first" else items
        if todo is not items:
            b.log("order: longest first -> " + ", ".join(n for _, n in todo))
        try:
            if a.gen_lanes == 2:
                b.run_lanes(todo, used)
            else:
                (b.run_serial if a.serial else b.run_overlap)(todo)
        except BaseException as e:  # serial-mode Ctrl+C; overlap mode handles its own
            b.log("ABORT: %r" % (e,)); b.abort(); b.aborted = e
        wall = time.time() - t0
    items = items + [(None, n) for n in getattr(b, "live_names", [])]   # jobs that arrived through --queue-dir
    path = b.write_csv([n for _, n in items], wall)
    b.log("wall %.1fs; csv %s; report %s" % (wall, path, b.write_report([n for _, n in items], wall)))
    if b.aborted is not None:
        return 130 if isinstance(b.aborted, KeyboardInterrupt) else 2
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
    r.add_argument("images", nargs="*", help="image files and/or directories (optional with --queue-dir)")
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
    r.add_argument("--dump-mode", choices=["cleaned", "raw"], default="cleaned",
                   help="cleaned = trellis-cli --dump-post (weld..drop runs in the generation process, post-replay remeshes "
                        "again); raw = TRELLIS_DUMP_POST + TRELLIS_STOP_AFTER_POST_DUMP (generation exits after the texture "
                        "decode; post-replay runs the serial CLI's exact chain). Do not pass --no-remesh / --fill-holes with raw")
    r.add_argument("--hash", action="store_true",
                   help="TRELLIS_POST_HASH=1 for generation and post: front_hash / dump_ok columns (T6 same-output check)")
    r.add_argument("--fail-on-cpu-qem", action="store_true",
                   help="treat post-replay's 'GPU decimation unavailable' CPU fallback as a failed post (default: warning)")
    r.add_argument("--remesh-mem-gib", type=float, default=None,
                   help="TRELLIS_REMESH_MEM_GIB for post-replay (default auto: 4 with one post worker; 0 = binary default)")
    r.add_argument("--post-workers", type=int, default=1,
                   help="default 1: two workers put two GPU QEMs on the 3060 Ti at once for no throughput gain at FS2 settings")
    r.add_argument("--total-threads", type=int, default=24, help="threads/worker = total/workers")
    r.add_argument("--max-queue", type=int, default=2, help="finished dumps allowed to wait for post")
    r.add_argument("--gen-timeout", type=float, default=3600, help="kill a trellis-cli running longer than this (s; 0 = never)")
    r.add_argument("--post-timeout", type=float, default=3600)
    r.add_argument("--finish-timeout", type=float, default=3600)
    r.add_argument("--no-low-priority", dest="low_priority", action="store_false",
                   help="post/finish run BELOW_NORMAL (nice 10) by default so the gen process keeps its CPU feed")
    r.add_argument("--finish-cmd", default="", help="optional Blender finish per GLB, template with {glb} {name} {out}, e.g. "
                   "'blender -b -P finish.py -- --in {glb} --out {out} --cycles-device CUDA --gpu-index 0'")
    r.add_argument("--finish-workers", type=int, default=1)
    r.add_argument("--finish-retries", type=int, default=1, help="retries of a finish whose log shows an out-of-memory")
    r.add_argument("--finish-retry-wait", type=float, default=30, help="seconds before that retry")
    r.add_argument("--finish-queue", type=int, default=4, help="GLBs allowed to wait for finish before post workers block")
    r.add_argument("--gpulock", default="", metavar="PATH_TO_gpulock.py",
                   help="hold the cross-project GPU locks for the batch (imports Expiramental3d tools/gpulock.py)")
    r.add_argument("--gpulock-assume-held", action="store_true",
                   help="the locks are already held by this owner (started under `gpulock run`); check, do not acquire")
    r.add_argument("--lock-owner", default="expiramental3d")
    r.add_argument("--lock-hours", type=float, default=3, help="reservation length per GPU (0 = no reservation)")
    r.add_argument("--lock-wait", type=float, default=0, help="seconds to wait for a busy GPU (default: refuse at once)")
    r.add_argument("--phase-lock", default="", metavar="PATH",
                   help="file serialising QEM (post-replay) and the Cycles bake (finish.py) on the post GPU; exported as "
                        "ARCTIC_GPU_PHASE_LOCK to post and finish only. With --gpulock: <lock dir>\\gpu<post-gpu>.phase")
    r.add_argument("--no-preflight", dest="preflight", action="store_false",
                   help="skip the check that the GPUs are free of other VRAM users")
    r.add_argument("--nvidia-smi", default="nvidia-smi")
    r.add_argument("--gen-gpu-max-gb", type=float, default=3.0, help="preflight: VRAM already used on the generation GPU")
    r.add_argument("--post-gpu-max-gb", type=float, default=1.5, help="preflight: VRAM already used on the post GPU")
    r.add_argument("--serial", action="store_true", help="gen then post, one asset at a time (A/B baseline)")
    g = r.add_argument_group("second generation lane (OPT-IN; default --gen-lanes 1 changes nothing)")
    g.add_argument("--gen-lanes", type=int, choices=[1, 2], default=1,
                   help="2 = a second generation lane on --lane2-gpu (the 3060 Ti) next to --gen-gpu, scheduled by dispatcher.py")
    g.add_argument("--lane2-gpu", type=int, default=0, help="PCI index of lane B; must equal --post-gpu (one GPU lock set)")
    g.add_argument("--lane2-speed", type=float, default=0.5, help="starting belief: lane B speed relative to lane A (learned from the first job)")
    g.add_argument("--lane2-env", action="append", default=[], metavar="KEY=VAL",
                   help="extra environment for lane B generations only, e.g. TRELLIS_CONVNEXT_STREAM_GB=6")
    g.add_argument("--lane2-unset", action="append", default=[], metavar="KEY",
                   help="remove KEY from lane B's environment (streaming off = unset TRELLIS_CONVNEXT_STREAM_ALL; stages >= "
                        "TRELLIS_CONVNEXT_STREAM_GB, default 2 GB, still stream)")
    g.add_argument("--est-from", default="", metavar="batch.csv",
                   help="gen_s per asset from an earlier batch.csv on lane A: the dispatcher's job-size estimates "
                        "(also the sizes for --order longest-first)")
    g.add_argument("--queue-dir", default="", metavar="DIR",
                   help="OPT-IN live queue (needs --gen-lanes 2): jobs added to DIR while the batch runs (`submit`), a priority per "
                        "job, ends on `close` (live_queue.py). Images on the command line are optional with it")
    g.add_argument("--idle-exit", type=float, default=0, metavar="S",
                   help="with --queue-dir: also exit after S seconds with nothing to do (default 0 = wait for `close`)")
    g.add_argument("--job-vram", action="append", default=[], metavar="NAME=GB", help="VRAM one generation needs (default --gen-need-gb)")
    g.add_argument("--pin", action="append", default=[], metavar="NAME=A|B", help="force a job onto a lane")
    g.add_argument("--min-gain-s", type=float, default=30.0, help="lane B takes a job only if the projected makespan drops by this much")
    g.add_argument("--gen-need-gb", type=float, default=4.5, help="declared generation VRAM (item 11: 4.3 GB with TRELLIS_CONVNEXT_STREAM_ALL=1)")
    g.add_argument("--post-need-gb", type=float, default=3.0, help="declared post-replay (QEM) VRAM on the 3060 Ti (item 11 peak 3.0 GB)")
    g.add_argument("--finish-need-gb", type=float, default=2.0, help="declared finish (bake) VRAM; QEM and bake share one slot (phase lock)")
    g.add_argument("--gpu-cap-gb", default="0=8,1=12", help="GPU index=capacity GB list")
    g.add_argument("--vram-margin-gb", type=float, default=0.4, help="VRAM never handed out")
    g.add_argument("--phase-placement", choices=["fixed", "flexible"], default="fixed",
                   help="fixed = post/finish only on --post-gpu (wait for VRAM); flexible = may use the generation GPU when the 3060 Ti "
                        "is full (needs the QEM cross-card check first)")
    g.add_argument("--gate-timeout", type=float, default=900, help="seconds a post/finish waits for VRAM before it starts anyway")
    r.add_argument("--order", choices=["queue", "longest-first"], default="queue",
                   help="job start order (OPT-IN; default queue = as given). longest-first starts the slowest jobs first so the "
                        "cards finish together; needs --est-from (batch.csv of an earlier run), without it the order is unchanged. "
                        "batch.csv rows stay in the given order; outputs are unchanged")
    r.add_argument("--keep-dumps", action="store_true")
    sb = sub.add_parser("submit", help="add a job to a live queue (--queue-dir of a running batch)")
    sb.add_argument("--queue-dir", required=True); sb.add_argument("--image", required=True); sb.add_argument("--name", default=None)
    sb.add_argument("--priority", type=int, default=0, help="higher starts first (default 0; 100 = urgent)")
    sb.add_argument("--est-s", type=float, default=None); sb.add_argument("--vram-gb", type=float, default=None)
    sb.add_argument("--pin", choices=["A", "B"], default=None)
    sb.add_argument("--job-res", type=int, default=None, help="this job only: --res (a draft is 512)")
    sb.add_argument("--job-seed", type=int, default=None); sb.add_argument("--job-faces", type=int, default=None)
    sb.add_argument("--job-cli-extra", default=None, help="this job only: extra trellis-cli args, appended after the batch's --cli-extra")
    cl = sub.add_parser("close", help="no more jobs for the live queue: the batch finishes what is queued and exits")
    cl.add_argument("--queue-dir", required=True)
    c = sub.add_parser("compare", help="compare GLBs of two output dirs (timestamp masked)")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "compare", "submit", "close", "-h", "--help"):
        argv.insert(0, "run")
    a = build_parser().parse_args(argv)
    if a.cmd == "compare":
        return cmd_compare(a)
    if a.cmd in ("submit", "close"):
        import live_queue
        if a.cmd == "close":
            live_queue.close(a.queue_dir)
            print("closed " + a.queue_dir)
        else:
            print(live_queue.submit(a.queue_dir, a.image, a.name, a.priority, a.est_s, a.vram_gb, a.pin,
                                    opts={"res": a.job_res, "seed": a.job_seed, "faces": a.job_faces, "cli_extra": a.job_cli_extra}))
        return 0
    if a.cmd != "run":
        build_parser().print_help()
        return 2
    if a.queue_dir and a.gen_lanes != 2:
        raise SystemExit("--queue-dir needs --gen-lanes 2")
    if not a.images and not a.queue_dir:
        raise SystemExit("no images (give images, or --queue-dir to feed jobs live)")
    if a.post_workers < 1 or a.max_queue < 1:
        raise SystemExit("--post-workers and --max-queue must be >= 1")
    if a.gen_lanes == 2 and (a.serial or a.lane2_gpu != a.post_gpu or a.lane2_gpu == a.gen_gpu):
        raise SystemExit("--gen-lanes 2 needs --lane2-gpu == --post-gpu (different from --gen-gpu) and no --serial")
    if a.dump_mode == "raw":
        bad = [t for t in split_cmd(a.replay_extra) if t in ("--no-remesh", "--fill-holes", "--no-weld", "--no-fill")]
        if bad:
            raise SystemExit("--dump-mode raw needs the full replay chain; remove %s from --replay-extra" % " ".join(bad))
    return cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
