#!/usr/bin/env python3
"""Resource-aware dispatcher for arctic-batch (stdlib only, no subprocess, no I/O): the scheduling rules and the VRAM
admission gate, kept apart from arctic_batch.py so the simulator and the runner use the very same code.

Lanes     : one generation lane per GPU. The primary lane (the 4070 SUPER) takes jobs in queue order. A secondary lane (the
            3060 Ti) takes a job only when that lowers the projected makespan (see Dispatcher.choose).
VramGate  : per-GPU admission. Every GPU activity (a generation, a post-replay QEM, a Cycles bake) declares how many GB it
            needs; it starts only when baseline + running + need <= capacity - margin, otherwise it waits (or the caller picks
            the other card). The declared needs come from measurements (item 11: generation 4.3-4.5 GB, post 3.0 GB, finish
            1.7 GB on the 3060 Ti), not from guesses, and live in arctic_batch's --*-need-gb flags.

CLI:  dispatcher.py sim --csv batch.csv [--speed 0.5 --cpu-slow 1.0 --placement vram|strict|off]   (what-if table)
      dispatcher.py threshold --csv batch.csv [--min-gain 0.10]                                   (lowest speed that pays)
"""
import argparse, csv, heapq, statistics, sys, threading, time


class Lane:
    def __init__(self, name, gpu, speed=1.0, cap_gb=12.0, baseline_gb=0.0, primary=False):
        self.name, self.gpu, self.speed, self.cap_gb, self.baseline_gb, self.primary = name, gpu, speed, cap_gb, baseline_gb, primary
        self.job, self.started, self.est_end = None, None, None   # current job, start time, projected end


class Job:
    def __init__(self, name, est_s=None, vram_gb=4.5, priority=0, pin=None):
        self.name, self.est_s, self.vram_gb, self.priority, self.pin = name, est_s, vram_gb, priority, pin


class Dispatcher:
    """Not thread-safe by itself: the runner calls it under its condition variable."""

    def __init__(self, lanes, margin_gb=0.4, min_gain_s=30.0, default_speed=0.5, longest_first=False):
        self.lanes = list(lanes)
        self.longest_first = longest_first
        self.margin_gb, self.min_gain_s = margin_gb, min_gain_s
        self.primary = next(l for l in self.lanes if l.primary)
        for l in self.lanes:
            if not l.primary and not l.speed:
                l.speed = default_speed
        self.samples = {}   # lane name -> list of observed speeds (est / actual, relative to the primary)

    def fits(self, job, lane):
        return job.vram_gb <= lane.cap_gb - lane.baseline_gb - self.margin_gb

    def allowed(self, job, lane):
        return (job.pin is None or job.pin == lane.name) and self.fits(job, lane)

    def order(self, pending):
        """Highest priority first; within a priority queue (arrival) order, or with longest_first the slowest job first
        (a job with no estimate counts as the median known one). Stable either way."""
        if not self.longest_first:
            return sorted(pending, key=lambda j: -j.priority)
        known = [j.est_s for j in pending if j.est_s is not None]
        mid = statistics.median(known) if known else 0.0
        return sorted(pending, key=lambda j: (-j.priority, -(j.est_s if j.est_s is not None else mid)))

    def _est(self, job, pending):
        if job.est_s is not None:
            return job.est_s
        known = [j.est_s for j in pending if j.est_s is not None]
        return statistics.median(known) if known else 1.0

    def primary_free_at(self, now):
        l = self.primary
        if l.job is None:
            return now
        return max(now, l.est_end if l.est_end is not None else now)

    def choose(self, now, lane, pending):
        """The job `lane` should start now, or None (idle for now). pending is a list of Job in queue order."""
        pend = self.order(pending)
        cands = [j for j in pend if self.allowed(j, lane)]
        if not cands:
            return None
        if lane.primary:
            return cands[0]
        pinned = [j for j in cands if j.pin == lane.name]
        if pinned:
            return pinned[0]
        # secondary lane: take job j only if max(finish on this lane, primary finishing everything else) beats the primary
        # doing all of it by min_gain_s. The tail matters: a big job on a slow lane at the end would be the long pole.
        ests = {j.name: self._est(j, pend) for j in pend}
        total, t_a = sum(ests.values()), self.primary_free_at(now)
        alone = t_a + total
        best, best_end = None, alone - self.min_gain_s
        for j in cands:
            e = ests[j.name]
            end = max(now + e / max(lane.speed, 1e-6), t_a + total - e)
            if end < best_end or (best is not None and abs(end - best_end) < 1e-9 and e < ests[best.name]):   # tie: the smaller job
                best, best_end = j, end
        return best

    def observe(self, lane, job, actual_s):
        """Learn the secondary lane's speed from a finished job whose estimate came from the primary lane."""
        if lane.primary or job.est_s is None or actual_s <= 0:
            return
        s = self.samples.setdefault(lane.name, [])
        s.append(job.est_s / actual_s)
        lane.speed = sum(s[-3:]) / len(s[-3:])


class VramGate:
    """Per-GPU admission. admit()/release() are thread-safe; wait_admit() blocks (polling a condition) until admitted,
    the timeout passes (returns False) or `stop` is set."""

    def __init__(self, cap_gb, baseline_gb=None, margin_gb=0.4):
        self.cap, self.base, self.margin = dict(cap_gb), dict(baseline_gb or {}), margin_gb
        self.running = {g: {} for g in self.cap}   # gpu -> {key: need_gb}
        self.shared = {}
        self.blocked = set()   # cards handed to another owner (slot yield): nothing is admitted until unblock()
        self.cond = threading.Condition()

    def _free(self, gpu):
        if gpu in self.blocked:
            return -1e9
        return self.cap[gpu] - self.base.get(gpu, 0.0) - self.margin - sum(self.running[gpu].values())

    def block(self, gpu):
        with self.cond:
            self.blocked.add(gpu)

    def unblock(self, gpu):
        with self.cond:
            self.blocked.discard(gpu)
            self.cond.notify_all()

    def is_blocked(self, gpu):
        return gpu in self.blocked

    def idle(self, gpu):
        """Nothing of ours holds a reservation on this card (no generation, post, finish)."""
        with self.cond:
            return not self.running[gpu]

    def can_admit(self, gpu, need):
        with self.cond:
            return need <= self._free(gpu) + 1e-9

    def admit(self, gpu, key, need):
        with self.cond:
            if need > self._free(gpu) + 1e-9:
                return False
            self.running[gpu][key] = need
            return True

    def wait_admit(self, gpu, key, need, timeout=None, stop=None, poll=2.0):
        t_end = None if timeout is None else time.time() + timeout
        with self.cond:
            while True:
                if need <= self._free(gpu) + 1e-9:
                    self.running[gpu][key] = need
                    return True
                if (stop is not None and stop.is_set()) or (t_end is not None and time.time() >= t_end):
                    return False
                self.cond.wait(poll)

    def release(self, gpu, key):
        with self.cond:
            self.running[gpu].pop(key, None)
            self.cond.notify_all()

    # A shared slot is one reservation used by several processes that never need the card at the same moment (post-replay's
    # QEM and Blender's bake are serialised by the phase lock): the first holder reserves `need`, the last one frees it.
    def acquire_shared(self, gpu, key, need, force=False):
        with self.cond:
            n = self.shared.get((gpu, key), 0)
            if n == 0:
                if not force and need > self._free(gpu) + 1e-9:
                    return False
                self.running[gpu][key] = need
            self.shared[(gpu, key)] = n + 1   # later holders join the slot the first one reserved
            return True

    def release_shared(self, gpu, key):
        with self.cond:
            n = self.shared.get((gpu, key), 0) - 1
            if n <= 0:
                self.shared.pop((gpu, key), None)
                self.running[gpu].pop(key, None)
            else:
                self.shared[(gpu, key)] = n
            self.cond.notify_all()


# -- simulator (same Dispatcher; durations are the measured ones) ------------------------------------------------------

def simulate(jobs, lanes_n=2, speed=0.5, cpu_slow=1.0, placement="vram", margin_gb=0.4, min_gain_s=30.0,
             order=None, post_workers=1, finish_workers=1):
    """jobs: list of dict(name, gen_s (on the primary), post_s, finish_s, vram_gb). Returns (wall, plan).
    placement: 'off'    = lane 2 generates and post/finish ignore it (optimistic: assumes everything fits);
               'strict' = post and finish cannot run while lane 2 generates, and lane 2 starts only with post/finish idle
                          and nothing queued (the case where phase work can only use the 3060 Ti);
               'vram'   = post/finish run beside lane 2 (admission by declared VRAM: assumed to fit, see the design note).
    cpu_slow multiplies post/finish durations that overlap a lane-2 generation (CPU and memory-bandwidth sharing)."""
    lanes = [Lane("A", 1, 1.0, 12.0, 1.4, primary=True)]
    if lanes_n > 1:
        lanes.append(Lane("B", 0, speed, 8.0, 0.0))
    disp = Dispatcher(lanes, margin_gb, min_gain_s)
    seq = list(jobs) if order is None else sorted(jobs, key=order)
    pend = [Job(j["name"], j["gen_s"], j.get("vram_gb", 4.5)) for j in seq]
    by = {j["name"]: j for j in jobs}
    t, ev, postq, finq = 0.0, [], [], []     # ev: heap of (end_time, seq, kind, name, lane)
    n = 0
    post_busy = fin_busy = 0
    plan, done = [], 0
    lane_b = lanes[1] if lanes_n > 1 else None
    while True:
        started = True
        while started:
            started = False
            b_gen = lane_b is not None and lane_b.job is not None
            if finq and fin_busy < finish_workers and not (placement == "strict" and b_gen):
                nm = finq.pop(0); d = by[nm]["finish_s"] * (cpu_slow if b_gen else 1.0)
                n += 1; heapq.heappush(ev, (t + d, n, "fin", nm, None)); fin_busy += 1; started = True
            if postq and post_busy < post_workers and not (placement == "strict" and b_gen):
                nm = postq.pop(0); d = by[nm]["post_s"] * (cpu_slow if b_gen else 1.0)
                n += 1; heapq.heappush(ev, (t + d, n, "post", nm, None)); post_busy += 1; started = True
            for lane in lanes:
                if lane.job is not None or not pend:
                    continue
                if lane is lane_b and placement == "strict" and (post_busy or fin_busy or postq or finq):
                    continue
                j = disp.choose(t, lane, pend)
                if j is None:
                    continue
                pend.remove(j)
                d = by[j.name]["gen_s"] / lane.speed if lane is lane_b else by[j.name]["gen_s"]
                if lane is lane_b:
                    d = by[j.name]["gen_s"] / speed       # the true speed, whatever the dispatcher believes
                lane.job, lane.started, lane.est_end = j, t, t + (j.est_s or 0) / lane.speed
                n += 1; heapq.heappush(ev, (t + d, n, "gen", j.name, lane)); plan.append((j.name, lane.name, t, t + d)); started = True
        if not ev:
            break
        t, _, kind, nm, lane = heapq.heappop(ev)
        if kind == "gen":
            lane.job = None; postq.append(nm)
        elif kind == "post":
            post_busy -= 1; finq.append(nm)
        else:
            fin_busy -= 1; done += 1
    return t, plan


def load_jobs(path):
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r.get("status") == "ok" and r.get("gen_s")]
    return [dict(name=r["asset"], gen_s=float(r["gen_s"]), post_s=float(r["post_s"]), finish_s=float(r["finish_s"])) for r in rows]


def threshold(jobs, min_gain=0.10, **kw):
    """Lowest lane-2 speed (relative to the primary) at which the wall drops by at least min_gain, per placement."""
    base, _ = simulate(jobs, lanes_n=1)
    out = {}
    for pl in ("off", "vram", "strict"):
        lo = None
        for k in range(10, 101, 5):
            w, _ = simulate(jobs, 2, k / 100.0, placement=pl, **kw)
            if w <= base * (1 - min_gain):
                lo = k / 100.0
                break
        out[pl] = lo
    return base, out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sim"); s.add_argument("--csv", required=True); s.add_argument("--speed", type=float, default=0.5)
    s.add_argument("--cpu-slow", type=float, default=1.0); s.add_argument("--placement", default="vram", choices=["off", "vram", "strict"])
    s.add_argument("--longest-first", action="store_true")
    t = sub.add_parser("threshold"); t.add_argument("--csv", required=True); t.add_argument("--min-gain", type=float, default=0.10)
    t.add_argument("--cpu-slow", type=float, default=1.0)
    a = p.parse_args(argv)
    jobs = load_jobs(a.csv)
    if a.cmd == "sim":
        key = (lambda j: -j["gen_s"]) if a.longest_first else None
        base, _ = simulate(jobs, 1, order=key)
        w, plan = simulate(jobs, 2, a.speed, a.cpu_slow, a.placement, order=key)
        print("one lane %.0f s; two lanes (speed %.2f, %s, cpu x%.1f) %.0f s (%+.1f%%)" % (base, a.speed, a.placement, a.cpu_slow, w, 100 * (w - base) / base))
        for nm, ln, t0, t1 in plan:
            print("  %-24s lane %s %6.0f -> %6.0f" % (nm, ln, t0, t1))
    else:
        base, th = threshold(jobs, a.min_gain, cpu_slow=a.cpu_slow)
        print("one lane %.0f s; lowest lane-2 speed for >= %.0f%% faster: %s" % (base, 100 * a.min_gain, th))
    return 0


if __name__ == "__main__":
    sys.exit(main())
