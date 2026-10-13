#!/usr/bin/env python3
"""Draft then refine (OPT-IN controller for a running `arctic_batch.py run --gen-lanes 2 --queue-dir Q`; stdlib only).

For each prop it submits one cheap DRAFT job per candidate image (priority 0, per-job --res 512 etc.), waits for the draft GLBs,
scores them with a judge command, and submits only the top K candidates as REFINE jobs (priority 100, no per-job options, so
they are the batch's own production command and their output is hash-equal to a plain run of the same image and seed).
The pick on the refined GLB stays with the person; a draft is a screen for shape and size, not for look.

  python draft_refine.py --queue-dir Q --out OUT --manifest props.json --judge-cmd "python judge.py {glb}" [--keep 1] [--close]
  props.json: {"chest": ["a.png", "b.png", "c.png"], "bunk": ["d.png", "e.png"]}      (paths relative to the manifest)

The judge command prints a number on its last line (higher = better); a non-zero exit, or `veto` / `nan`, drops the candidate.
Draft flags default to the design note's profile: --res 512 --faces 150000 and cli-extra "--steps 12 --steps-ss 24".
Writes OUT/draft_refine.json (scores, keepers, seconds per stage). Draft jobs are named <prop>_<n>_d, refine jobs <prop>_<n>.
"""
import argparse, json, os, shlex, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import live_queue

DRAFT_OPTS = {"res": 512, "faces": 150000, "cli_extra": "--steps 12 --steps-ss 24"}


def load_manifest(path):
    base = os.path.dirname(os.path.abspath(path))
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    out = {}
    for prop, imgs in m.items():
        if not live_queue.NAME_RE.match(prop) or not imgs:
            raise ValueError("bad prop %r" % prop)
        out[prop] = [i if os.path.isabs(i) else os.path.join(base, i) for i in imgs]
        for i in out[prop]:
            if not os.path.isfile(i):
                raise ValueError("image not found: " + i)
    return out


def score_glb(judge_cmd, glb, image, timeout=600):
    """Returns a float, or None when the judge vetoes (non-zero exit, `veto`, `nan`, no number)."""
    cmd = shlex.split(judge_cmd.format(glb=glb, image=image), posix=(os.name != "nt"))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [l.strip() for l in r.stdout.splitlines() if l.strip()]
    if r.returncode != 0 or not lines:
        return None
    try:
        v = float(lines[-1])
    except ValueError:
        return None
    return None if v != v else v


def wait_for(path, qdir, name, timeout_s, poll_s=0.5):
    """True when the GLB exists; False when the job was rejected or the wait timed out."""
    t0 = time.time()
    rej = os.path.join(qdir, "rejected")
    while time.time() - t0 < timeout_s:
        if os.path.isfile(path):
            return True
        try:
            if any(f.endswith(name + ".job.json") or ("-" + name + ".job.json") in f for f in os.listdir(rej)):
                return False
        except OSError:
            pass
        time.sleep(poll_s)
    return False


def run(a, log=print):
    props = load_manifest(a.manifest)
    draft_opts = dict(DRAFT_OPTS)
    if a.draft_res:
        draft_opts["res"] = a.draft_res
    if a.draft_cli_extra is not None:
        draft_opts["cli_extra"] = a.draft_cli_extra
    t_all = time.time()
    rep = {"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "keep": a.keep, "draft_opts": draft_opts, "props": {}}
    for prop, imgs in props.items():          # all drafts first: they fill both cards while earlier refines run
        for n, img in enumerate(imgs, 1):
            live_queue.submit(a.queue_dir, img, "%s_%d_d" % (prop, n), priority=0, est_s=a.draft_est_s, vram_gb=a.draft_vram_gb, opts=draft_opts)
    rep["draft_submitted_s"] = time.time() - t_all
    for prop, imgs in props.items():
        cands = []
        for n, img in enumerate(imgs, 1):
            name = "%s_%d" % (prop, n)
            t0 = time.time()
            ok = wait_for(os.path.join(a.out, name + "_d.glb"), a.queue_dir, name + "_d", a.draft_timeout)
            sc = score_glb(a.judge_cmd, os.path.join(a.out, name + "_d.glb"), img) if ok else None
            cands.append({"name": name, "image": img, "draft_ok": ok, "score": sc, "judge_s": time.time() - t0})
            log("%s: draft %s, score %s" % (name, "ok" if ok else "missing", sc))
        alive = sorted((c for c in cands if c["score"] is not None), key=lambda c: (-c["score"], c["name"]))
        keep = alive[:a.keep]
        for c in keep:
            live_queue.submit(a.queue_dir, c["image"], c["name"], priority=100, est_s=a.refine_est_s)
            c["refine_submitted"] = time.time() - t_all
        rep["props"][prop] = {"candidates": cands, "keepers": [c["name"] for c in keep]}
        log("%s: refine %s" % (prop, ", ".join(c["name"] for c in keep) or "(none survived)"))
    if a.close:
        live_queue.close(a.queue_dir)
    rep["seconds"] = time.time() - t_all
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "draft_refine.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--queue-dir", required=True); p.add_argument("--out", required=True, help="the running batch's output dir")
    p.add_argument("--manifest", required=True); p.add_argument("--judge-cmd", required=True, help="template with {glb} {image}")
    p.add_argument("--keep", type=int, default=1, help="candidates refined per prop (default 1)")
    p.add_argument("--draft-res", type=int, default=None); p.add_argument("--draft-cli-extra", default=None)
    p.add_argument("--draft-est-s", type=float, default=75); p.add_argument("--draft-vram-gb", type=float, default=None)
    p.add_argument("--refine-est-s", type=float, default=600)
    p.add_argument("--draft-timeout", type=float, default=3600, help="seconds to wait for one draft GLB")
    p.add_argument("--close", action="store_true", help="close the queue after the refine jobs are submitted")
    a = p.parse_args(argv)
    if a.keep < 1:
        p.error("--keep must be >= 1")
    rep = run(a)
    print(json.dumps({k: v["keepers"] for k, v in rep["props"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
