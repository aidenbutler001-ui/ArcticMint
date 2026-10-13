"""Minimal stand-in for Expiramental3d tools/gpulock.py: just the functions arctic_batch.GpuLocks calls.
The real module is exercised too when $GPULOCK_PY points at it (test_real_gpulock)."""
import json, os
from pathlib import Path


def lock_dir():
    d = Path(os.environ["GPULOCK_DIR"]); d.mkdir(parents=True, exist_ok=True); return d


def lock_path(g): return lock_dir() / ("gpu%d.lock" % g)


def res_path(g): return lock_dir() / ("gpu%d.reserved" % g)


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return None


def acquire(gpu, owner, what="", session="", wait=0, log=print):
    try:
        fd = os.open(str(lock_path(gpu)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        log("fake gpulock: GPU %d held" % gpu)
        return False
    with os.fdopen(fd, "w") as f:
        json.dump({"owner": owner, "pid": os.getpid()}, f)
    return True


def release(gpu, owner=None, pid=None, force=False, log=print):
    i = read_json(lock_path(gpu))
    if i and (force or i.get("owner") == owner):
        lock_path(gpu).unlink()
    return True


def reserve(gpu, owner, hours, reason=""):
    r = read_json(res_path(gpu))
    if r and r.get("owner") != owner:
        return False, "GPU %d is already reserved by %s" % (gpu, r["owner"])
    res_path(gpu).write_text(json.dumps({"owner": owner, "hours": hours}))
    return True, {}


def unreserve(gpu, owner):
    if res_path(gpu).exists():
        res_path(gpu).unlink()
    return True


# -- the Space Gothic slot (gpulock want / unwant / foreign_demand), same shape as the real module -----------------------
import time as _time


def want_path(g): return lock_dir() / ("gpu%d.wanted" % g)


def active_want(g):
    w = read_json(want_path(g))
    return w if w and w.get("until", 0) > _time.time() else None


def want(gpu, owner, minutes=40, what=""):
    cur = active_want(gpu)
    if cur and cur["owner"] != owner:
        return False, "GPU %d is already wanted by %s" % (gpu, cur["owner"])
    d = {"owner": owner, "what": what, "until": _time.time() + minutes * 60}
    want_path(gpu).write_text(json.dumps(d))
    return True, d


def unwant(gpu, owner):
    w = read_json(want_path(gpu))
    if w and w["owner"] != owner:
        return False
    if want_path(gpu).exists():
        want_path(gpu).unlink()
    return True


def foreign_demand(gpu, owner):
    w = active_want(gpu)
    if w and w["owner"] != owner:
        return {"kind": "want", "owner": w["owner"], "what": w.get("what", ""), "until_utc": ""}
    r = read_json(res_path(gpu))
    if r and r["owner"] != owner:
        return {"kind": "reservation", "owner": r["owner"], "what": "", "until_utc": ""}
    return None
