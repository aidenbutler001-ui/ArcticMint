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
