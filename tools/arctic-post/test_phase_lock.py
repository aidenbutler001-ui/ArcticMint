"""Tests for include/phase_lock.h and its use in decimate_qem.cpp (cloud / Linux; the _WIN32 branch is checked on the PC, T9).

Needs the binaries phase-lock-test and qem-lock-test: set ARCTIC_POST_BUILD to a build dir made with
  cmake -S tools/arctic-post -B build-post -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-post phase-lock-test qem-lock-test
or leave it unset and the test builds them into a temp dir (needs cmake and ninja, ~30 s). Run:
  python3 -m unittest tools/arctic-post/test_phase_lock.py
"""
import os, shutil, signal, subprocess, sys, tempfile, time, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = ".exe" if os.name == "nt" else ""
_build = None


def build_dir():
    global _build
    if _build:
        return _build
    d = os.environ.get("ARCTIC_POST_BUILD")
    if not d:
        if not (shutil.which("cmake") and shutil.which("ninja")):
            raise unittest.SkipTest("set ARCTIC_POST_BUILD or install cmake + ninja")
        d = tempfile.mkdtemp(prefix="phase-lock-build-")
        subprocess.run(["cmake", "-S", HERE, "-B", d, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release"], check=True, capture_output=True)
        subprocess.run(["ninja", "-C", d, "phase-lock-test", "qem-lock-test"], check=True, capture_output=True)
    _build = d
    return d


def exe(name):
    return os.path.join(build_dir(), name + EXE)


def wait_held(proc):
    """Read the holder's stdout until its own 'held=' line (the lock's '[phase-lock] ... acquired' line comes first)."""
    for line in proc.stdout:
        if line.startswith("held="):
            return line
    raise AssertionError("holder exited without printing held=")


def parse_qem(text):
    t = {}
    for line in text.splitlines():
        p = line.split()
        if p and p[0] in ("QEM_ENTER", "QEM_EXIT"):
            t.setdefault(p[1], {})[p[0]] = int(p[2])
    return t


class PhaseLockCpp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory()
        self.lock = os.path.join(self.d.name, "gpu0.phase")

    def tearDown(self):
        self.d.cleanup()

    def env(self, **kw):
        e = dict(os.environ)
        e.pop("ARCTIC_GPU_PHASE_LOCK", None)
        e.update(kw)
        return e

    def two_qems(self, env):
        ps = [subprocess.Popen([exe("qem-lock-test")], env=env, stdout=subprocess.PIPE, text=True) for _ in range(2)]
        return [parse_qem(p.communicate()[0]) for p in ps]

    def test_qem_phases_do_not_overlap_with_the_lock(self):
        a, b = self.two_qems(self.env(ARCTIC_GPU_PHASE_LOCK=self.lock, QEM_STUB_MS="700"))
        (ea,), (eb,) = a.values(), b.values()
        first, second = sorted((ea, eb), key=lambda x: x["QEM_ENTER"])
        self.assertLessEqual(first["QEM_EXIT"], second["QEM_ENTER"])
        self.assertGreaterEqual(first["QEM_EXIT"] - first["QEM_ENTER"], 650)

    def test_qem_phases_overlap_without_the_lock(self):
        a, b = self.two_qems(self.env(QEM_STUB_MS="700"))
        (ea,), (eb,) = a.values(), b.values()
        first, second = sorted((ea, eb), key=lambda x: x["QEM_ENTER"])
        self.assertLess(second["QEM_ENTER"], first["QEM_EXIT"])

    def test_gpu_failure_releases_the_lock_and_falls_back_to_cpu(self):
        r = subprocess.run([exe("qem-lock-test")], env=self.env(ARCTIC_GPU_PHASE_LOCK=self.lock, QEM_STUB_MS="50", QEM_STUB_FAIL="1"),
                           capture_output=True, text=True)
        self.assertIn("GPU decimation unavailable", r.stderr); self.assertIn("DONE faces=", r.stdout)
        w = subprocess.run([exe("phase-lock-test"), "wait", self.lock], capture_output=True, text=True, timeout=10)
        self.assertIn("held=1", w.stdout)

    def test_killed_holder_frees_the_lock(self):
        h = subprocess.Popen([exe("phase-lock-test"), "hold", self.lock, "60"], stdout=subprocess.PIPE, text=True)
        self.assertIn("held=1", wait_held(h))
        t0 = time.time()
        h.kill(); h.wait()
        w = subprocess.run([exe("phase-lock-test"), "wait", self.lock], capture_output=True, text=True, timeout=15)
        self.assertIn("held=1", w.stdout); self.assertLess(time.time() - t0, 5)

    def test_waiter_gets_the_lock_when_the_holder_finishes(self):
        h = subprocess.Popen([exe("phase-lock-test"), "hold", self.lock, "1.2"], stdout=subprocess.PIPE, text=True)
        wait_held(h)
        w = subprocess.run([exe("phase-lock-test"), "wait", self.lock], capture_output=True, text=True, timeout=15)
        h.wait()
        waited = float(w.stdout.split("waited=")[1].split()[0])
        self.assertGreater(waited, 0.5); self.assertIn("held=1", w.stdout)

    def test_timeout_proceeds_without_the_lock(self):
        h = subprocess.Popen([exe("phase-lock-test"), "hold", self.lock, "20"], stdout=subprocess.PIPE, text=True)
        wait_held(h)
        try:
            w = subprocess.run([exe("phase-lock-test"), "wait", self.lock], capture_output=True, text=True, timeout=15,
                               env=self.env(ARCTIC_GPU_PHASE_LOCK_TIMEOUT_S="1"))
        finally:
            h.kill(); h.wait()
        self.assertIn("held=0", w.stdout); self.assertIn("running without the lock", w.stderr)

    def test_unset_variable_is_a_no_op(self):
        w = subprocess.run([exe("qem-lock-test")], env=self.env(QEM_STUB_MS="10"), capture_output=True, text=True)
        self.assertNotIn("phase-lock", w.stdout + w.stderr)


if __name__ == "__main__":
    unittest.main()
