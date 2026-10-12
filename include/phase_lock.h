// phase_lock.h — serialise the post GPU's (3060 Ti) VRAM-heavy phases across processes.
//
// A batch run overlaps post-replay (GPU QEM, 3-5 GB) with the Blender finish (Cycles bake, 2-4 GB) on an 8 GB card with no
// sysmem fallback. Two at once can OOM; a QEM OOM silently falls back to a 15-25 min CPU decimation. When the environment
// variable ARCTIC_GPU_PHASE_LOCK names a file, PhaseLock holds an exclusive OS file lock on byte 0 of it for its lifetime:
//
//     { trellis::PhaseLock lock("qem"); ok = decimate_qem_gpu(...); }
//
// The same file is taken by finish/finish.py around the bake through tools/phaselock.py (flock / msvcrt.locking byte 0).
// The OS drops the lock when the holder exits or is killed, so there is no stale-lock rule. Unset or empty variable = no-op,
// so default behaviour (and the same-output rule) is unchanged. After ARCTIC_GPU_PHASE_LOCK_TIMEOUT_S (default 900) the
// caller proceeds without the lock and says so: a stuck holder must not hang a batch.
//
// Byte 0 is the lock; a human-readable holder line is written at offset 64 (never inside the locked byte, which Windows
// makes unreadable to other handles). Header-only; POSIX flock and Win32 LockFileEx.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#else
#  include <fcntl.h>
#  include <sys/file.h>
#  include <unistd.h>
#endif

namespace trellis {

class PhaseLock {
public:
    static constexpr const char* kEnv = "ARCTIC_GPU_PHASE_LOCK";
    static constexpr const char* kTimeoutEnv = "ARCTIC_GPU_PHASE_LOCK_TIMEOUT_S";

    // path == nullptr reads $ARCTIC_GPU_PHASE_LOCK. what = "qem" etc., logged and written as the holder line.
    explicit PhaseLock(const char* what, const char* path = nullptr) : what_(what ? what : "") {
        const char* p = path ? path : std::getenv(kEnv);
        if (p && *p) acquire(p);
    }
    ~PhaseLock() { release(); }
    PhaseLock(const PhaseLock&) = delete;
    PhaseLock& operator=(const PhaseLock&) = delete;

    bool active() const { return active_; }   // a lock file was configured
    bool held() const { return held_; }       // and this object holds it
    double waited_s() const { return waited_; }

private:
    std::string what_;
    bool active_ = false, held_ = false;
    double waited_ = 0;
#ifdef _WIN32
    HANDLE h_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif

    static double timeout_s() {
        const char* t = std::getenv(kTimeoutEnv);
        const double v = t ? std::atof(t) : 0.0;
        return v > 0 ? v : 900.0;
    }

    std::string holder_line() const {
        char ts[32];
        const std::time_t now = std::time(nullptr);
        std::strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
#ifdef _WIN32
        const unsigned long pid = GetCurrentProcessId();
#else
        const unsigned long pid = (unsigned long)getpid();
#endif
        char buf[256];
        std::snprintf(buf, sizeof buf, "what=%s pid=%lu since=%s\n", what_.c_str(), pid, ts);
        return buf;
    }

    bool try_lock() {
#ifdef _WIN32
        OVERLAPPED ov;
        std::memset(&ov, 0, sizeof ov);
        return LockFileEx(h_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &ov) != 0;
#else
        return flock(fd_, LOCK_EX | LOCK_NB) == 0;
#endif
    }

    void write_holder() {
        const std::string line = holder_line();
#ifdef _WIN32
        OVERLAPPED ov;
        std::memset(&ov, 0, sizeof ov);
        ov.Offset = 64;
        DWORD n = 0;
        WriteFile(h_, line.data(), (DWORD)line.size(), &n, &ov);
#else
        if (pwrite(fd_, line.data(), line.size(), 64) < 0) { /* diagnostic only */ }
#endif
    }

    void acquire(const char* path) {
        active_ = true;
#ifdef _WIN32
        const int wn = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
        std::wstring wp((size_t)(wn > 0 ? wn : 1), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path, -1, &wp[0], wn);
        h_ = CreateFileW(wp.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                         nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h_ == INVALID_HANDLE_VALUE) {
#else
        fd_ = open(path, O_RDWR | O_CREAT, 0666);
        if (fd_ < 0) {
#endif
            std::fprintf(stderr, "[phase-lock] %s: cannot open %s; running without the lock\n", what_.c_str(), path);
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const double limit = timeout_s();
        bool warned = false;
        for (;;) {
            if (try_lock()) { held_ = true; break; }
            waited_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (!warned) {
                std::printf("[phase-lock] %s: waiting for %s\n", what_.c_str(), path);
                std::fflush(stdout);
                warned = true;
            }
            if (waited_ >= limit) {
                std::fprintf(stderr, "[phase-lock] %s: still locked after %.0fs; running without the lock\n", what_.c_str(), waited_);
                close_file();
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        waited_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        write_holder();
        std::printf("[phase-lock] %s: acquired after %.1fs\n", what_.c_str(), waited_);
        std::fflush(stdout);
    }

    void close_file() {
#ifdef _WIN32
        if (h_ != INVALID_HANDLE_VALUE) { CloseHandle(h_); h_ = INVALID_HANDLE_VALUE; }
#else
        if (fd_ >= 0) { close(fd_); fd_ = -1; }
#endif
    }

    void release() {
        if (held_) {
#ifdef _WIN32
            OVERLAPPED ov;
            std::memset(&ov, 0, sizeof ov);
            UnlockFileEx(h_, 0, 1, 0, &ov);
#else
            flock(fd_, LOCK_UN);
#endif
            held_ = false;
        }
        close_file();
    }
};

}  // namespace trellis
