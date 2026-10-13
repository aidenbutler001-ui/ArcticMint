// phase-lock-test: exercises include/phase_lock.h from tools/arctic-post/test_phase_lock.py (and by hand on the PC, T9).
//   phase-lock-test hold PATH SECONDS   take the lock, print "held", keep it SECONDS, exit
//   phase-lock-test wait PATH           take the lock (waiting as long as needed), print "acquired waited=<s> held=<0|1>", exit
// ARCTIC_GPU_PHASE_LOCK_TIMEOUT_S shortens the wait; after it the lock is skipped (held=0), by design.
#include "phase_lock.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: phase-lock-test hold PATH SECONDS | wait PATH\n"); return 2; }
    trellis::PhaseLock lock("test", argv[2]);
    if (!std::strcmp(argv[1], "hold")) {
        std::printf("held=%d\n", lock.held() ? 1 : 0);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * (argc > 3 ? std::atof(argv[3]) : 1.0))));
        return 0;
    }
    std::printf("acquired waited=%.3f held=%d\n", lock.waited_s(), lock.held() ? 1 : 0);
    std::fflush(stdout);
    return 0;
}
