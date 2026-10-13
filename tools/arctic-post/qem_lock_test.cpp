// qem-lock-test: decimate_qem.cpp compiled WITH TRELLIS_HAVE_GPU_DECIMATE and a stub decimate_qem_gpu, so the real call site
// (the PhaseLock around the GPU phase) runs in a CPU-only build. The stub logs "QEM_ENTER <pid> <ms>" and "QEM_EXIT <pid> <ms>"
// (wall-clock milliseconds) and sleeps QEM_STUB_MS (default 1500); QEM_STUB_FAIL=1 makes it return false (OOM-like), after
// which decimate_qem must release the lock and take the CPU path. Two simultaneous runs on one ARCTIC_GPU_PHASE_LOCK must not
// overlap between ENTER and EXIT; with the variable unset they must.
#include "uv_bake.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#ifdef _WIN32
#  include <windows.h>
static unsigned long pid_() { return GetCurrentProcessId(); }
#else
#  include <unistd.h>
static unsigned long pid_() { return (unsigned long)getpid(); }
#endif

static long long wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

namespace trellis {
bool decimate_qem_gpu(const std::vector<float>& verts, int V, const std::vector<int32_t>& faces, int F,
                      int target_faces, std::vector<float>& ov, std::vector<int32_t>& of) {
    (void)verts; (void)V; (void)faces; (void)F; (void)target_faces; (void)ov; (void)of;
    std::printf("QEM_ENTER %lu %lld\n", pid_(), wall_ms());
    std::fflush(stdout);
    const char* ms = std::getenv("QEM_STUB_MS");
    std::this_thread::sleep_for(std::chrono::milliseconds(ms ? std::atoi(ms) : 1500));
    std::printf("QEM_EXIT %lu %lld\n", pid_(), wall_ms());
    std::fflush(stdout);
    if (std::getenv("QEM_STUB_FAIL")) return false;
    ov = verts; of = faces;
    return true;
}
}

int main() {
    // a flat grid, 2*N*N faces, decimated to a quarter
    const int N = 20;
    std::vector<float> v; std::vector<int32_t> f;
    for (int y = 0; y <= N; ++y) for (int x = 0; x <= N; ++x) { v.push_back((float)x); v.push_back((float)y); v.push_back(0.f); }
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const int a = y * (N + 1) + x, b = a + 1, c = a + N + 1, d = c + 1;
        f.insert(f.end(), {a, b, c, b, d, c});
    }
    std::vector<float> ov; std::vector<int32_t> of;
    trellis::decimate_qem(v, (N + 1) * (N + 1), f, (int)f.size() / 3, (int)f.size() / 12, ov, of);
    std::printf("DONE faces=%zu\n", of.size() / 3);
    return 0;
}
