// dual_grid_to_mesh equivalence + timing: builds a synthetic ShapeOut from a --dump-post file
// (voxel coords from the mesh vertices at --res, parent-major order like the C2S decoder, random
// head features) and compares the legacy path with the parallel one.
//   dualgrid-bench <dump.bin> [res=1024]
#include "dual_grid.h"
#include "post_parallel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#ifdef _WIN32
static void set_env(const char* k, const char* v) { _putenv_s(k, v ? v : ""); }
#else
static void set_env(const char* k, const char* v) { if (v) setenv(k, v, 1); else unsetenv(k); }
#endif
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dualgrid-bench <dump.bin> [res]\n"); return 1; }
    const int res = argc > 2 ? atoi(argv[2]) : 1024;
    FILE* f = fopen(argv[1], "rb");
    if (!f) return 1;
    int V, F, Mv, r0;
    if (fread(&V, 4, 1, f) + fread(&F, 4, 1, f) + fread(&Mv, 4, 1, f) + fread(&r0, 4, 1, f) != 4) return 1;
    std::vector<float> verts((size_t)V * 3);
    if (fread(verts.data(), 4, verts.size(), f) != verts.size()) return 1;
    fclose(f);
    // unique voxel coords of the vertices, ordered parent-major (coords>>1, then child) like C2S output
    std::vector<uint64_t> keys;
    keys.reserve(V);
    for (int v = 0; v < V; ++v) {
        int c[3];
        for (int a = 0; a < 3; ++a) c[a] = std::min(res - 1, std::max(0, (int)std::floor((verts[3*(size_t)v+a] + 0.5f) * res)));
        const uint64_t parent = ((uint64_t)(c[0] >> 1) << 40) | ((uint64_t)(c[1] >> 1) << 20) | (uint64_t)(c[2] >> 1);
        const uint64_t child = (uint64_t)(((c[0] & 1) << 2) | ((c[1] & 1) << 1) | (c[2] & 1));
        keys.push_back(parent * 8 + child);   // parent < 2^60, fits
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    trellis::ShapeOut so;
    so.res = res;
    const int N = (int)keys.size();
    so.coords.resize(N);
    for (int i = 0; i < N; ++i) {
        const uint64_t child = keys[i] & 7, parent = keys[i] >> 3;
        so.coords[i] = { (int)(parent >> 40) * 2 + (int)(child >> 2), (int)((parent >> 20) & 0xFFFFF) * 2 + (int)((child >> 1) & 1),
                         (int)(parent & 0xFFFFF) * 2 + (int)(child & 1) };
    }
    if (argc > 3 && std::string(argv[3]) == "dup")   // a few duplicate coordinates: last one must win
        for (int i = 0; i + 1000 < N; i += 997) so.coords[i + 1000] = so.coords[i];
    std::mt19937 rng(7);
    std::normal_distribution<float> G(0.f, 2.f);
    so.feats7.resize((size_t)N * 7);
    for (auto& x : so.feats7) x = G(rng);
    printf("voxels N=%d res=%d\n", N, res);

    set_env("TRELLIS_POST_LEGACY", "1");
    double t = now();
    trellis::Mesh ref = trellis::dual_grid_to_mesh(so);
    const double tl = now() - t;
    set_env("TRELLIS_POST_LEGACY", nullptr);
    printf("legacy: %.2fs  V=%d F=%d\n", tl, ref.V(), ref.F());
    int bad = 0;
    for (int th : {1, 2, 4, 7, 16}) {
        set_env("TRELLIS_POST_THREADS", std::to_string(th).c_str());
        t = now();
        trellis::Mesh m = trellis::dual_grid_to_mesh(so);
        const double tn = now() - t;
        const bool same = m.verts.size() == ref.verts.size() && m.faces.size() == ref.faces.size() &&
                          !std::memcmp(m.verts.data(), ref.verts.data(), m.verts.size() * 4) &&
                          !std::memcmp(m.faces.data(), ref.faces.data(), m.faces.size() * 4);
        printf("threads %2d: %.2fs  %s\n", th, tn, same ? "identical" : "MISMATCH");
        bad += !same;
    }
    return bad ? 1 : 0;
}
