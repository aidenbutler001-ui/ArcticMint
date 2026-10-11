// Randomized equivalence test: Arctic post stages (post_parallel / post_mesh)
// vs the legacy single-threaded code (TRELLIS_POST_LEGACY=1), at several
// thread counts. Exits non-zero on the first mismatch.
//   post-equiv-test [iterations]
#include "uv_bake.h"
#include "tri_bvh.h"
#include "remesh_dc.h"
#include "post_parallel.h"

#include <algorithm>
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

static const int kThreads[] = {1, 2, 3, 4, 7, 16};
static int g_fail = 0;

template <class T>
static bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

struct M { std::vector<float> v; std::vector<int32_t> f; };

// Run fn(mesh) legacy once and new at each thread count; compare.
template <class Fn>
static void check(const char* name, int it, const M& in, Fn fn) {
    set_env("TRELLIS_POST_LEGACY", "1");
    M ref = in; const long long rr = fn(ref);
    set_env("TRELLIS_POST_LEGACY", nullptr);
    for (int t : kThreads) {
        set_env("TRELLIS_POST_THREADS", std::to_string(t).c_str());
        M m = in; const long long r = fn(m);
        if (r != rr || !same(m.v, ref.v) || !same(m.f, ref.f)) {
            fprintf(stderr, "MISMATCH %s iter %d threads %d (ret %lld vs %lld, V %zu vs %zu, F %zu vs %zu)\n", name, it, t,
                    r, rr, m.v.size() / 3, ref.v.size() / 3, m.f.size() / 3, ref.f.size() / 3);
            ++g_fail;
            return;
        }
    }
    set_env("TRELLIS_POST_THREADS", nullptr);
}

// Grid surface (n x n vertices), optionally a Moebius-like twisted strip,
// random face flips, duplicates, degenerates, extra non-manifold fins.
static M make_grid(std::mt19937& rng, int n, bool twist, float flip_p, int extra) {
    M m;
    std::uniform_real_distribution<float> U(-0.5f, 0.5f);
    for (int y = 0; y < n; ++y)
        for (int x = 0; x < n; ++x) { m.v.push_back(U(rng)); m.v.push_back(U(rng)); m.v.push_back(U(rng)); }
    auto id = [&](int x, int y) { return y * n + x; };
    std::bernoulli_distribution flip(flip_p);
    for (int y = 0; y + 1 < n; ++y)
        for (int x = 0; x + 1 < n; ++x) {
            int a = id(x, y), b = id(x + 1, y), c = id(x + 1, y + 1), d = id(x, y + 1);
            int t1[3] = {a, b, c}, t2[3] = {a, c, d};
            if (flip(rng)) std::swap(t1[1], t1[2]);
            if (flip(rng)) std::swap(t2[1], t2[2]);
            m.f.insert(m.f.end(), t1, t1 + 3); m.f.insert(m.f.end(), t2, t2 + 3);
        }
    if (twist) {   // glue last column to first column reversed: non-orientable band
        for (int y = 0; y + 1 < n; ++y) {
            int a = id(n - 1, y), b = id(0, n - 1 - y), c = id(0, n - 2 - y), d = id(n - 1, y + 1);
            m.f.insert(m.f.end(), {a, b, c, a, c, d});
        }
    }
    std::uniform_int_distribution<int> V(0, n * n - 1);
    std::uniform_int_distribution<int> kind(0, 3);
    for (int i = 0; i < extra; ++i) {
        const int k = kind(rng);
        if (k == 0 && !m.f.empty()) {   // duplicate an existing face (maybe flipped)
            const size_t f = std::uniform_int_distribution<size_t>(0, m.f.size() / 3 - 1)(rng);
            int t[3] = {m.f[3*f], m.f[3*f+1], m.f[3*f+2]};
            if (flip(rng)) std::swap(t[1], t[2]);
            m.f.insert(m.f.end(), t, t + 3);
        } else if (k == 1) {            // degenerate
            int a = V(rng), b = V(rng); m.f.insert(m.f.end(), {a, a, b});
        } else {                        // random fin / non-manifold
            int a = V(rng), b = V(rng), c = V(rng);
            if (a != b && b != c && a != c) m.f.insert(m.f.end(), {a, b, c});
        }
    }
    // shuffle face order a bit (changes DFS seeds) but deterministically
    for (size_t f = m.f.size() / 3; f > 1; --f) {
        if (!flip(rng)) continue;
        const size_t g = std::uniform_int_distribution<size_t>(0, f - 1)(rng);
        for (int j = 0; j < 3; ++j) std::swap(m.f[3*(f-1)+j], m.f[3*g+j]);
    }
    return m;
}

// Clustered points around a few centres: many near-duplicates at ~eps,
// negative coordinates (XOR key collisions in weld's cell key).
static M make_cloud(std::mt19937& rng, int nv, float step) {
    M m;
    std::uniform_real_distribution<float> U(-0.6f, 0.6f), J(-1.5f, 1.5f);
    std::vector<float> centres;
    const int nc = std::max(1, nv / 4);
    for (int i = 0; i < nc * 3; ++i) centres.push_back(U(rng));
    std::uniform_int_distribution<int> C(0, nc - 1);
    for (int i = 0; i < nv; ++i) {
        const int c = C(rng);
        for (int k = 0; k < 3; ++k) m.v.push_back(centres[3*c+k] + J(rng) * step);
    }
    std::uniform_int_distribution<int> V(0, nv - 1);
    for (int i = 0; i < nv * 2; ++i) m.f.insert(m.f.end(), {V(rng), V(rng), V(rng)});
    return m;
}

// Canonical hole order (TRELLIS_POST_CANON_HOLES=1): not byte-equal to legacy (loop order / fan apex follow
// the sorted order instead of unordered_map order), so check instead that (1) it is identical at every
// thread count, (2) it fills the same number of loops and adds the same number of faces as legacy, and
// (3) the boundary edge set afterwards equals legacy's (strict mode).
static std::vector<uint64_t> bset(const std::vector<int32_t>& f) {
    auto b = trellis::post::boundary_edges(f);
    for (auto& k : b) { uint32_t u = (uint32_t)(k >> 32), v = (uint32_t)k; if (u > v) std::swap(u, v); k = ((uint64_t)u << 32) | v; }
    std::sort(b.begin(), b.end());
    return b;
}
template <class Fn>
static void check_canon(const char* name, int it, const M& in, Fn fn, bool strict = true) {
    set_env("TRELLIS_POST_CANON_HOLES", nullptr);
    set_env("TRELLIS_POST_LEGACY", "1");
    M ref = in; const long long rr = fn(ref);
    set_env("TRELLIS_POST_LEGACY", nullptr);
    set_env("TRELLIS_POST_CANON_HOLES", "1");
    M first; long long r0 = 0; bool have = false;
    for (int t : kThreads) {
        set_env("TRELLIS_POST_THREADS", std::to_string(t).c_str());
        M m = in; const long long r = fn(m);
        if (!have) {
            first = m; r0 = r; have = true;
            if (strict && (r != rr || m.f.size() != ref.f.size() || m.v.size() != ref.v.size() || bset(m.f) != bset(ref.f))) {
                fprintf(stderr, "CANON MISMATCH vs legacy %s iter %d (filled %lld vs %lld, F %zu vs %zu)\n", name, it, r, rr,
                        m.f.size() / 3, ref.f.size() / 3);
                ++g_fail;
            }
        } else if (r != r0 || !same(m.v, first.v) || !same(m.f, first.f)) {
            fprintf(stderr, "CANON NOT DETERMINISTIC %s iter %d threads %d\n", name, it, t);
            ++g_fail;
        }
    }
    set_env("TRELLIS_POST_CANON_HOLES", nullptr);
    set_env("TRELLIS_POST_THREADS", nullptr);
}

int main(int argc, char** argv) {
    const int iters = argc > 1 ? atoi(argv[1]) : 40;
    std::mt19937 rng(12345);
    for (int it = 0; it < iters; ++it) {
        const int n = 4 + (int)(rng() % 60);
        // clean_mesh
        for (int tw = 0; tw < 2; ++tw) {
            M g = make_grid(rng, n, tw == 1, (it % 3) * 0.2f, (int)(rng() % (n * 3 + 1)));
            check("clean_mesh", it, g, [](M& m) { trellis::clean_mesh((int)m.v.size() / 3, m.f); return 0LL; });
            check("drop_small_components", it, g, [](M& m) { return (long long)trellis::drop_small_components(m.v, m.f, 0.3f); });
            check("fill_small_holes", it, g, [](M& m) { return (long long)trellis::fill_small_holes(m.f); });
            check("fill_holes", it, g, [](M& m) { return (long long)trellis::fill_holes(m.v, m.f, 1e9f); });
            check_canon("fill_small_holes", it, g, [](M& m) { return (long long)trellis::fill_small_holes(m.f); });
            // fill_holes' walk marks a whole boundary path as visited, including a cycle it runs into, so on
            // non-manifold input even the legacy set of filled loops depends on hash order: deterministic only.
            check_canon("fill_holes", it, g, [](M& m) { return (long long)trellis::fill_holes(m.v, m.f, 1e9f); }, false);
        }
        // weld (positions and colours both checked: colours ride in m.v of a second run)
        {
            const float step = 1.0f / (8.0f * (float)(64 << (it % 4)));
            M c = make_cloud(rng, 50 + (int)(rng() % 20000), step);
            check("weld_vertices", it, c, [step](M& m) { return (long long)trellis::weld_vertices(m.v, m.f, nullptr, step); });
            check("weld_vertices+colors", it, c, [step](M& m) {
                std::vector<float> col(m.v.size());
                for (size_t i = 0; i < col.size(); ++i) col[i] = (float)(i % 97);
                const long long r = trellis::weld_vertices(m.v, m.f, &col, step);
                m.v.insert(m.v.end(), col.begin(), col.end());
                return r;
            });
            check("drop_small_components(cloud)", it, c, [](M& m) { return (long long)trellis::drop_small_components(m.v, m.f, 0.5f); });
        }
        if (g_fail) break;
    }
    if (g_fail) { fprintf(stderr, "FAILED (%d)\n", g_fail); return 1; }
    printf("post-equiv-test: all %d iterations identical\n", iters);
    return 0;
}
