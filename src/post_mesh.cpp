// Arctic: sort-based, multithreaded rewrites of the CPU mesh post stages in
// uv_bake.cpp (weld_vertices, clean_mesh, drop_small_components). Each one
// reproduces the legacy function's output byte for byte, at any thread count;
// docs/arctic/post-mt.md explains why. The legacy code is still reachable
// with TRELLIS_POST_LEGACY=1.
#include "post_parallel.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <vector>

namespace trellis {
namespace post {

namespace {

// Exclusive prefix sum of per-chunk counts; returns the total.
int64_t exclusive_scan(std::vector<int64_t>& v) {
    int64_t s = 0;
    for (auto& x : v) { const int64_t t = x; x = s; s += t; }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// weld_vertices
//
// Legacy semantics: scan vertices in index order; vertex v joins the FIRST kept
// vertex c found within eps, scanning the 27 neighbour cell keys in
// (dz, dy, dx) order and, inside a key's bucket, kept vertices in creation
// (= index) order; otherwise v is kept. Cell keys are XOR-packed and can
// collide; buckets are keyed by the packed key, exactly like the legacy map.
//
// Here: (1) sort (key, index) pairs; (2) in parallel, flag every vertex that
// has ANY earlier vertex within eps in its 27 buckets ("conflicted"); vertices
// without one are kept whatever happens, because the legacy scan only ever
// sees earlier kept vertices; (3) resolve only the conflicted vertices, serially
// in index order, with the legacy scan order; (4) remap in parallel.
int weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces, std::vector<float>* colors3, float step) {
    const int V = (int)verts.size() / 3;
    if (colors3 && colors3->size() != verts.size()) {   // guard: per-vertex colors out of sync
        fprintf(stderr, "  weld: colors/verts size mismatch (%zu vs %zu), ignoring colors\n",
                colors3->size(), verts.size());
        colors3 = nullptr;
    }
    const float eps2 = step * step;
    auto ckey = [](int64_t x, int64_t y, int64_t z) {
        return ((uint64_t)(uint32_t)(int32_t)x << 42) ^ ((uint64_t)(uint32_t)(int32_t)y << 21) ^ (uint64_t)(uint32_t)(int32_t)z;
    };
    auto cellof = [&](int v, int64_t c[3]) {
        const float* p = &verts[3*(size_t)v];
        c[0] = (int64_t)std::floor(p[0] / step);
        c[1] = (int64_t)std::floor(p[1] / step);
        c[2] = (int64_t)std::floor(p[2] / step);
    };

    // 1. buckets: (key, vertex) sorted by key, then by vertex (stable sort)
    std::vector<uint64_t> skey((size_t)V);
    std::vector<uint32_t> sidx((size_t)V);
    parallel_for(V, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) {
            int64_t c[3]; cellof((int)v, c);
            skey[(size_t)v] = ckey(c[0], c[1], c[2]);
            sidx[(size_t)v] = (uint32_t)v;
        }
    });
    radix_sort_kv(skey, sidx);
    std::vector<int64_t> run_start;   // bucket r = sidx[run_start[r] .. run_start[r+1])
    {
        const std::vector<int64_t> cb = fixed_chunks(V, 1 << 16);
        const int64_t nc = (int64_t)cb.size() - 1;
        std::vector<int64_t> cnt((size_t)nc, 0);
        auto is_start = [&](int64_t i) { return i == 0 || skey[(size_t)i] != skey[(size_t)i - 1]; };
        parallel_tasks(nc, [&](int64_t k) {
            int64_t n = 0;
            for (int64_t i = cb[(size_t)k]; i < cb[(size_t)k + 1]; ++i) n += is_start(i);
            cnt[(size_t)k] = n;
        });
        const int64_t nr = exclusive_scan(cnt);
        run_start.resize((size_t)nr + 1);
        run_start[(size_t)nr] = V;
        parallel_tasks(nc, [&](int64_t k) {
            int64_t w = cnt[(size_t)k];
            for (int64_t i = cb[(size_t)k]; i < cb[(size_t)k + 1]; ++i) if (is_start(i)) run_start[(size_t)w++] = i;
        });
    }
    const int64_t NR = (int64_t)run_start.size() - 1;
    FlatU64Map bucket;
    {
        std::vector<uint64_t> rk((size_t)NR);
        std::vector<int32_t> rv((size_t)NR);
        parallel_for(NR, 1 << 14, [&](int64_t b, int64_t e) {
            for (int64_t r = b; r < e; ++r) { rk[(size_t)r] = skey[(size_t)run_start[(size_t)r]]; rv[(size_t)r] = (int32_t)r; }
        });
        bucket.build(rk.data(), rv.data(), NR);
    }
    std::vector<uint64_t>().swap(skey);

    auto close = [&](int c, const float* p) {
        const float ex = verts[3*(size_t)c]-p[0], ey = verts[3*(size_t)c+1]-p[1], ez = verts[3*(size_t)c+2]-p[2];
        return ex*ex + ey*ey + ez*ez <= eps2;
    };

    // 2. conflicted = some earlier vertex (kept or not) lies within eps in the 27 buckets
    std::vector<uint8_t> conflicted((size_t)V, 0);
    parallel_for(V, 1 << 12, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) {
            const float* p = &verts[3*(size_t)v];
            int64_t c[3]; cellof((int)v, c);
            bool hit = false;
            for (int dz = -1; dz <= 1 && !hit; ++dz)
            for (int dy = -1; dy <= 1 && !hit; ++dy)
            for (int dx = -1; dx <= 1 && !hit; ++dx) {
                const int32_t r = bucket.find(ckey(c[0]+dx, c[1]+dy, c[2]+dz));
                if (r < 0) continue;
                for (int64_t i = run_start[(size_t)r]; i < run_start[(size_t)r + 1]; ++i) {
                    const int u = (int)sidx[(size_t)i];
                    if (u >= v) break;   // bucket is in index order
                    if (close(u, p)) { hit = true; break; }
                }
            }
            conflicted[(size_t)v] = hit;
        }
    });

    // 3. serial resolution of the conflicted vertices, legacy scan order
    std::vector<int32_t> rep((size_t)V);
    std::vector<uint8_t> kept((size_t)V);
    parallel_for(V, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) { rep[(size_t)v] = (int32_t)v; kept[(size_t)v] = !conflicted[(size_t)v]; }
    });
    int64_t nconf = 0;
    for (int v = 0; v < V; ++v) {
        if (!conflicted[(size_t)v]) continue;
        ++nconf;
        const float* p = &verts[3*(size_t)v];
        int64_t c[3]; cellof(v, c);
        int found = -1;
        for (int dz = -1; dz <= 1 && found < 0; ++dz)
        for (int dy = -1; dy <= 1 && found < 0; ++dy)
        for (int dx = -1; dx <= 1 && found < 0; ++dx) {
            const int32_t r = bucket.find(ckey(c[0]+dx, c[1]+dy, c[2]+dz));
            if (r < 0) continue;
            for (int64_t i = run_start[(size_t)r]; i < run_start[(size_t)r + 1]; ++i) {
                const int u = (int)sidx[(size_t)i];
                if (u >= v) break;
                if (kept[(size_t)u] && close(u, p)) { found = u; break; }
            }
        }
        if (found < 0) kept[(size_t)v] = 1;
        else rep[(size_t)v] = found;
    }
    std::vector<uint8_t>().swap(conflicted);
    std::vector<uint32_t>().swap(sidx);
    std::vector<int64_t>().swap(run_start);

    // 4. new ids = rank among kept vertices; compact and remap
    std::vector<int32_t> newid((size_t)V);
    const std::vector<int64_t> cb = fixed_chunks(V, 1 << 16);
    const int64_t nc = (int64_t)cb.size() - 1;
    std::vector<int64_t> cnt((size_t)nc, 0);
    parallel_tasks(nc, [&](int64_t k) {
        int64_t n = 0;
        for (int64_t v = cb[(size_t)k]; v < cb[(size_t)k + 1]; ++v) n += kept[(size_t)v];
        cnt[(size_t)k] = n;
    });
    const int keep = (int)exclusive_scan(cnt);
    std::vector<float> nv((size_t)keep * 3);
    std::vector<float> ncol; if (colors3) ncol.resize((size_t)keep * 3);
    parallel_tasks(nc, [&](int64_t k) {
        int64_t w = cnt[(size_t)k];
        for (int64_t v = cb[(size_t)k]; v < cb[(size_t)k + 1]; ++v) {
            if (!kept[(size_t)v]) continue;
            newid[(size_t)v] = (int32_t)w;
            for (int a = 0; a < 3; ++a) nv[3*(size_t)w + a] = verts[3*(size_t)v + a];
            if (colors3) for (int a = 0; a < 3; ++a) ncol[3*(size_t)w + a] = (*colors3)[3*(size_t)v + a];
            ++w;
        }
    });
    parallel_for((int64_t)faces.size(), 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) faces[(size_t)i] = newid[(size_t)rep[(size_t)faces[(size_t)i]]];
    });
    const int welded = V - keep;
    verts.swap(nv);
    if (colors3) colors3->swap(ncol);
    if (welded) { printf("  weld: V %d->%d (%d near-duplicates merged, eps=%.2g)\n", V, keep, welded, step); fflush(stdout); }
    (void)nconf;
    return welded;
}

// ---------------------------------------------------------------------------
// clean_mesh
//
// The legacy DFS needs, per face edge, "the other face if the edge has exactly
// two incident faces". That table (mate) comes from the edge CSR in parallel;
// the DFS itself then runs on flat arrays, unchanged, so the result is the
// same even for non-orientable regions where traversal order matters.
void clean_mesh(int V, std::vector<int32_t>& faces) {
    // 1. drop degenerate faces (order preserving)
    {
        const int64_t F0 = (int64_t)(faces.size() / 3);
        const std::vector<int64_t> cb = fixed_chunks(F0, 1 << 16);
        const int64_t nc = (int64_t)cb.size() - 1;
        std::vector<int64_t> cnt((size_t)nc, 0);
        auto good = [&](int64_t f) {
            const int a = faces[3*f], b = faces[3*f+1], c = faces[3*f+2];
            return a != b && b != c && a != c;
        };
        parallel_tasks(nc, [&](int64_t k) {
            int64_t n = 0;
            for (int64_t f = cb[(size_t)k]; f < cb[(size_t)k + 1]; ++f) n += good(f);
            cnt[(size_t)k] = n;
        });
        const int64_t nk = exclusive_scan(cnt);
        if (nk != F0) {
            std::vector<int32_t> kept((size_t)nk * 3);
            parallel_tasks(nc, [&](int64_t k) {
                int64_t w = cnt[(size_t)k];
                for (int64_t f = cb[(size_t)k]; f < cb[(size_t)k + 1]; ++f)
                    if (good(f)) { for (int j = 0; j < 3; ++j) kept[3*(size_t)w + j] = faces[3*f + j]; ++w; }
            });
            faces.swap(kept);
        }
        faces.resize((size_t)nk * 3);
    }
    const int F = (int)(faces.size() / 3);
    if (F == 0) return;
    const int64_t Vb = std::max<int64_t>(V, vertex_bound(faces.data(), faces.size()));

    // 2. mate[3f+j] = (g << 1) | same_direction for a 2-face edge, else -1
    std::vector<int32_t> mate((size_t)F * 3, -1);
    {
        const EdgeCsr csr = build_edge_csr(faces.data(), F, Vb);
        parallel_for(Vb, 1 << 12, [&](int64_t b, int64_t e) {
            for (int64_t a = b; a < e; ++a) {
                const int64_t lo = csr.off[(size_t)a], hi = csr.off[(size_t)a + 1];
                for (int64_t i = lo; i < hi;) {
                    int64_t j = i + 1;
                    const int32_t o = EdgeCsr::other(csr.ent[(size_t)i]);
                    while (j < hi && EdgeCsr::other(csr.ent[(size_t)j]) == o) ++j;
                    if (j - i == 2) {
                        const int32_t c1 = EdgeCsr::corner(csr.ent[(size_t)i]), c2 = EdgeCsr::corner(csr.ent[(size_t)i + 1]);
                        const int32_t same = faces[(size_t)c1] == faces[(size_t)c2];   // both edges start at the same vertex
                        mate[(size_t)c1] = ((c2 / 3) << 1) | same;
                        mate[(size_t)c2] = ((c1 / 3) << 1) | same;
                    }
                    i = j;
                }
            }
        });
    }

    // 3. legacy DFS flood over flat arrays. A flipped face walks its corrected
    // winding (v0, v2, v1), i.e. the original edges in order 2, 1, 0.
    std::vector<char> vis((size_t)F, 0), flip((size_t)F, 0);
    std::vector<int> st;
    for (int seed = 0; seed < F; ++seed) {
        if (vis[(size_t)seed]) continue;
        vis[(size_t)seed] = 1; st.push_back(seed);
        while (!st.empty()) {
            const int f = st.back(); st.pop_back();
            const char ff = flip[(size_t)f];
            for (int jj = 0; jj < 3; ++jj) {
                const int j = ff ? 2 - jj : jj;
                const int32_t m = mate[3*(size_t)f + j];
                if (m < 0) continue;
                const int g = m >> 1;
                if (vis[(size_t)g]) continue;
                // g traverses this edge the same way as f's corrected winding?
                if ((m & 1) ^ ff) flip[(size_t)g] = 1;
                vis[(size_t)g] = 1; st.push_back(g);
            }
        }
    }
    std::vector<int32_t>().swap(mate);
    std::atomic<int64_t> nflip{0};
    parallel_for(F, 1 << 16, [&](int64_t b, int64_t e) {
        int64_t n = 0;
        for (int64_t f = b; f < e; ++f) if (flip[(size_t)f]) { std::swap(faces[3*f+1], faces[3*f+2]); ++n; }
        nflip.fetch_add(n, std::memory_order_relaxed);
    });
    printf("    [clean] faces=%d, flipped %d for consistent winding\n", F, (int)nflip.load());
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// drop_small_components
//
// Same union-find (serial, order dependent only through the root labels, which
// are internal); the face-count map becomes a flat array; compaction and the
// first-appearance vertex renumbering are parallel.
int drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac) {
    const int V = (int)verts.size() / 3;
    const int64_t F = (int64_t)(faces.size() / 3);
    if (V == 0 || F == 0) return 0;
    std::vector<int> par((size_t)V);
    for (int i = 0; i < V; ++i) par[(size_t)i] = i;
    auto find = [&](int x) { while (par[(size_t)x] != x) { par[(size_t)x] = par[(size_t)par[(size_t)x]]; x = par[(size_t)x]; } return x; };
    auto uni  = [&](int a, int b) { int ra = find(a), rb = find(b); if (ra != rb) par[(size_t)ra] = rb; };
    for (int64_t f = 0; f < F; ++f) { uni(faces[3*f], faces[3*f+1]); uni(faces[3*f+1], faces[3*f+2]); }
    // full compression: root[v] (read-only afterwards)
    std::vector<int32_t> root((size_t)V);
    for (int v = 0; v < V; ++v) root[(size_t)v] = find(v);
    std::vector<int32_t>().swap(par);
    std::unique_ptr<std::atomic<int32_t>[]> fc(new std::atomic<int32_t>[(size_t)V]);
    parallel_for(V, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) fc[(size_t)v].store(0, std::memory_order_relaxed);
    });
    parallel_for(F, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f) fc[(size_t)root[(size_t)faces[3*f]]].fetch_add(1, std::memory_order_relaxed);
    });
    int maxfc = 0;
    for (int v = 0; v < V; ++v) maxfc = std::max(maxfc, (int)fc[(size_t)v].load(std::memory_order_relaxed));
    const int thresh = (int)(frac * maxfc);
    int dropped = 0;
    for (int v = 0; v < V; ++v) {
        const int n = fc[(size_t)v].load(std::memory_order_relaxed);
        if (n > 0 && n < thresh) ++dropped;
    }
    if (dropped == 0) return 0;
    auto keepf = [&](int64_t f) { return fc[(size_t)root[(size_t)faces[3*f]]].load(std::memory_order_relaxed) >= thresh; };

    const std::vector<int64_t> cb = fixed_chunks(F, 1 << 16);
    const int64_t nc = (int64_t)cb.size() - 1;
    std::vector<int64_t> cnt((size_t)nc, 0);
    parallel_tasks(nc, [&](int64_t k) {
        int64_t n = 0;
        for (int64_t f = cb[(size_t)k]; f < cb[(size_t)k + 1]; ++f) n += keepf(f);
        cnt[(size_t)k] = n;
    });
    const int64_t nkf = exclusive_scan(cnt);
    std::vector<int32_t> kf((size_t)nkf * 3);
    parallel_tasks(nc, [&](int64_t k) {
        int64_t w = cnt[(size_t)k];
        for (int64_t f = cb[(size_t)k]; f < cb[(size_t)k + 1]; ++f)
            if (keepf(f)) { for (int j = 0; j < 3; ++j) kf[3*(size_t)w + j] = faces[3*f + j]; ++w; }
    });
    fc.reset();
    std::vector<int32_t>().swap(root);

    // Legacy renumbering: vertices in order of first appearance in kf.
    const int64_t NI = nkf * 3;
    std::unique_ptr<std::atomic<int64_t>[]> first(new std::atomic<int64_t>[(size_t)V]);
    parallel_for(V, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t v = b; v < e; ++v) first[(size_t)v].store(INT64_MAX, std::memory_order_relaxed);
    });
    parallel_for(NI, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            std::atomic<int64_t>& slot = first[(size_t)kf[(size_t)i]];
            int64_t cur = slot.load(std::memory_order_relaxed);
            while (i < cur && !slot.compare_exchange_weak(cur, i, std::memory_order_relaxed)) {}
        }
    });
    const std::vector<int64_t> ib = fixed_chunks(NI, 1 << 16);
    const int64_t ni = (int64_t)ib.size() - 1;
    std::vector<int64_t> icnt((size_t)ni, 0);
    auto is_first = [&](int64_t i) { return first[(size_t)kf[(size_t)i]].load(std::memory_order_relaxed) == i; };
    parallel_tasks(ni, [&](int64_t k) {
        int64_t n = 0;
        for (int64_t i = ib[(size_t)k]; i < ib[(size_t)k + 1]; ++i) n += is_first(i);
        icnt[(size_t)k] = n;
    });
    const int64_t nvk = exclusive_scan(icnt);
    std::vector<int32_t> remap((size_t)V, -1);
    std::vector<float> nv((size_t)nvk * 3);
    parallel_tasks(ni, [&](int64_t k) {
        int64_t w = icnt[(size_t)k];
        for (int64_t i = ib[(size_t)k]; i < ib[(size_t)k + 1]; ++i)
            if (is_first(i)) {
                const int v = kf[(size_t)i];
                remap[(size_t)v] = (int32_t)w;
                for (int a = 0; a < 3; ++a) nv[3*(size_t)w + a] = verts[3*(size_t)v + a];
                ++w;
            }
    });
    first.reset();
    parallel_for(NI, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) kf[(size_t)i] = remap[(size_t)kf[(size_t)i]];
    });
    verts.swap(nv); faces.swap(kf);
    return dropped;
}

}  // namespace post
}  // namespace trellis
