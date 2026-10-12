#include "remesh_dc.h"
#include "tri_bvh.h"
#include "post_parallel.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace trellis {

namespace {

// TRELLIS_POST_TIMING=1: print wall time of each remesh sub-step.
struct StepTimer {
    bool on = std::getenv("TRELLIS_POST_TIMING") != nullptr;
    std::chrono::steady_clock::time_point t = std::chrono::steady_clock::now();
    void lap(const char* what) {
        if (!on) return;
        const auto n = std::chrono::steady_clock::now();
        printf("    [remesh-step] %-22s %.2fs\n", what, std::chrono::duration<double>(n - t).count());
        fflush(stdout);
        t = n;
    }
};

inline int ctz64(uint64_t v) {
#ifdef _MSC_VER
    unsigned long i;
    _BitScanForward64(&i, v);
    return (int)i;
#else
    return __builtin_ctzll(v);
#endif
}

inline uint64_t key3(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 42) | ((uint64_t)(uint32_t)y << 21) | (uint32_t)z;
}

// Arctic: dynamically scheduled chunks (TRELLIS_POST_THREADS workers); every caller
// writes results by index, so the output is independent of the schedule.
void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn, int64_t grain = 256) {
    post::parallel_for(n, grain, fn);
}

inline int popcount64(uint64_t v) {
    v = v - ((v >> 1) & 0x5555555555555555ull);
    v = (v & 0x3333333333333333ull) + ((v >> 2) & 0x3333333333333333ull);
    v = (v + (v >> 4)) & 0x0f0f0f0f0f0f0f0full;
    return (int)((v * 0x0101010101010101ull) >> 56);
}

// Arctic: occupancy bitset over a dim^3 grid with a per-word prefix popcount, so "does cell
// (x,y,z) exist, and what is its rank among the set cells in linear order" costs one bit test
// and one popcount, with no hashing. Voxel ids are exactly that rank (acoord is in linear order).
struct DenseSet {
    int64_t dim = 0, nwords = 0;
    std::unique_ptr<std::atomic<uint64_t>[]> bits;
    std::vector<uint32_t> pre;   // pre[w] = set bits in words [0, w)
    void init(int64_t d) {
        dim = d;
        nwords = (d * d * d + 63) / 64;
        bits.reset(new std::atomic<uint64_t>[(size_t)nwords]);
        post::parallel_for(nwords, 1 << 16, [&](int64_t b, int64_t e) {
            for (int64_t w = b; w < e; ++w) bits[(size_t)w].store(0, std::memory_order_relaxed);
        });
    }
    int64_t index(int x, int y, int z) const { return ((int64_t)x * dim + y) * dim + z; }
    void set(int64_t i) { bits[(size_t)(i >> 6)].fetch_or(1ull << (i & 63), std::memory_order_relaxed); }
    int64_t finalize();   // builds pre[]; returns the number of set cells
    int find(int x, int y, int z) const {   // rank, or -1 if absent / outside the grid
        if ((unsigned)x >= (unsigned)dim || (unsigned)y >= (unsigned)dim || (unsigned)z >= (unsigned)dim) return -1;
        const int64_t i = index(x, y, z);
        const uint64_t w = bits[(size_t)(i >> 6)].load(std::memory_order_relaxed);
        if (!((w >> (i & 63)) & 1)) return -1;
        return (int)(pre[(size_t)(i >> 6)] + popcount64(w & ((1ull << (i & 63)) - 1)));
    }
};

int64_t DenseSet::finalize() {
    const std::vector<int64_t> cb = post::fixed_chunks(nwords, 1 << 16);
    const int64_t nc = (int64_t)cb.size() - 1;
    std::vector<int64_t> off((size_t)nc + 1, 0);
    post::parallel_tasks(nc, [&](int64_t k) {
        int64_t n = 0;
        for (int64_t w = cb[(size_t)k]; w < cb[(size_t)k + 1]; ++w) n += popcount64(bits[(size_t)w].load(std::memory_order_relaxed));
        off[(size_t)k + 1] = n;
    });
    for (int64_t k = 0; k < nc; ++k) off[(size_t)k + 1] += off[(size_t)k];
    if (off[(size_t)nc] >= (int64_t)0xffffffffll) throw std::runtime_error("remesh: too many cells for the rank table");
    pre.assign((size_t)nwords + 1, 0);
    post::parallel_tasks(nc, [&](int64_t k) {
        int64_t run = off[(size_t)k];
        for (int64_t w = cb[(size_t)k]; w < cb[(size_t)k + 1]; ++w) {
            pre[(size_t)w] = (uint32_t)run;
            run += popcount64(bits[(size_t)w].load(std::memory_order_relaxed));
        }
    });
    pre[(size_t)nwords] = (uint32_t)off[(size_t)nc];
    return off[(size_t)nc];
}


}  // namespace

Mesh remesh_narrow_band_dc(const float* iverts, int64_t iV, const int32_t* ifaces, int64_t iF,
                           const TriBvh& bvh, int res, int band, float project_back) {
    (void)iV;
    Mesh out;
    if (iF == 0 || bvh.empty() || res <= 0) return out;

    // Reference domain: the world cube is inflated by (res+3·band)/res so the
    // offset shell never touches the boundary; eps is the offset distance.
    const float scale = (float)(res + 3 * band) / (float)res;
    const float cell = scale / (float)res;
    const float eps = (float)band * cell;
    const float keep = 0.87f * cell;

    // Candidate cells: conservative dilation of every triangle's AABB by the
    // band-plus-crossing radius, marked in a res^3 bitset.
    StepTimer timer;
    const int64_t nbits = (int64_t)res * res * res;
    std::vector<uint64_t> cand((size_t)((nbits + 63) / 64), 0);
    auto bit_set = [&cand, res](int x, int y, int z) {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        cand[(size_t)(i >> 6)] |= 1ull << (i & 63);
    };
    auto bit_get = [&cand, res](int x, int y, int z) -> bool {
        const int64_t i = ((int64_t)x * res + y) * res + z;
        return (cand[(size_t)(i >> 6)] >> (i & 63)) & 1;
    };
    // A cell is active iff UDF(center) < (band+0.87)·cell, and its closest
    // surface point lies inside some triangle-marked cell, so the per-axis
    // index distance is < band+1.37, i.e. ≤ band+1.
    const int dil = band + 1;
    {
        const int F = (int)iF;
        std::vector<std::vector<uint64_t>> parts;
        const int hw = post::threads();
        // Each worker previously allocated a full res^3 candidate bitset.  At
        // res=1024 that is 128 MiB per worker, so a 32-thread CPU consumed ~4 GiB
        // here before any geometry/BVH memory.  Cap only the replication memory;
        // the OR result and therefore the remesh are bit-for-bit equivalent.
        const size_t bytes_per_part = cand.size() * sizeof(uint64_t);
        // TRELLIS_REMESH_MEM_GIB raises (or lowers) that budget: 4 gives 24 workers at res 1024 for 3 GiB of RAM. Default 1 GiB.
        double budget_gib = 1.0;
        if (const char* e = std::getenv("TRELLIS_REMESH_MEM_GIB")) {
            const double v = std::atof(e);
            if (v >= 0.001 && v <= 64.0) budget_gib = v;
        }
        const size_t parts_budget = (size_t)(budget_gib * 1024.0 * 1024.0 * 1024.0);
        const int mem_workers = bytes_per_part ? (int)std::max<size_t>(1, parts_budget / bytes_per_part) : hw;
        const int nt = std::max(1, std::min(hw, mem_workers));
        if (nt < hw) {
            printf("  [remesh-mem] candidate bitset %.1f MiB/worker, workers %d->%d\n",
                   bytes_per_part / (1024.0*1024.0), hw, nt);
            fflush(stdout);
        }
        parts.assign(nt, {});
        std::vector<std::thread> ts;
        const int chunk = (F + nt - 1) / nt;
        for (int t = 0; t < nt; ++t) {
            const int b = t * chunk, e = std::min(F, b + chunk);
            if (b >= e) break;
            parts[t].assign(cand.size(), 0);
            ts.emplace_back([&, t, b, e]() {
                auto& bits = parts[t];
                auto setb = [&bits, res](int x, int y, int z) {
                    const int64_t i = ((int64_t)x * res + y) * res + z;
                    bits[(size_t)(i >> 6)] |= 1ull << (i & 63);
                };
                for (int f = b; f < e; ++f) {
                    float bmin[3] = {1e30f, 1e30f, 1e30f}, bmax[3] = {-1e30f, -1e30f, -1e30f};
                    for (int j = 0; j < 3; ++j) {
                        const float* p = &iverts[3 * ifaces[3*f+j]];
                        for (int k = 0; k < 3; ++k) {
                            bmin[k] = std::min(bmin[k], p[k]);
                            bmax[k] = std::max(bmax[k], p[k]);
                        }
                    }
                    int c0[3], c1[3];
                    for (int k = 0; k < 3; ++k) {
                        c0[k] = std::max(0, (int)std::floor((bmin[k] / scale + 0.5f) * res) - dil);
                        c1[k] = std::min(res - 1, (int)std::floor((bmax[k] / scale + 0.5f) * res) + dil);
                    }
                    for (int x = c0[0]; x <= c1[0]; ++x)
                        for (int y = c0[1]; y <= c1[1]; ++y)
                            for (int z = c0[2]; z <= c1[2]; ++z) setb(x, y, z);
                }
            });
        }
        for (auto& th : ts) th.join();
        for (auto& bits : parts)
            if (!bits.empty())
                for (size_t i = 0; i < cand.size(); ++i) cand[i] |= bits[i];
    }

    timer.lap("candidate bitset");
    // Active voxels: |UDF(center) - eps| < 0.87*cell (spec 27 §4.1).
    std::vector<int> acoord;
    {
        // Candidate cells in linear order (parallel count + fill; same list as a serial scan).
        const std::vector<int64_t> wb = post::fixed_chunks((int64_t)cand.size(), 1 << 14);
        const int64_t nwc = (int64_t)wb.size() - 1;
        std::vector<int64_t> woff((size_t)nwc + 1, 0);
        post::parallel_tasks(nwc, [&](int64_t k) {
            int64_t n = 0;
            for (int64_t w = wb[(size_t)k]; w < wb[(size_t)k + 1]; ++w) n += popcount64(cand[(size_t)w]);
            woff[(size_t)k + 1] = n;
        });
        for (int64_t k = 0; k < nwc; ++k) woff[(size_t)k + 1] += woff[(size_t)k];
        std::vector<int64_t> cand_cells((size_t)woff[(size_t)nwc]);
        post::parallel_tasks(nwc, [&](int64_t k) {
            int64_t o = woff[(size_t)k];
            for (int64_t w = wb[(size_t)k]; w < wb[(size_t)k + 1]; ++w) {
                uint64_t bits = cand[(size_t)w];
                while (bits) {
                    const int b = ctz64(bits);
                    bits &= bits - 1;
                    cand_cells[(size_t)o++] = (w << 6) | b;
                }
            }
        });
        std::vector<uint8_t> act(cand_cells.size(), 0);
        parallel_for((int64_t)cand_cells.size(), [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) {
                const int64_t c = cand_cells[i];
                const int x = (int)(c / ((int64_t)res * res)), y = (int)((c / res) % res), z = (int)(c % res);
                const float p[3] = { ((x + 0.5f) / res - 0.5f) * scale,
                                     ((y + 0.5f) / res - 0.5f) * scale,
                                     ((z + 0.5f) / res - 0.5f) * scale };
                const TriBvh::Hit h = bvh.closest(p, eps + keep);
                if (h.face < 0) continue;
                const float f = std::sqrt(h.dist2) - eps;
                if (std::fabs(f) < keep) act[i] = 1;
            }
        });
        const std::vector<int64_t> ab = post::fixed_chunks((int64_t)act.size(), 1 << 18);
        const int64_t nac = (int64_t)ab.size() - 1;
        std::vector<int64_t> aoff((size_t)nac + 1, 0);
        post::parallel_tasks(nac, [&](int64_t k) {
            int64_t n = 0;
            for (int64_t i = ab[(size_t)k]; i < ab[(size_t)k + 1]; ++i) n += act[(size_t)i];
            aoff[(size_t)k + 1] = n;
        });
        for (int64_t k = 0; k < nac; ++k) aoff[(size_t)k + 1] += aoff[(size_t)k];
        acoord.resize((size_t)aoff[(size_t)nac] * 3);
        post::parallel_tasks(nac, [&](int64_t k) {
            int64_t o = aoff[(size_t)k];
            for (int64_t i = ab[(size_t)k]; i < ab[(size_t)k + 1]; ++i) {
                if (!act[(size_t)i]) continue;
                const int64_t c = cand_cells[(size_t)i];
                acoord[(size_t)(3*o)]   = (int)(c / ((int64_t)res * res));
                acoord[(size_t)(3*o+1)] = (int)((c / res) % res);
                acoord[(size_t)(3*o+2)] = (int)(c % res);
                ++o;
            }
        });
        cand.clear(); cand.shrink_to_fit();
    }
    timer.lap("active voxels (UDF)");
    const int64_t Na = (int64_t)acoord.size() / 3;
    if (Na < 100) {
        fprintf(stderr, "  remesh: only %lld active voxels; skipping remesh\n", (long long)Na);
        return out;
    }

    // Arctic: sorted-key flat tables replace the node-based hash maps. acoord is in
    // ascending (x, y, z) = ascending key3 order, which lets the vertex numbering of
    // the legacy first-occurrence scan be reproduced without any hash build.
    const bool fast = !post::legacy();
    std::unordered_map<uint64_t, int> vox;       // legacy only
    std::unordered_map<uint64_t, int> vmap;      // legacy only
    DenseSet dvox, dvmap;                        // fast path: occupancy + rank, no hashing
    auto vox_id = [&](int x, int y, int z) -> int {
        if (fast) return dvox.find(x, y, z);
        auto it = vox.find(key3(x, y, z));
        return it == vox.end() ? -1 : it->second;
    };
    auto vmap_id = [&](int x, int y, int z) -> int {
        if (fast) return dvmap.find(x, y, z);
        auto it = vmap.find(key3(x, y, z));
        return it == vmap.end() ? -1 : it->second;
    };

    // f = UDF - eps at the grid VERTICES (corner mapping v/res, spec 27 §4.3).
    std::vector<int> vcoord;
    if (!fast) {
        vox.reserve((size_t)Na * 2);
        for (int64_t i = 0; i < Na; ++i) vox.emplace(key3(acoord[3*i], acoord[3*i+1], acoord[3*i+2]), (int)i);
        vmap.reserve((size_t)Na * 3);
        for (int64_t i = 0; i < Na; ++i)
            for (int dx = 0; dx < 2; ++dx) for (int dy = 0; dy < 2; ++dy) for (int dz = 0; dz < 2; ++dz) {
                const int x = acoord[3*i] + dx, y = acoord[3*i+1] + dy, z = acoord[3*i+2] + dz;
                if (vmap.emplace(key3(x, y, z), (int)(vcoord.size() / 3)).second) {
                    vcoord.push_back(x); vcoord.push_back(y); vcoord.push_back(z);
                }
            }
    } else {
        // Voxel id = rank among active cells (acoord is in linear order). Grid vertices are numbered by
        // rank in linear vertex order instead of first visit; ids only index fvert, so the output is
        // unchanged.
        dvox.init(res);
        parallel_for(Na, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) dvox.set(dvox.index(acoord[3*i], acoord[3*i+1], acoord[3*i+2]));
        }, 1 << 12);
        dvox.finalize();
        dvmap.init((int64_t)res + 1);
        parallel_for(Na, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i)
                for (int d = 0; d < 8; ++d)
                    dvmap.set(dvmap.index(acoord[3*i] + (d >> 2), acoord[3*i+1] + ((d >> 1) & 1), acoord[3*i+2] + (d & 1)));
        }, 1 << 12);
        const int64_t nv_total = dvmap.finalize();
        vcoord.resize((size_t)nv_total * 3);
        const std::vector<int64_t> wb = post::fixed_chunks(dvmap.nwords, 1 << 14);
        const int64_t nwc = (int64_t)wb.size() - 1;
        const int64_t dim = dvmap.dim;
        post::parallel_tasks(nwc, [&](int64_t k) {
            int64_t id = dvmap.pre[(size_t)wb[(size_t)k]];
            for (int64_t w = wb[(size_t)k]; w < wb[(size_t)k + 1]; ++w) {
                uint64_t bits = dvmap.bits[(size_t)w].load(std::memory_order_relaxed);
                while (bits) {
                    const int64_t c = (w << 6) | ctz64(bits);
                    bits &= bits - 1;
                    vcoord[(size_t)(3*id)]   = (int)(c / (dim * dim));
                    vcoord[(size_t)(3*id+1)] = (int)((c / dim) % dim);
                    vcoord[(size_t)(3*id+2)] = (int)(c % dim);
                    ++id;
                }
            }
        });
    }
    timer.lap("voxel/vertex tables");
    const int64_t Nv = (int64_t)vcoord.size() / 3;
    std::vector<float> fvert((size_t)Nv);
    parallel_for(Nv, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const float p[3] = { ((float)vcoord[3*i]   / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+1] / res - 0.5f) * scale,
                                 ((float)vcoord[3*i+2] / res - 0.5f) * scale };
            // Crossing edges always have their far endpoint under eps+cell
            // (f changes at most one cell-length per edge), so this bound
            // never clips a value that feeds the crossing interpolation.
            const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
            fvert[i] = (h.face >= 0 ? std::sqrt(h.dist2) : eps + 2 * cell) - eps;
        }
    });
    timer.lap("UDF at grid vertices");
    auto fval = [&](int x, int y, int z) -> float {
        const int id = vmap_id(x, y, z);
        return id < 0 ? 1e9f : fvert[id];
    };

    if (std::getenv("TRELLIS_REMESH_TETS")) {
        float min_fraction = 0.0f;
        if (const char* setting = std::getenv("TRELLIS_TETS_MIN_EDGE_FRACTION")) {
            char* end = nullptr;
            min_fraction = std::strtof(setting, &end);
            if (end == setting || *end || !std::isfinite(min_fraction)
                || min_fraction < 0.0f || min_fraction >= 0.5f)
                throw std::invalid_argument("TRELLIS_TETS_MIN_EDGE_FRACTION must be in [0, 0.5)");
        }
        static const int PERM[6][3] = {
            {0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}
        };
        std::unordered_map<uint64_t, int32_t> intersections;
        intersections.reserve((size_t)Na * 2);
        size_t clamped_intersections = 0;
        auto cut = [&](int a, int b) -> int32_t {
            const uint32_t lo = (uint32_t)std::min(a, b), hi = (uint32_t)std::max(a, b);
            const uint64_t key = (uint64_t(lo) << 32) | hi;
            auto it = intersections.find(key);
            if (it != intersections.end()) return it->second;
            const float fa = fvert[a], fb = fvert[b];
            const float raw_t = fa / (fa - fb);
            const float t = std::clamp(raw_t, min_fraction, 1.0f - min_fraction);
            clamped_intersections += t != raw_t;
            const int32_t index = (int32_t)(out.verts.size() / 3);
            for (int k = 0; k < 3; ++k) {
                const float coord = vcoord[3*a+k] + t * (vcoord[3*b+k] - vcoord[3*a+k]);
                out.verts.push_back((coord / res - 0.5f) * scale);
            }
            intersections.emplace(key, index);
            return index;
        };
        auto emit = [&](int32_t a, int32_t b, int32_t c, const float* direction) {
            const float* pa = out.verts.data() + 3*a;
            const float* pb = out.verts.data() + 3*b;
            const float* pc = out.verts.data() + 3*c;
            const float u[3] = {pb[0]-pa[0], pb[1]-pa[1], pb[2]-pa[2]};
            const float v[3] = {pc[0]-pa[0], pc[1]-pa[1], pc[2]-pa[2]};
            const float dot = (u[1]*v[2]-u[2]*v[1])*direction[0]
                            + (u[2]*v[0]-u[0]*v[2])*direction[1]
                            + (u[0]*v[1]-u[1]*v[0])*direction[2];
            if (dot < 0) std::swap(b, c);
            out.faces.push_back(a); out.faces.push_back(b); out.faces.push_back(c);
        };
        for (int64_t i = 0; i < Na; ++i) {
            const int x = acoord[3*i], y = acoord[3*i+1], z = acoord[3*i+2];
            int corner[8];
            for (int mask = 0; mask < 8; ++mask)
                corner[mask] = vmap_id(x + (mask & 1), y + ((mask >> 1) & 1), z + ((mask >> 2) & 1));
            for (const auto& perm : PERM) {
                const int tet[4] = {corner[0], corner[1 << perm[0]],
                                    corner[(1 << perm[0]) | (1 << perm[1])], corner[7]};
                int inside[3], outside[3], ni = 0, no = 0;
                for (int t = 0; t < 4; ++t)
                    (fvert[tet[t]] < 0 ? inside[ni++] : outside[no++]) = tet[t];
                if (!ni || !no) continue;
                float direction[3] = {};
                for (int j = 0; j < no; ++j)
                    for (int k = 0; k < 3; ++k) direction[k] += float(vcoord[3*outside[j]+k]) / no;
                for (int j = 0; j < ni; ++j)
                    for (int k = 0; k < 3; ++k) direction[k] -= float(vcoord[3*inside[j]+k]) / ni;
                if (ni == 1) {
                    emit(cut(inside[0], outside[0]), cut(inside[0], outside[1]),
                         cut(inside[0], outside[2]), direction);
                } else if (no == 1) {
                    emit(cut(outside[0], inside[0]), cut(outside[0], inside[1]),
                         cut(outside[0], inside[2]), direction);
                } else {
                    const int32_t a = cut(inside[0], outside[0]);
                    const int32_t b = cut(inside[0], outside[1]);
                    const int32_t c = cut(inside[1], outside[1]);
                    const int32_t d = cut(inside[1], outside[0]);
                    emit(a, b, c, direction);
                    emit(a, c, d, direction);
                }
            }
        }
        printf("  remesh_tets: %lld active voxels -> V=%d F=%d (edge_fraction=%.4g, clamped=%zu)\n",
               (long long)Na, out.V(), out.F(), min_fraction, clamped_intersections);
        fflush(stdout);
        return out;
    }

    // Dual vertices: plain mean of edge crossings, cell-center fallback; per
    // voxel, ownership of the 3 "far" edges records crossing direction
    // (spec 27 §4.4).
    std::vector<float> dual((size_t)Na * 3);
    std::vector<int8_t> owned((size_t)Na * 3, 0);
    parallel_for(Na, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) {
            const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
            double sum[3] = {0, 0, 0};
            int cnt = 0;
            float fc[8];   // field at the voxel's 8 corners, index (dx<<2)|(dy<<1)|dz (every one exists)
            for (int d = 0; d < 8; ++d) fc[d] = fval(vx + (d >> 2), vy + ((d >> 1) & 1), vz + (d & 1));
            for (int axis = 0; axis < 3; ++axis)
                for (int u = 0; u < 2; ++u) for (int v = 0; v < 2; ++v) {
                    int a0[3] = {vx, vy, vz}, a1[3];
                    int o0[3] = {0, 0, 0};
                    o0[(axis + 1) % 3] += u;
                    o0[(axis + 2) % 3] += v;
                    a0[0] += o0[0]; a0[1] += o0[1]; a0[2] += o0[2];
                    a1[0] = a0[0]; a1[1] = a0[1]; a1[2] = a0[2];
                    a1[axis] += 1;
                    const int i1 = (o0[0] << 2) | (o0[1] << 1) | o0[2];
                    const float v1 = fc[i1];
                    const float v2 = fc[i1 | (4 >> axis)];
                    const bool c12 = v1 < 0 && v2 >= 0, c21 = v1 >= 0 && v2 < 0;
                    if (c12 || c21) {
                        const float t = -v1 / (v2 - v1);
                        double pt[3] = {(double)a0[0], (double)a0[1], (double)a0[2]};
                        pt[axis] += t;
                        for (int k = 0; k < 3; ++k) sum[k] += pt[k];
                        ++cnt;
                    }
                    if (u == 1 && v == 1) owned[3*i + axis] = c12 ? 1 : (c21 ? -1 : 0);
                }
            if (cnt) for (int k = 0; k < 3; ++k) dual[3*i+k] = (float)(sum[k] / cnt);
            else { dual[3*i] = vx + 0.5f; dual[3*i+1] = vy + 0.5f; dual[3*i+2] = vz + 0.5f; }
        }
    });

    timer.lap("dual vertices");
    // Quad assembly per owned crossing edge (spec 27 §4.5); winding from the
    // crossing direction. The reference's "planar diagonal" selection is a
    // latent no-op upstream (always diagonal q0-q2 unless triangle q0q1q2 is
    // exactly degenerate) — reproduced faithfully here as split 1 always.
    static const int OFF[3][4][3] = {
        {{0,0,0}, {0,0,1}, {0,1,1}, {0,1,0}},
        {{0,0,0}, {1,0,0}, {1,0,1}, {0,0,1}},
        {{0,0,0}, {0,1,0}, {1,1,0}, {1,0,0}},
    };
    static const int S1N[6] = {0, 1, 2, 0, 2, 3};
    static const int S1P[6] = {0, 2, 1, 0, 3, 2};
    if (!fast) {
    std::vector<int32_t> qfaces;
    qfaces.reserve((size_t)Na * 6);
    std::vector<uint8_t> used((size_t)Na, 0);
    for (int64_t i = 0; i < Na; ++i) {
        const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
        for (int axis = 0; axis < 3; ++axis) {
            const int dir = owned[3*i + axis];
            if (!dir) continue;
            int q[4];
            bool ok = true;
            for (int k = 0; k < 4 && ok; ++k) {
                const int id = vox_id(vx + OFF[axis][k][0], vy + OFF[axis][k][1], vz + OFF[axis][k][2]);
                if (id < 0) ok = false;
                else q[k] = id;
            }
            if (!ok) continue;
            const int* sp = dir > 0 ? S1P : S1N;
            for (int k = 0; k < 6; ++k) qfaces.push_back(q[sp[k]]);
            for (int k = 0; k < 4; ++k) used[q[k]] = 1;
        }
    }
    if (qfaces.empty()) return out;

    // Compact used dual vertices; map back to world coordinates.
    std::vector<int32_t> remap((size_t)Na, -1);
    int nv2 = 0;
    for (int64_t i = 0; i < Na; ++i)
        if (used[i]) {
            remap[i] = nv2++;
            out.verts.push_back((dual[3*i]   / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+1] / res - 0.5f) * scale);
            out.verts.push_back((dual[3*i+2] / res - 0.5f) * scale);
        }
    out.faces.resize(qfaces.size());
    for (size_t k = 0; k < qfaces.size(); ++k) out.faces[k] = remap[qfaces[k]];
    } else {
        // Arctic: per-chunk quad lists (fixed chunking) concatenated in chunk order = legacy face order.
        const std::vector<int64_t> qb = post::fixed_chunks(Na, 1 << 14);
        const int64_t nqc = (int64_t)qb.size() - 1;
        std::vector<std::vector<int32_t>> qparts((size_t)nqc);
        std::unique_ptr<std::atomic<uint8_t>[]> used(new std::atomic<uint8_t>[(size_t)Na]);
        parallel_for(Na, [&](int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) used[(size_t)i].store(0, std::memory_order_relaxed);
        }, 1 << 16);
        post::parallel_tasks(nqc, [&](int64_t k) {
            std::vector<int32_t>& qf = qparts[(size_t)k];
            for (int64_t i = qb[(size_t)k]; i < qb[(size_t)k + 1]; ++i) {
                const int vx = acoord[3*i], vy = acoord[3*i+1], vz = acoord[3*i+2];
                for (int axis = 0; axis < 3; ++axis) {
                    const int dir = owned[3*i + axis];
                    if (!dir) continue;
                    int q[4];
                    bool ok = true;
                    for (int c = 0; c < 4 && ok; ++c) {
                        const int id = dvox.find(vx + OFF[axis][c][0], vy + OFF[axis][c][1], vz + OFF[axis][c][2]);
                        if (id < 0) ok = false;
                        else q[c] = id;
                    }
                    if (!ok) continue;
                    const int* sp = dir > 0 ? S1P : S1N;
                    for (int c = 0; c < 6; ++c) qf.push_back(q[sp[c]]);
                    for (int c = 0; c < 4; ++c) used[(size_t)q[c]].store(1, std::memory_order_relaxed);
                }
            }
        });
        std::vector<int64_t> qoff((size_t)nqc + 1, 0);
        for (int64_t k = 0; k < nqc; ++k) qoff[(size_t)k + 1] = qoff[(size_t)k] + (int64_t)qparts[(size_t)k].size();
        if (qoff[(size_t)nqc] == 0) return out;

        const std::vector<int64_t> ub = post::fixed_chunks(Na, 1 << 16);
        const int64_t nuc = (int64_t)ub.size() - 1;
        std::vector<int64_t> uoff((size_t)nuc + 1, 0);
        post::parallel_tasks(nuc, [&](int64_t k) {
            int64_t n = 0;
            for (int64_t i = ub[(size_t)k]; i < ub[(size_t)k + 1]; ++i) n += used[(size_t)i].load(std::memory_order_relaxed);
            uoff[(size_t)k + 1] = n;
        });
        for (int64_t k = 0; k < nuc; ++k) uoff[(size_t)k + 1] += uoff[(size_t)k];
        std::vector<int32_t> remap((size_t)Na, -1);
        out.verts.resize((size_t)uoff[(size_t)nuc] * 3);
        post::parallel_tasks(nuc, [&](int64_t k) {
            int64_t w = uoff[(size_t)k];
            for (int64_t i = ub[(size_t)k]; i < ub[(size_t)k + 1]; ++i)
                if (used[(size_t)i].load(std::memory_order_relaxed)) {
                    remap[(size_t)i] = (int32_t)w;
                    out.verts[(size_t)(3*w)]   = (dual[3*i]   / res - 0.5f) * scale;
                    out.verts[(size_t)(3*w+1)] = (dual[3*i+1] / res - 0.5f) * scale;
                    out.verts[(size_t)(3*w+2)] = (dual[3*i+2] / res - 0.5f) * scale;
                    ++w;
                }
        });
        out.faces.resize((size_t)qoff[(size_t)nqc]);
        post::parallel_tasks(nqc, [&](int64_t k) {
            const std::vector<int32_t>& qf = qparts[(size_t)k];
            int32_t* dst = out.faces.data() + qoff[(size_t)k];
            for (size_t c = 0; c < qf.size(); ++c) dst[c] = remap[(size_t)qf[c]];
            std::vector<int32_t>().swap(qparts[(size_t)k]);
        });
    }

    timer.lap("quads + compaction");
    if (std::getenv("TRELLIS_DIAG_DC_AMBIGUITY")) {
        int64_t ambiguous = 0;
        for (int64_t i = 0; i < Na; ++i) {
            const int base[3] = {acoord[3*i], acoord[3*i+1], acoord[3*i+2]};
            for (int axis = 0; axis < 3; ++axis) {
                int corner[3] = {base[0], base[1], base[2]};
                ++corner[axis];
                if (vox_id(corner[0], corner[1], corner[2]) < 0) continue;
                const int u = (axis + 1) % 3, v = (axis + 2) % 3;
                auto sign = [&](int du, int dv) {
                    int p[3] = {corner[0], corner[1], corner[2]};
                    p[u] += du; p[v] += dv;
                    return fval(p[0], p[1], p[2]) < 0;
                };
                const bool s00 = sign(0, 0), s10 = sign(1, 0);
                const bool s01 = sign(0, 1), s11 = sign(1, 1);
                ambiguous += s00 == s11 && s10 == s01 && s00 != s10;
            }
        }
        printf("  remesh_dc: ambiguous shared voxel faces=%lld\n", (long long)ambiguous);
    }

    // Project the dual vertices back onto the input surface (reference:
    // remesh_project=0.9, o_voxel/postprocess.py::to_glb -> remeshing.py §8).
    // Dual contouring places each vertex at the plain MEAN of its cell's edge
    // crossings, on the eps-offset shell — so sharp features come out rounded and
    // the shell keeps its own offset noise. Lerping each vertex back toward its
    // closest point on the original surface restores those features. The BVH's
    // closest point is exactly the reference's barycentric interpolation of the
    // hit triangle, so no uvw round-trip is needed.
    if (project_back > 0.0f) {
        const int64_t NV = (int64_t)out.verts.size() / 3;
        std::atomic<int64_t> missed{0};
        parallel_for(NV, [&](int64_t b, int64_t e) {
            int64_t local = 0;
            for (int64_t i = b; i < e; ++i) {
                float* v = &out.verts[3 * i];
                const float p[3] = {v[0], v[1], v[2]};
                // Vertices sit on the eps shell, so the surface is ~eps away; the
                // same bound the field pass uses is comfortably sufficient.
                const TriBvh::Hit h = bvh.closest(p, eps + 2 * cell);
                if (h.face < 0) { ++local; continue; }   // no hit in range: leave as-is
                for (int k = 0; k < 3; ++k) v[k] = p[k] - project_back * (p[k] - h.point[k]);
            }
            if (local) missed.fetch_add(local, std::memory_order_relaxed);
        });
        if (missed.load())
            printf("  remesh_dc: project_back %.2f (%lld/%lld vertices had no hit in range)\n",
                   project_back, (long long)missed.load(), (long long)NV);
    }

    timer.lap("project back");
    printf("  remesh_dc: %lld active voxels -> V=%d F=%d (eps=%.4g, project_back=%.2f)\n",
           (long long)Na, out.V(), out.F(), eps, project_back);
    fflush(stdout);
    return out;
}

}  // namespace trellis
