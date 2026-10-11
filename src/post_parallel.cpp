// Arctic: shared helpers for the multithreaded CPU post-processing stages.
// See include/post_parallel.h and docs/arctic/post-mt.md.
#include "post_parallel.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>

namespace trellis {
namespace post {

int threads() {
    if (const char* s = std::getenv("TRELLIS_POST_THREADS")) {
        const int n = std::atoi(s);
        if (n > 0) return std::min(n, 1024);
    }
    return (int)std::max(1u, std::thread::hardware_concurrency());
}

bool legacy() {
    const char* s = std::getenv("TRELLIS_POST_LEGACY");
    return s && *s && std::strcmp(s, "0") != 0;
}

namespace {

// Runs body(worker) on nt workers (worker 0 on the calling thread) and
// rethrows the first exception after all have joined.
void run_workers(int nt, const std::function<void(int)>& body) {
    if (nt <= 1) { body(0); return; }
    std::exception_ptr err;
    std::mutex mu;
    auto guarded = [&](int w) {
        try { body(w); }
        catch (...) { std::lock_guard<std::mutex> lk(mu); if (!err) err = std::current_exception(); }
    };
    std::vector<std::thread> ts;
    ts.reserve((size_t)nt - 1);
    for (int w = 1; w < nt; ++w) ts.emplace_back(guarded, w);
    guarded(0);
    for (auto& t : ts) t.join();
    if (err) std::rethrow_exception(err);
}

}  // namespace

void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    grain = std::max<int64_t>(1, grain);
    const int64_t max_chunks = (n + grain - 1) / grain;
    const int nt = (int)std::min<int64_t>(threads(), max_chunks);
    if (nt <= 1) { fn(0, n); return; }
    // ~16 chunks per worker keeps uneven per-item cost balanced.
    const int64_t chunk = std::max(grain, n / ((int64_t)nt * 16));
    std::atomic<int64_t> next{0};
    run_workers(nt, [&](int) {
        for (;;) {
            const int64_t b = next.fetch_add(chunk, std::memory_order_relaxed);
            if (b >= n) break;
            fn(b, std::min(n, b + chunk));
        }
    });
}

void parallel_tasks(int64_t ntasks, const std::function<void(int64_t)>& fn) {
    if (ntasks <= 0) return;
    const int nt = (int)std::min<int64_t>(threads(), ntasks);
    if (nt <= 1) { for (int64_t i = 0; i < ntasks; ++i) fn(i); return; }
    std::atomic<int64_t> next{0};
    run_workers(nt, [&](int) {
        for (;;) {
            const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= ntasks) break;
            fn(i);
        }
    });
}

std::vector<int64_t> fixed_chunks(int64_t n, int64_t target_chunk) {
    std::vector<int64_t> b;
    if (n <= 0) { b.push_back(0); return b; }
    target_chunk = std::max<int64_t>(1, target_chunk);
    const int64_t nc = (n + target_chunk - 1) / target_chunk;
    b.resize((size_t)nc + 1);
    for (int64_t c = 0; c <= nc; ++c) b[(size_t)c] = n * c / nc;
    return b;
}

// ---------------------------------------------------------------------------
// debug hashing

bool hash_enabled() {
    const char* s = std::getenv("TRELLIS_POST_HASH");
    return s && *s && std::strcmp(s, "0") != 0;
}

uint64_t hash_bytes(const void* data, size_t bytes, uint64_t seed) {
    // 64-bit multiply-rotate hash over 8-byte words, length-terminated. Not
    // cryptographic; only used to compare buffers between runs.
    const unsigned char* p = (const unsigned char*)data;
    uint64_t h = seed ^ 0x9e3779b97f4a7c15ull ^ ((uint64_t)bytes * 0xff51afd7ed558ccdull);
    size_t i = 0;
    for (; i + 8 <= bytes; i += 8) {
        uint64_t w; std::memcpy(&w, p + i, 8);
        w *= 0x87c37b91114253d5ull; w = (w << 31) | (w >> 33); w *= 0x4cf5ad432745937full;
        h ^= w; h = (h << 27) | (h >> 37); h = h * 5 + 0x52dce729;
    }
    uint64_t t = 0;
    for (size_t k = 0; i + k < bytes; ++k) t |= (uint64_t)p[i + k] << (8 * k);
    h ^= t * 0x87c37b91114253d5ull;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdull; h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ull; h ^= h >> 33;
    return h;
}

void hash_report(const char* stage, const std::vector<float>& verts, const std::vector<int32_t>& faces) {
    if (!hash_enabled()) return;
    printf("  [post-hash] %-20s V=%-10zu F=%-10zu verts=%016llx faces=%016llx\n", stage, verts.size() / 3,
           faces.size() / 3, (unsigned long long)hash_bytes(verts.data(), verts.size() * sizeof(float)),
           (unsigned long long)hash_bytes(faces.data(), faces.size() * sizeof(int32_t)));
    fflush(stdout);
}

void hash_report_raw(const char* stage, const char* what, uint64_t h) {
    if (!hash_enabled()) return;
    printf("  [post-hash] %-20s %s=%016llx\n", stage, what, (unsigned long long)h);
    fflush(stdout);
}

// ---------------------------------------------------------------------------
// radix sort

void radix_sort_kv(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals) {
    const int64_t n = (int64_t)keys.size();
    if (n <= 1) return;
    if (n < 4096) {   // small: stable index sort
        std::vector<uint32_t> idx((size_t)n);
        for (int64_t i = 0; i < n; ++i) idx[(size_t)i] = (uint32_t)i;
        std::stable_sort(idx.begin(), idx.end(), [&](uint32_t a, uint32_t b) { return keys[a] < keys[b]; });
        std::vector<uint64_t> k2((size_t)n); std::vector<uint32_t> v2((size_t)n);
        for (int64_t i = 0; i < n; ++i) { k2[(size_t)i] = keys[idx[(size_t)i]]; v2[(size_t)i] = vals[idx[(size_t)i]]; }
        keys.swap(k2); vals.swap(v2);
        return;
    }
    // Chunking only affects scheduling: a stable sort has exactly one result.
    const int64_t nc = std::max<int64_t>(1, std::min<int64_t>((int64_t)threads() * 4, n / 2048));
    std::vector<int64_t> cb((size_t)nc + 1);
    for (int64_t c = 0; c <= nc; ++c) cb[(size_t)c] = n * c / nc;
    std::vector<uint64_t> k2((size_t)n);
    std::vector<uint32_t> v2((size_t)n);
    std::vector<int64_t> hist((size_t)nc * 256);
    // Which digits vary at all: OR/AND over all keys.
    uint64_t all_or = 0, all_and = ~0ull;
    {
        std::vector<uint64_t> po((size_t)nc, 0), pa((size_t)nc, ~0ull);
        parallel_tasks(nc, [&](int64_t c) {
            uint64_t o = 0, a = ~0ull;
            for (int64_t i = cb[(size_t)c]; i < cb[(size_t)c + 1]; ++i) { o |= keys[(size_t)i]; a &= keys[(size_t)i]; }
            po[(size_t)c] = o; pa[(size_t)c] = a;
        });
        for (int64_t c = 0; c < nc; ++c) { all_or |= po[(size_t)c]; all_and &= pa[(size_t)c]; }
    }
    const uint64_t varying = all_or ^ all_and;
    for (int shift = 0; shift < 64; shift += 8) {
        if (((varying >> shift) & 0xff) == 0) continue;
        std::fill(hist.begin(), hist.end(), 0);
        parallel_tasks(nc, [&](int64_t c) {
            int64_t* h = &hist[(size_t)c * 256];
            for (int64_t i = cb[(size_t)c]; i < cb[(size_t)c + 1]; ++i) ++h[(keys[(size_t)i] >> shift) & 0xff];
        });
        int64_t sum = 0;
        for (int d = 0; d < 256; ++d)
            for (int64_t c = 0; c < nc; ++c) {
                int64_t& h = hist[(size_t)c * 256 + d];
                const int64_t t = h; h = sum; sum += t;
            }
        parallel_tasks(nc, [&](int64_t c) {
            int64_t* h = &hist[(size_t)c * 256];
            for (int64_t i = cb[(size_t)c]; i < cb[(size_t)c + 1]; ++i) {
                const int64_t pos = h[(keys[(size_t)i] >> shift) & 0xff]++;
                k2[(size_t)pos] = keys[(size_t)i];
                v2[(size_t)pos] = vals[(size_t)i];
            }
        });
        keys.swap(k2); vals.swap(v2);
    }
}

// ---------------------------------------------------------------------------
// flat map

void FlatU64Map::init(int64_t n) {
    uint64_t cap = 16;
    while (cap < (uint64_t)n * 2) cap <<= 1;
    mask_ = cap - 1;
    has_empty_ = false; empty_val_ = -1;
    keys_.reset(new std::atomic<uint64_t>[cap]);
    vals_.reset(new int32_t[cap]);
    parallel_for((int64_t)cap, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) { keys_[(size_t)i].store(kEmpty, std::memory_order_relaxed); vals_[(size_t)i] = -1; }
    });
}

void FlatU64Map::build(const uint64_t* keys, const int32_t* vals, int64_t n) {
    init(n);
    for (int64_t i = 0; i < n; ++i)   // at most one sentinel key; handled outside the table
        if (keys[i] == kEmpty) { has_empty_ = true; empty_val_ = vals[i]; break; }
    parallel_for(n, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i)
            if (keys[i] != kEmpty) insert(keys[i], vals[i]);
    });
}

// ---------------------------------------------------------------------------
// edge CSR

int64_t vertex_bound(const int32_t* faces, size_t n) {
    std::atomic<int64_t> mx{-1};
    parallel_for((int64_t)n, 1 << 16, [&](int64_t b, int64_t e) {
        int64_t m = -1;
        for (int64_t i = b; i < e; ++i) m = std::max<int64_t>(m, faces[i]);
        int64_t cur = mx.load();
        while (m > cur && !mx.compare_exchange_weak(cur, m)) {}
    });
    return mx.load() + 1;
}

EdgeCsr build_edge_csr(const int32_t* faces, int64_t F, int64_t V) {
    EdgeCsr c;
    c.off.assign((size_t)V + 1, 0);
    const int64_t NC = F * 3;
    std::unique_ptr<std::atomic<int32_t>[]> cnt(new std::atomic<int32_t>[(size_t)std::max<int64_t>(V, 1)]);
    parallel_for(V, 1 << 16, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) cnt[(size_t)i].store(0, std::memory_order_relaxed);
    });
    parallel_for(F, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f)
            for (int j = 0; j < 3; ++j) {
                const int32_t u = faces[3*f+j], v = faces[3*f+(j+1)%3];
                cnt[(size_t)std::min(u, v)].fetch_add(1, std::memory_order_relaxed);
            }
    });
    // prefix sum (two-level, parallel)
    {
        const std::vector<int64_t> cb = fixed_chunks(V, 1 << 18);
        const int64_t nc = (int64_t)cb.size() - 1;
        std::vector<int64_t> part((size_t)nc + 1, 0);
        parallel_tasks(nc, [&](int64_t k) {
            int64_t s = 0;
            for (int64_t i = cb[(size_t)k]; i < cb[(size_t)k + 1]; ++i) s += cnt[(size_t)i].load(std::memory_order_relaxed);
            part[(size_t)k + 1] = s;
        });
        for (int64_t k = 0; k < nc; ++k) part[(size_t)k + 1] += part[(size_t)k];
        parallel_tasks(nc, [&](int64_t k) {
            int64_t s = part[(size_t)k];
            for (int64_t i = cb[(size_t)k]; i < cb[(size_t)k + 1]; ++i) {
                c.off[(size_t)i] = s;
                s += cnt[(size_t)i].load(std::memory_order_relaxed);
                cnt[(size_t)i].store(0, std::memory_order_relaxed);   // reused as fill cursor
            }
        });
        c.off[(size_t)V] = part[(size_t)nc];
    }
    c.ent.resize((size_t)NC);
    parallel_for(F, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t f = b; f < e; ++f)
            for (int j = 0; j < 3; ++j) {
                const int32_t u = faces[3*f+j], v = faces[3*f+(j+1)%3];
                const int32_t a = std::min(u, v), o = std::max(u, v);
                const int64_t pos = c.off[(size_t)a] + cnt[(size_t)a].fetch_add(1, std::memory_order_relaxed);
                c.ent[(size_t)pos] = ((uint64_t)(uint32_t)o << 32) | (uint32_t)(3*f + j);
            }
    });
    cnt.reset();
    // Sort each bucket by (other, corner): makes the layout independent of the
    // fill order above.
    parallel_for(V, 1 << 12, [&](int64_t b, int64_t e) {
        for (int64_t a = b; a < e; ++a) {
            uint64_t* lo = c.ent.data() + c.off[(size_t)a];
            uint64_t* hi = c.ent.data() + c.off[(size_t)a + 1];
            if (hi - lo <= 16) {
                for (uint64_t* p = lo + 1; p < hi; ++p) {
                    const uint64_t x = *p; uint64_t* q = p;
                    while (q > lo && q[-1] > x) { *q = q[-1]; --q; }
                    *q = x;
                }
            } else {
                std::sort(lo, hi);
            }
        }
    });
    return c;
}

}  // namespace post
}  // namespace trellis
