// Arctic: shared helpers for the multithreaded CPU post-processing stages
// (weld, hole filling, BVH build, narrow-band remesh, clean). std::thread only,
// portable C++17, no OpenMP, no std::execution.
//
// Environment:
//   TRELLIS_POST_THREADS=N  worker threads for the post stages (default: hardware_concurrency)
//   TRELLIS_POST_LEGACY=1   run the original single-threaded hash-map code paths (A/B checks)
//   TRELLIS_POST_HASH=1     print a 64-bit hash of the mesh buffers after each post stage
//
// Every rewritten stage must give byte-identical output to the legacy path at
// any thread count; see docs/arctic/post-mt.md.
#pragma once
#include <cstddef>
#include <cstdint>
#include <atomic>
#include <functional>
#include <memory>
#include <vector>

namespace trellis {
namespace post {

int  threads();      // >= 1
bool legacy();       // TRELLIS_POST_LEGACY set and not "0"

// Runs fn(begin, end) over [0, n) in dynamically scheduled chunks of at least
// `grain` items. Runs inline when one thread suffices. Exceptions thrown by fn
// are rethrown on the calling thread.
void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn);

// Runs fn(task) for task in [0, ntasks), dynamically scheduled, one task at a time.
void parallel_tasks(int64_t ntasks, const std::function<void(int64_t)>& fn);

// Splits [0, n) into a fixed number of contiguous chunks that depends only on n
// (never on the thread count), so per-chunk outputs concatenated in chunk order
// are deterministic. Returns the chunk boundaries (size nchunks + 1).
std::vector<int64_t> fixed_chunks(int64_t n, int64_t target_chunk = 1 << 16);

// Debug hashing (TRELLIS_POST_HASH=1).
bool hash_enabled();
uint64_t hash_bytes(const void* data, size_t bytes, uint64_t seed = 0);
void hash_report(const char* stage, const std::vector<float>& verts, const std::vector<int32_t>& faces);
void hash_report_raw(const char* stage, const char* what, uint64_t h);

// Stable parallel LSD radix sort of 64-bit keys carrying a 32-bit payload.
// Equal keys keep their input order. Passes whose digit is constant are skipped.
void radix_sort_kv(std::vector<uint64_t>& keys, std::vector<uint32_t>& vals);

// Open-addressing u64 -> int32 map, built once (in parallel) and then read-only.
class FlatU64Map {
public:
    // keys must be unique.
    void build(const uint64_t* keys, const int32_t* vals, int64_t n);
    // Incremental form: init(n) sizes the table for n keys, then insert() may be
    // called concurrently from any number of threads (unique keys, key != ~0ull).
    void init(int64_t n);
    void insert(uint64_t key, int32_t val) {
        size_t s = (size_t)slot_of(key);
        for (;;) {
            uint64_t expected = kEmpty;
            if (keys_[s].compare_exchange_strong(expected, key, std::memory_order_relaxed) || expected == key) {
                vals_[s] = val;
                return;
            }
            s = (s + 1) & mask_;
        }
    }
    // Returns -1 when the key is absent.
    int32_t find(uint64_t key) const {
        if (key == kEmpty) return has_empty_ ? empty_val_ : -1;
        size_t i = (size_t)slot_of(key);
        for (;;) {
            const uint64_t k = keys_[i].load(std::memory_order_relaxed);
            if (k == key) return vals_[i];
            if (k == kEmpty) return -1;
            i = (i + 1) & mask_;
        }
    }

private:
    static constexpr uint64_t kEmpty = ~0ull;
    uint64_t slot_of(uint64_t k) const {
        k ^= k >> 33; k *= 0xff51afd7ed558ccdull; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ull; k ^= k >> 33;
        return k & mask_;
    }
    std::unique_ptr<std::atomic<uint64_t>[]> keys_;
    std::unique_ptr<int32_t[]> vals_;
    uint64_t mask_ = 0;
    bool has_empty_ = false;
    int32_t empty_val_ = -1;
};

// Undirected edge -> face-corner incidence in CSR form, grouped by the edge's
// smaller vertex. For bucket a = min(u,v), entries [off[a], off[a+1]) hold
// (other = max(u,v), corner = 3*f + j) for every face edge j of face f
// (j: faces[3f+j] -> faces[3f+(j+1)%3]), sorted by (other, corner). A run of
// equal `other` is one undirected edge; its corners are in face order, which
// matches the push order of the legacy `unordered_map<edge, vector<face>>`.
struct EdgeCsr {
    std::vector<int64_t> off;     // V + 1
    std::vector<uint64_t> ent;    // 3F entries, (uint64_t)other << 32 | corner
    static int32_t other(uint64_t e) { return (int32_t)(e >> 32); }
    static int32_t corner(uint64_t e) { return (int32_t)(uint32_t)e; }
};
// V = number of vertices; every index in faces must be in [0, V).
EdgeCsr build_edge_csr(const int32_t* faces, int64_t F, int64_t V);
// max index + 1 (0 when empty).
int64_t vertex_bound(const int32_t* faces, size_t n);

// Rewritten stages (dispatched from the legacy functions unless legacy()).
int  weld_vertices(std::vector<float>& verts, std::vector<int32_t>& faces, std::vector<float>* colors3, float step);
void clean_mesh(int V, std::vector<int32_t>& faces);
int  drop_small_components(std::vector<float>& verts, std::vector<int32_t>& faces, float frac);

}  // namespace post
}  // namespace trellis
