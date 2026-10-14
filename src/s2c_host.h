// Host-side bookkeeping for the sparse space-to-channel (S2C) down block, the inverse of the
// decoder's channel-to-space (C2S) up block. Pure functions, no ggml, so they can be unit
// tested on any machine (src/test_s2c_host.cpp).
//
// Octant convention matches sparse_c2s: child offset o = (x&1) | (y&1)<<1 | (z&1)<<2, and the
// stacked channel of child o's feature k is o*K + k. [M: official SparseSpatial2Channel order,
// to be confirmed by the reconstruction test on real weights.]
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace trellis {

struct S2CMap {
    std::vector<std::array<int,3>> parents;   // [P] coords at res/2, sorted by (x,y,z)
    std::vector<int32_t> child_parent;        // [N] parent row of each fine voxel
    std::vector<uint8_t> child_oct;           // [N] octant 0..7 of each fine voxel
};

inline uint64_t s2c_key(int x, int y, int z) {
    return ((uint64_t)(uint32_t)x << 40) | ((uint64_t)(uint32_t)y << 20) | (uint32_t)z;
}

inline S2CMap build_s2c_map(const std::vector<std::array<int,3>>& coords) {
    S2CMap r;
    const size_t N = coords.size();
    std::vector<uint64_t> keys(N);
    for (size_t i = 0; i < N; ++i)
        keys[i] = s2c_key(coords[i][0] >> 1, coords[i][1] >> 1, coords[i][2] >> 1);
    std::vector<uint64_t> uniq = keys;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    std::unordered_map<uint64_t, int32_t> row;
    row.reserve(uniq.size() * 2);
    r.parents.resize(uniq.size());
    for (size_t p = 0; p < uniq.size(); ++p) {
        row[uniq[p]] = (int32_t)p;
        r.parents[p] = { (int)(uniq[p] >> 40), (int)((uniq[p] >> 20) & 0xFFFFF), (int)(uniq[p] & 0xFFFFF) };
    }
    r.child_parent.resize(N);
    r.child_oct.resize(N);
    for (size_t i = 0; i < N; ++i) {
        r.child_parent[i] = row[keys[i]];
        r.child_oct[i] = (uint8_t)((coords[i][0] & 1) | ((coords[i][1] & 1) << 1) | ((coords[i][2] & 1) << 2));
    }
    return r;
}

// [K,N] channel-major fine features -> [8K,P]: child o of parent p lands in channels
// o*K .. o*K+K-1; absent children stay zero.
inline std::vector<float> s2c_stack(const std::vector<float>& h, int K, const S2CMap& mp) {
    const size_t P = mp.parents.size(), N = mp.child_parent.size();
    std::vector<float> out((size_t)8 * K * P, 0.0f);
    for (size_t i = 0; i < N; ++i)
        std::copy_n(h.data() + (size_t)K * i, K,
                    out.data() + (size_t)8 * K * mp.child_parent[i] + (size_t)mp.child_oct[i] * K);
    return out;
}

// Skip path of the down block: stack the raw input the same way, view the 8*Cin channels as
// [Cout, G] (G = 8*Cin/Cout, groups contiguous) and average over G. Returns [Cout,P].
inline std::vector<float> s2c_skip(const std::vector<float>& x, int Cin, int Cout, const S2CMap& mp) {
    const size_t P = mp.parents.size(), N = mp.child_parent.size();
    const int G = 8 * Cin / Cout;
    std::vector<float> out((size_t)Cout * P, 0.0f);
    const float inv = 1.0f / (float)G;
    for (size_t i = 0; i < N; ++i) {
        float* dst = out.data() + (size_t)Cout * mp.child_parent[i];
        const float* src = x.data() + (size_t)Cin * i;
        const int base = mp.child_oct[i] * Cin;
        for (int c = 0; c < Cin; ++c) dst[(base + c) / G] += src[c] * inv;
    }
    return out;
}

// Compact the neighbour columns an output chunk [r0, r0+nr) touches (see sparse_c2s).
struct S2CCompact {
    std::vector<int32_t> table;     // [27*nr] tap-major local indices, sentinel = globals.size()
    std::vector<int32_t> globals;   // local column -> source column
};
inline S2CCompact s2c_compact(const std::vector<int32_t>& full, int full_n, int64_t r0, int nr) {
    S2CCompact o;
    o.table.resize((size_t)27 * nr);
    std::unordered_map<int32_t, int32_t> remap;
    remap.reserve((size_t)nr * 2 + 1);
    for (int t = 0; t < 27; ++t)
        for (int j = 0; j < nr; ++j) {
            const int32_t g = full[(size_t)t * full_n + (size_t)r0 + j];
            if (g == full_n) { o.table[(size_t)t * nr + j] = -1; continue; }
            auto it = remap.find(g);
            if (it == remap.end()) {
                it = remap.emplace(g, (int32_t)o.globals.size()).first;
                o.globals.push_back(g);
            }
            o.table[(size_t)t * nr + j] = it->second;
        }
    const int32_t sent = (int32_t)o.globals.size();
    for (int32_t& v : o.table) if (v < 0) v = sent;
    return o;
}
inline std::vector<float> s2c_gather_cols(const std::vector<float>& full, int C, const std::vector<int32_t>& globals) {
    std::vector<float> local((size_t)C * (globals.size() + 1), 0.0f);
    for (size_t li = 0; li < globals.size(); ++li)
        std::copy_n(full.data() + (size_t)C * globals[li], C, local.data() + (size_t)C * li);
    return local;
}

} // namespace trellis
