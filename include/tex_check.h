#pragma once
// Colour check helper for --tex-reroll. Pure C++, no ggml.
#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace trellis {

// Which sparse voxels a camera could see from outside. A voxel counts as exposed when it is the first or the last
// voxel along its row in at least one of the three axis directions (a six-way orthographic depth test). Hidden inner
// shells, enclosed cavities and the far side of thick walls fail all six tests, so a brightness average over the
// exposed voxels follows what a viewer sees instead of every surface voxel. Result[i] is 1 for coords[i] exposed.
inline std::vector<uint8_t> exposed_voxels(const std::vector<std::array<int,3>>& coords) {
    std::vector<uint8_t> vis(coords.size(), 0);
    for (int ax = 0; ax < 3; ++ax) {
        const int a = (ax + 1) % 3, b = (ax + 2) % 3;
        struct Span { int lo, hi; };
        std::unordered_map<uint64_t, Span> rows;
        rows.reserve(coords.size());
        auto key = [&](const std::array<int,3>& c) {
            return ((uint64_t)(uint32_t)c[a] << 32) | (uint32_t)c[b];
        };
        for (const auto& c : coords) {
            auto it = rows.find(key(c));
            if (it == rows.end()) rows.emplace(key(c), Span{c[ax], c[ax]});
            else { if (c[ax] < it->second.lo) it->second.lo = c[ax]; if (c[ax] > it->second.hi) it->second.hi = c[ax]; }
        }
        for (size_t i = 0; i < coords.size(); ++i) {
            const Span& s = rows.find(key(coords[i]))->second;
            if (coords[i][ax] == s.lo || coords[i][ax] == s.hi) vis[i] = 1;
        }
    }
    return vis;
}

}  // namespace trellis
