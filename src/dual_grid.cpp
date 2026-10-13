#include "dual_grid.h"
#include "post_parallel.h"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace trellis {

// edge_neighbor_voxel_offset[axis][corner][xyz] (from o_voxel flexible_dual_grid.py)
static const int OFF[3][4][3] = {
    { {0,0,0}, {0,0,1}, {0,1,1}, {0,1,0} },   // x-axis
    { {0,0,0}, {1,0,0}, {1,0,1}, {0,0,1} },   // y-axis
    { {0,0,0}, {0,1,0}, {1,1,0}, {1,0,0} },   // z-axis
};

static inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

Mesh dual_grid_to_mesh(const ShapeOut& so) {
    const int N = (int)so.coords.size();
    const float* F7 = so.feats7.data();   // [7, N] channel-major: F7[ch + 7*i]
    const float vs = 1.0f / so.res, org = -0.5f;
    const bool fast = !post::legacy();   // Arctic: parallel + flat table, identical output

    // dual vertices: world pos = (coord + (2*sigmoid(off)-0.5)) * vs - 0.5
    Mesh mesh; mesh.verts.resize((size_t)N * 3);
    auto fill_verts = [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i)
            for (int a = 0; a < 3; ++a) {
                float dv = 2.0f * sigmoidf(F7[a + 7*i]) - 0.5f;
                mesh.verts[3*i + a] = ((float)so.coords[i][a] + dv) * vs + org;
            }
    };
    if (fast) post::parallel_for(N, 1 << 14, fill_verts); else fill_verts(0, N);

    // coord hashmap
    auto key = [](int x,int y,int z){ return ((uint64_t)(uint32_t)x<<40)|((uint64_t)(uint32_t)y<<20)|(uint32_t)z; };
    // quad split for the 4 voxels of an intersected edge
    auto emit = [&](int a, const int q[4], std::vector<int32_t>& out) {
        (void)a;
        float w02 = softplusf(F7[6 + 7*q[0]]) * softplusf(F7[6 + 7*q[2]]);
        float w13 = softplusf(F7[6 + 7*q[1]]) * softplusf(F7[6 + 7*q[3]]);
        if (w02 > w13) {   // quad_split_1 = [0,1,2, 0,2,3]
            out.insert(out.end(), { q[0],q[1],q[2], q[0],q[2],q[3] });
        } else {           // quad_split_2 = [0,1,3, 3,1,2]
            out.insert(out.end(), { q[0],q[1],q[3], q[3],q[1],q[2] });
        }
    };

    if (!fast) {
        std::unordered_map<uint64_t,int> cmap; cmap.reserve(N * 2);
        for (int i = 0; i < N; ++i) cmap[key(so.coords[i][0], so.coords[i][1], so.coords[i][2])] = i;

        for (int i = 0; i < N; ++i) {
            const int x = so.coords[i][0], y = so.coords[i][1], z = so.coords[i][2];
            for (int a = 0; a < 3; ++a) {
                if (F7[3 + a + 7*i] <= 0.0f) continue;        // intersected edge?
                int q[4]; bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    auto it = cmap.find(key(x + OFF[a][k][0], y + OFF[a][k][1], z + OFF[a][k][2]));
                    if (it == cmap.end()) { ok = false; break; }
                    q[k] = it->second;
                }
                if (!ok) continue;
                emit(a, q, mesh.faces);
            }
        }
        return mesh;
    }

    // Fast path. Duplicate coordinates: the legacy map keeps the last index, insert_max does the same.
    post::FlatU64Map cmap;
    cmap.init(N);
    post::parallel_for(N, 1 << 14, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; ++i) cmap.insert_max(key(so.coords[i][0], so.coords[i][1], so.coords[i][2]), (int32_t)i);
    });
    // Faces per fixed chunk of voxels, concatenated in chunk order = legacy voxel/axis order.
    const std::vector<int64_t> cb = post::fixed_chunks(N, 1 << 14);
    const int64_t nc = (int64_t)cb.size() - 1;
    std::vector<std::vector<int32_t>> parts((size_t)nc);
    post::parallel_tasks(nc, [&](int64_t c) {
        std::vector<int32_t>& out = parts[(size_t)c];
        for (int64_t i = cb[(size_t)c]; i < cb[(size_t)c + 1]; ++i) {
            const int x = so.coords[i][0], y = so.coords[i][1], z = so.coords[i][2];
            for (int a = 0; a < 3; ++a) {
                if (F7[3 + a + 7*i] <= 0.0f) continue;        // intersected edge?
                int q[4]; bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    const int id = cmap.find(key(x + OFF[a][k][0], y + OFF[a][k][1], z + OFF[a][k][2]));
                    if (id < 0) { ok = false; break; }
                    q[k] = id;
                }
                if (!ok) continue;
                emit(a, q, out);
            }
        }
    });
    std::vector<int64_t> off((size_t)nc + 1, 0);
    for (int64_t c = 0; c < nc; ++c) off[(size_t)c + 1] = off[(size_t)c] + (int64_t)parts[(size_t)c].size();
    mesh.faces.resize((size_t)off[(size_t)nc]);
    post::parallel_tasks(nc, [&](int64_t c) {
        std::copy(parts[(size_t)c].begin(), parts[(size_t)c].end(), mesh.faces.begin() + off[(size_t)c]);
        std::vector<int32_t>().swap(parts[(size_t)c]);
    });
    return mesh;
}

} // namespace trellis
