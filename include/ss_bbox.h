#pragma once
// Sparse-structure bounding box for `trellis-cli --ss-only` (docs/arctic/ss-only.md). Pure C++, no ggml.
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace trellis {

struct SsBox {
    int voxels = 0;
    std::array<int,3> mn{0, 0, 0}, mx{0, 0, 0};   // lowest / highest active cell on each voxel axis (x, y, z)
    std::array<int,3> extent{0, 0, 0};            // mx - mn + 1, in cells of the sparse-structure grid
};

// Box of the active cells ss_coords() returns (res-32 grid). Empty input gives voxels == 0 and zero extents.
SsBox ss_bbox(const std::vector<std::array<int,3>>& coords);

std::string json_escape(const std::string& s);

// One JSON document for a whole --ss-only run, one entry per seed. Voxel axes: x = width, y = depth, z = up; after
// finish.py's rotate-z they are Blender Y (W), X (D) and Z (H).
std::string ss_json(const std::string& image, const std::string& model, int steps, float gss, int grid,
                    const std::vector<std::pair<uint32_t, SsBox>>& runs);

}  // namespace trellis
