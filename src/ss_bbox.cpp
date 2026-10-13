#include "ss_bbox.h"

#include <algorithm>
#include <cstdio>

namespace trellis {

SsBox ss_bbox(const std::vector<std::array<int,3>>& coords) {
    SsBox b;
    if (coords.empty()) return b;
    b.voxels = (int)coords.size();
    b.mn = b.mx = coords[0];
    for (const auto& c : coords)
        for (int a = 0; a < 3; ++a) { b.mn[a] = std::min(b.mn[a], c[a]); b.mx[a] = std::max(b.mx[a], c[a]); }
    for (int a = 0; a < 3; ++a) b.extent[a] = b.mx[a] - b.mn[a] + 1;
    return b;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (unsigned char ch : s) {
        switch (ch) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (ch < 0x20) { char u[8]; std::snprintf(u, sizeof u, "\\u%04x", ch); o += u; }
                else o += (char)ch;
        }
    }
    return o;
}

static std::string triple(const std::array<int,3>& v) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "[%d,%d,%d]", v[0], v[1], v[2]);
    return buf;
}

std::string ss_json(const std::string& image, const std::string& model, int steps, float gss, int grid,
                    const std::vector<std::pair<uint32_t, SsBox>>& runs) {
    char head[160];
    std::snprintf(head, sizeof head, "\"model\":\"%s\",\"steps\":%d,\"gss\":%g,\"grid\":%d,", model.c_str(), steps, gss, grid);
    std::string o = "{\"tool\":\"trellis-cli --ss-only\",\"image\":\"" + json_escape(image) + "\"," + head +
                    "\"axes\":\"voxel x,y,z = width,depth,up (Blender Y,X,Z after finish.py rotate-z)\",\"seeds\":[";
    for (size_t i = 0; i < runs.size(); ++i) {
        const SsBox& b = runs[i].second;
        char seed[48];
        std::snprintf(seed, sizeof seed, "%s{\"seed\":%u,\"voxels\":%d,", i ? "," : "", runs[i].first, b.voxels);
        o += seed;
        o += "\"min\":" + triple(b.mn) + ",\"max\":" + triple(b.mx) + ",\"extent\":" + triple(b.extent) + "}";
    }
    return o + "]}\n";
}

}  // namespace trellis
