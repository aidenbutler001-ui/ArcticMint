#include "mv_views.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>

namespace trellis {

namespace {

// Minimal JSON reader: enough for transforms.json (objects, arrays, numbers, strings, literals).
struct J {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    double n = 0; std::string s; std::vector<J> a; std::map<std::string, J> o;
};

struct Reader {
    const std::string& s; size_t i = 0; std::string err;
    explicit Reader(const std::string& str) : s(str) {}
    void ws() { while (i < s.size() && std::isspace((unsigned char)s[i])) ++i; }
    bool fail(const char* m) { if (err.empty()) err = std::string("json: ") + m + " at byte " + std::to_string(i); return false; }
    bool str(std::string& out) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i; out.clear();
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                switch (s[i]) { case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'u': out += '?'; i += 4; break; default: out += s[i]; }
            } else out += s[i];
            ++i;
        }
        if (i >= s.size()) return fail("unterminated string");
        ++i; return true;
    }
    bool val(J& v, int depth = 0) {
        if (depth > 32) return fail("nested too deep");
        ws();
        if (i >= s.size()) return fail("unexpected end");
        char c = s[i];
        if (c == '{') {
            v.t = J::Obj; ++i; ws();
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                ws(); std::string k; if (!str(k)) return false;
                ws(); if (i >= s.size() || s[i] != ':') return fail("expected ':'"); ++i;
                if (!val(v.o[k], depth + 1)) return false;
                ws(); if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            v.t = J::Arr; ++i; ws();
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                v.a.emplace_back();
                if (!val(v.a.back(), depth + 1)) return false;
                ws(); if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { v.t = J::Str; return str(v.s); }
        if (s.compare(i, 4, "true") == 0)  { v.t = J::Bool; v.n = 1; i += 4; return true; }
        if (s.compare(i, 5, "false") == 0) { v.t = J::Bool; v.n = 0; i += 5; return true; }
        if (s.compare(i, 4, "null") == 0)  { v.t = J::Null; i += 4; return true; }
        char* e = nullptr; const double d = std::strtod(s.c_str() + i, &e);
        if (e == s.c_str() + i) return fail("unexpected character");
        v.t = J::Num; v.n = d; i = (size_t)(e - s.c_str()); return true;
    }
};

const J* member(const J& o, const char* k) {
    if (o.t != J::Obj) return nullptr;
    auto it = o.o.find(k);
    return it == o.o.end() ? nullptr : &it->second;
}

} // namespace

bool parse_transforms_json(const std::string& text, MvTransforms& out, std::string& error) {
    out = MvTransforms{};
    Reader r(text); J root;
    if (!r.val(root)) { error = r.err; return false; }
    if (root.t != J::Obj) { error = "transforms.json: top level must be an object"; return false; }
    const J* frames = member(root, "frames");
    if (!frames || frames->t != J::Arr || frames->a.empty()) { error = "transforms.json: 'frames' must be a non-empty array"; return false; }
    if (const J* ms = member(root, "mesh_scale")) {
        if (ms->t != J::Num || !std::isfinite(ms->n) || ms->n <= 0) { error = "transforms.json: bad mesh_scale"; return false; }
        out.mesh_scale = (float)ms->n;
    }
    const J* top_fov = member(root, "camera_angle_x");
    float az0 = 0.0f;
    for (size_t f = 0; f < frames->a.size(); ++f) {
        const J& fr = frames->a[f];
        const std::string tag = "transforms.json frame " + std::to_string(f) + ": ";
        const J* fp = member(fr, "file_path");
        if (!fp || fp->t != J::Str) { error = tag + "missing file_path"; return false; }
        const J* fov = member(fr, "camera_angle_x"); if (!fov) fov = top_fov;
        if (!fov || fov->t != J::Num || !std::isfinite(fov->n) || fov->n <= 0 || fov->n >= 3.14159265358979) {
            error = tag + "missing or invalid camera_angle_x (radians)"; return false;
        }
        const J* tm = member(fr, "transform_matrix");
        if (!tm || tm->t != J::Arr || tm->a.size() != 4) { error = tag + "transform_matrix must be 4x4"; return false; }
        double M[4][4];
        for (int r4 = 0; r4 < 4; ++r4) {
            const J& row = tm->a[r4];
            if (row.t != J::Arr || row.a.size() != 4) { error = tag + "transform_matrix must be 4x4"; return false; }
            for (int c = 0; c < 4; ++c) {
                if (row.a[c].t != J::Num || !std::isfinite(row.a[c].n)) { error = tag + "non-numeric transform_matrix entry"; return false; }
                M[r4][c] = row.a[c].n;
            }
        }
        const double px = M[0][3], py = M[1][3], pz = M[2][3];
        const double d = std::sqrt(px * px + py * py + pz * pz);
        if (!(d > 0)) { error = tag + "camera sits at the origin"; return false; }
        if (std::fabs(pz) > 1e-3 * d) { error = tag + "camera is off the equator (elevation != 0); the engine has no elevation"; return false; }
        ViewCam v;
        v.fov_rad = (float)fov->n;
        v.distance = (float)d;
        // Canonical front camera sits at (0,-d,0); azimuth 90 at (+d,0,0): azimuth = atan2(x, -y).
        const float az = (float)std::atan2(px, -py);
        if (f == 0) az0 = az;
        float rel = az - az0;
        const float PI = 3.14159265358979f;
        while (rel > PI) rel -= 2 * PI;
        while (rel <= -PI) rel += 2 * PI;
        v.azimuth_rad = rel;
        out.files.push_back(fp->s);
        out.cams.push_back(v);
    }
    return true;
}

static bool pick(const std::vector<float>& list, size_t n, size_t i, const char* name,
                 float& val, std::string& error) {
    if (list.empty()) return false;
    if (list.size() != 1 && list.size() != n) {
        error = std::string(name) + " needs 1 value (all views) or " + std::to_string(n) + " (one per view, main first), got " + std::to_string(list.size());
        val = -1; return false;
    }
    val = list.size() == 1 ? list[0] : list[i];
    return true;
}

bool resolve_view_cameras(size_t n, const std::vector<float>& fov_deg, const std::vector<float>& distance,
                          const std::vector<float>& ortho, const std::vector<float>& extra_azim_deg,
                          const MvTransforms* tf, float default_fov_rad, float mesh_scale, int extend_pixel,
                          std::vector<ViewCam>& out, std::string& error) {
    out.clear();
    if (n == 0) { error = "no views"; return false; }
    if (tf && tf->cams.size() != n) {
        error = "transforms.json has " + std::to_string(tf->cams.size()) + " frames but " + std::to_string(n) + " pictures were given (main + --views)";
        return false;
    }
    if (!extra_azim_deg.empty() && extra_azim_deg.size() != n - 1) { error = "--view-azim needs one value per extra view"; return false; }
    const float PI = 3.14159265358979f;
    for (size_t i = 0; i < n; ++i) {
        ViewCam v;
        float x = 0;
        error.clear();
        if (pick(fov_deg, n, i, "--view-fov", x, error)) v.fov_rad = x * PI / 180.0f;
        else if (!error.empty()) return false;
        else v.fov_rad = tf ? tf->cams[i].fov_rad : default_fov_rad;
        if (pick(distance, n, i, "--view-distance", x, error)) v.distance = x;
        else if (!error.empty()) return false;
        else if (tf) v.distance = tf->cams[i].distance;
        else v.distance = 0.0f;      // derived below from this view's fov
        if (pick(ortho, n, i, "--view-ortho", x, error)) v.ortho_half_width = x;
        else if (!error.empty()) return false;
        if (i > 0) {
            if (!extra_azim_deg.empty()) v.azimuth_rad = extra_azim_deg[i - 1] * PI / 180.0f;
            else if (tf) v.azimuth_rad = tf->cams[i].azimuth_rad;
        }
        if (!(v.fov_rad > 0 && v.fov_rad < PI) || !std::isfinite(v.fov_rad)) { error = "view " + std::to_string(i) + ": fov must be between 0 and 180 degrees"; return false; }
        if (v.distance == 0.0f) {
            try { v.distance = pixal3d_camera(v.fov_rad, mesh_scale, 512, extend_pixel).distance; }
            catch (const std::exception& e) { error = e.what(); return false; }
        }
        if (!(v.distance > 0) || !std::isfinite(v.distance)) { error = "view " + std::to_string(i) + ": distance must be positive"; return false; }
        if (!(v.ortho_half_width >= 0) || !std::isfinite(v.ortho_half_width)) { error = "view " + std::to_string(i) + ": ortho half-width must be >= 0"; return false; }
        out.push_back(v);
    }
    return true;
}

CameraParams view_camera_params(const ViewCam& v, float mesh_scale) {
    CameraParams c = pixal3d_camera_explicit(v.fov_rad, v.distance, mesh_scale, v.ortho_half_width);
    c.azimuth = v.azimuth_rad;
    return c;
}

bool frame_view_rgba(const unsigned char* rgba, int W, int H, const std::vector<float>& alpha,
                     std::vector<unsigned char>& out, int& sz, std::string& error) {
    sz = 0; out.clear();
    if (W <= 0 || H <= 0 || alpha.size() != (size_t)W * H) { error = "bad view dimensions"; return false; }
    if (W != H) {
        error = "view is " + std::to_string(W) + "x" + std::to_string(H) + "; --views-as-framed needs square pictures "
                "(cropping or padding would change the framing the camera describes)";
        return false;
    }
    sz = W;
    out.assign((size_t)W * H * 4, 0);
    for (size_t p = 0; p < (size_t)W * H; ++p) {
        const float a = std::min(std::max(alpha[p], 0.0f), 1.0f);
        for (int c = 0; c < 3; ++c) out[p * 4 + c] = (unsigned char)(rgba[p * 4 + c] * a);
        out[p * 4 + 3] = (unsigned char)std::lround(255.0f * a);
    }
    return true;
}

} // namespace trellis
