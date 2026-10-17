// CPU-only checks for the opt-in "views as framed" multi-view mode: per-view cameras, the
// orthographic projection, the transforms.json reader, the no-crop framing and the flags.
// No models, no GPU, no ggml: build `trellis-test-mv-views` and run it.
#include "birefnet.h"
#include "mv_views.h"
#include "pixal3d.h"
#include "preprocess.h"
#include "trellis_args.h"
#include "trellis_model.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <stdexcept>
#include <cstdio>
#include <string>
#include <vector>

// preprocess.cpp is linked without ggml: stand in for the BiRefNet network with a fixed matte
// (everything inside a centred square is foreground) so birefnet_framed can run on CPU.
namespace trellis {
std::vector<float> birefnet_matte(const Model&, const std::vector<float>&, int) {
    const int R = 1024;
    std::vector<float> l((size_t)R * R, -10.0f);
    for (int y = R / 4; y < 3 * R / 4; ++y) for (int x = R / 4; x < R / 2; ++x) l[(size_t)y * R + x] = 10.0f;   // left half of the centre square
    return l;
}
}

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)
#define NEAR(a, b, tol) do { const double _a = (a), _b = (b); if (!(std::fabs(_a - _b) <= (tol))) { std::fprintf(stderr, "FAIL line %d: %s = %.6f, want %.6f (+-%g)\n", __LINE__, #a, _a, _b, (double)(tol)); ++bad; } } while (0)

static constexpr double PI = 3.14159265358979323846;
static constexpr float CAX = 0.8575560450553894f;

// upstream assets/mv_images/example/transforms.json, verbatim apart from whitespace
static const char* kExample = R"({
  "camera_angle_x": 0.3490658503988659, "mesh_scale": 1.0,
  "frames": [
    {"file_path": "view00_azim000.png", "transform_matrix": [[1.0,0.0,0.0,0.0],[0.0,0.0,-1.0,-3.1192049980163574],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]]},
    {"file_path": "view01_azim090.png", "transform_matrix": [[0.0,0.0,1.0,3.1192049980163574],[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]]},
    {"file_path": "view02_azim180.png", "transform_matrix": [[-1.0,0.0,0.0,0.0],[0.0,0.0,1.0,3.1192049980163574],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]]},
    {"file_path": "view03_azim270.png", "transform_matrix": [[0.0,0.0,-1.0,-3.1192049980163574],[-1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,0.0,1.0]]}
  ]})";

static bool parse(std::vector<std::string> args, trellis::TrellisParams& p) {
    std::vector<char*> argv;
    static std::string prog = "trellis-cli";
    argv.push_back(&prog[0]);
    for (auto& a : args) argv.push_back(&a[0]);
    return trellis::parse_args((int)argv.size(), argv.data(), p);
}

static void test_camera() {
    using namespace trellis;
    // explicit camera with the closed-form distance reproduces the default camera exactly
    const CameraParams a = pixal3d_camera(CAX, 1.0f, 512, 0);
    const CameraParams b = pixal3d_camera_explicit(CAX, a.distance, 1.0f);
    CHECK(b.ortho_half_width == 0.0f);
    for (int tok : {0, 1, 777, 4095}) {
        float ax, ay, bx, by;
        pixal3d_project_cell(16, tok / 256, (tok / 16) % 16, tok % 16, a, 512, ax, ay);
        pixal3d_project_cell(16, tok / 256, (tok / 16) % 16, tok % 16, b, 512, bx, by);
        NEAR(ax, bx, 1e-4); NEAR(ay, by, 1e-4);
    }
    // a different distance really moves the projection (the old code shared one distance)
    const CameraParams far_cam = pixal3d_camera_explicit(CAX, a.distance * 2.0f, 1.0f);
    float x1, y1, x2, y2;
    pixal3d_project_cell(16, 15, 15, 15, a, 512, x1, y1);
    pixal3d_project_cell(16, 15, 15, 15, far_cam, 512, x2, y2);
    CHECK(std::fabs(x1 - x2) > 10.0f);
    // bad inputs
    int thrown = 0;
    for (float d : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN()}) {
        try { (void)pixal3d_camera_explicit(CAX, d); } catch (const std::invalid_argument&) { ++thrown; }
    }
    try { (void)pixal3d_camera_explicit(CAX, 1.0f, 1.0f, -0.5f); } catch (const std::invalid_argument&) { ++thrown; }
    try { (void)pixal3d_camera_explicit(0.0f, 1.0f); } catch (const std::invalid_argument&) { ++thrown; }
    CHECK(thrown == 5);
}

static void test_ortho() {
    using namespace trellis;
    // MV-Adapter framing: half-width 0.55, distance 1.8; the frame spans +-0.55 world units.
    CameraParams o = pixal3d_camera_explicit(CAX, 1.8f, 1.0f, 0.55f);
    const double k = 256.0 / 0.55;
    // grid cell centre coordinates for R=2 are +-1, world = grid/2 (mesh_scale 1)
    float px, py;
    pixal3d_project_cell(2, 1, 1, 1, o, 512, px, py);          // gx=1 -> wx=0.5; gy=1 -> wz=0.5
    NEAR(px, 256.0 + k * 0.5, 1e-3); NEAR(py, 256.0 - k * 0.5, 1e-3);
    pixal3d_project_cell(2, 0, 0, 0, o, 512, px, py);
    NEAR(px, 256.0 - k * 0.5, 1e-3); NEAR(py, 256.0 + k * 0.5, 1e-3);
    // depth does not matter: moving along the viewing axis (gz -> wy) leaves the pixel alone
    float qx, qy;
    pixal3d_project_cell(2, 1, 1, 0, o, 512, px, py);
    pixal3d_project_cell(2, 1, 1, 1, o, 512, qx, qy);
    NEAR(px, qx, 1e-4); NEAR(py, qy, 1e-4);
    // and neither does the camera distance or fov
    CameraParams o2 = pixal3d_camera_explicit(0.2f, 9.0f, 1.0f, 0.55f);
    pixal3d_project_cell(2, 1, 1, 1, o2, 512, qx, qy);
    NEAR(px, qx, 1e-4); NEAR(py, qy, 1e-4);
    // resolution invariance, as for the perspective camera
    float a1, a2, b1, b2;
    pixal3d_project_cell(16, 3, 9, 5, o, 512, a1, a2);
    pixal3d_project_cell(16, 3, 9, 5, o, 1024, b1, b2);
    NEAR(b1 - 512.0, 2.0 * (a1 - 256.0), 2e-3); NEAR(b2 - 512.0, 2.0 * (a2 - 256.0), 2e-3);
    // azimuth 90: the screen x axis becomes the old depth axis (wy = -gz/2)
    o.azimuth = (float)(PI / 2);
    pixal3d_project_cell(2, 0, 1, 1, o, 512, px, py);           // gz=1 -> wy=-0.5 -> rotated wx' = -0.5
    NEAR(px, 256.0 - k * 0.5, 1e-3);
    // mesh_scale halves the world extent of the grid
    CameraParams m = pixal3d_camera_explicit(CAX, 1.8f, 2.0f, 0.55f);
    pixal3d_project_cell(2, 1, 1, 1, m, 512, px, py);
    NEAR(px, 256.0 + k * 0.25, 1e-3);
    // a far perspective camera converges to the orthographic one with k = f_px / distance
    const double D = 200.0, f_px = 16.0 / std::tan(CAX * 0.5) * 512.0 / 32.0;
    CameraParams far_p = pixal3d_camera_explicit(CAX, (float)D, 1.0f);
    CameraParams eq = pixal3d_camera_explicit(CAX, (float)D, 1.0f, (float)(256.0 * D / f_px));
    for (int tok : {0, 13, 600, 4095}) {
        float fx, fy, ex, ey;
        pixal3d_project_cell(16, tok / 256, (tok / 16) % 16, tok % 16, far_p, 512, fx, fy);
        pixal3d_project_cell(16, tok / 256, (tok / 16) % 16, tok % 16, eq, 512, ex, ey);
        CHECK(std::fabs(fx - ex) < 1.5 && std::fabs(fy - ey) < 1.5);
    }
}

static void test_transforms() {
    using namespace trellis;
    MvTransforms t; std::string err;
    CHECK(parse_transforms_json(kExample, t, err));
    CHECK(t.cams.size() == 4 && t.files.size() == 4);
    CHECK(t.files[1] == "view01_azim090.png");
    NEAR(t.mesh_scale, 1.0, 1e-9);
    const double want_az[4] = {0, 90, 180, 270};
    for (int i = 0; i < 4; ++i) {
        NEAR(t.cams[i].fov_rad, 0.3490658503988659, 1e-6);
        NEAR(t.cams[i].distance, 3.1192049980163574, 1e-5);
        double az = t.cams[i].azimuth_rad * 180.0 / PI; if (az < 0) az += 360.0;
        NEAR(az, want_az[i], 1e-3);
    }
    // main view not at azimuth 0: azimuths are relative to frame 0
    const std::string rot = R"({"camera_angle_x":0.5,"frames":[
      {"file_path":"a.png","transform_matrix":[[0,0,1,2],[1,0,0,0],[0,1,0,0],[0,0,0,1]]},
      {"file_path":"b.png","camera_angle_x":0.25,"transform_matrix":[[-1,0,0,0],[0,0,1,3],[0,1,0,0],[0,0,0,1]]}]})";
    CHECK(parse_transforms_json(rot, t, err));
    NEAR(t.cams[1].azimuth_rad * 180.0 / PI, 90.0, 1e-3);     // 180 - 90
    NEAR(t.cams[1].fov_rad, 0.25, 1e-6); NEAR(t.cams[0].fov_rad, 0.5, 1e-6);
    NEAR(t.cams[1].distance, 3.0, 1e-6);
    // refusals
    const std::string elev = R"({"camera_angle_x":0.5,"frames":[{"file_path":"a.png","transform_matrix":[[1,0,0,0],[0,0,-1,-2],[0,1,0,1],[0,0,0,1]]}]})";
    CHECK(!parse_transforms_json(elev, t, err) && err.find("elevation") != std::string::npos);
    CHECK(!parse_transforms_json("{\"frames\":[]}", t, err));
    CHECK(!parse_transforms_json("{\"frames\":[{\"file_path\":\"a\",\"transform_matrix\":[[1]]}]}", t, err));
    CHECK(!parse_transforms_json("{\"camera_angle_x\":0.5,\"frames\":[{\"file_path\":\"a\"}]}", t, err));
    CHECK(!parse_transforms_json("not json", t, err));
    CHECK(!parse_transforms_json("{\"frames\": [", t, err));
    CHECK(!parse_transforms_json("{\"mesh_scale\":-1,\"camera_angle_x\":0.5,\"frames\":[{\"file_path\":\"a\",\"transform_matrix\":[[1,0,0,0],[0,0,-1,-2],[0,1,0,0],[0,0,0,1]]}]}", t, err));
}

static void test_resolve() {
    using namespace trellis;
    std::vector<ViewCam> v; std::string err;
    const std::vector<float> none;
    // defaults: every view gets the single-view fov and its closed-form distance
    CHECK(resolve_view_cameras(3, none, none, none, {90, 180}, nullptr, CAX, 1.0f, 0, v, err));
    CHECK(v.size() == 3);
    const float d0 = pixal3d_camera(CAX, 1.0f, 512, 0).distance;
    for (const auto& c : v) { NEAR(c.fov_rad, CAX, 1e-6); NEAR(c.distance, d0, 1e-6); CHECK(c.ortho_half_width == 0.0f); }
    NEAR(v[0].azimuth_rad, 0.0, 1e-9); NEAR(v[2].azimuth_rad, PI, 1e-6);
    // per-view flags, broadcast, main first
    CHECK(resolve_view_cameras(2, {20, 30}, {1.8f}, {0, 0.55f}, {90}, nullptr, CAX, 1.0f, 0, v, err));
    NEAR(v[0].fov_rad, 20 * PI / 180, 1e-6); NEAR(v[1].fov_rad, 30 * PI / 180, 1e-6);
    NEAR(v[0].distance, 1.8, 1e-6); NEAR(v[1].distance, 1.8, 1e-6);
    CHECK(v[0].ortho_half_width == 0.0f); NEAR(v[1].ortho_half_width, 0.55, 1e-6);
    // wrong list length
    CHECK(!resolve_view_cameras(3, {20, 30}, none, none, {90, 180}, nullptr, CAX, 1.0f, 0, v, err) && err.find("--view-fov") != std::string::npos);
    CHECK(resolve_view_cameras(2, none, none, none, {}, nullptr, CAX, 1.0f, 0, v, err) && v[1].azimuth_rad == 0.0f);   // no azimuth source: azimuth 0 (trellis_run rejects this case before getting here)
    CHECK(!resolve_view_cameras(2, none, none, none, {90, 180}, nullptr, CAX, 1.0f, 0, v, err));
    CHECK(!resolve_view_cameras(1, {0.0f}, none, none, {}, nullptr, CAX, 1.0f, 0, v, err));
    CHECK(!resolve_view_cameras(1, {200.0f}, none, none, {}, nullptr, CAX, 1.0f, 0, v, err));
    // from a transforms.json; flags override it
    MvTransforms t;
    CHECK(parse_transforms_json(kExample, t, err));
    CHECK(resolve_view_cameras(4, none, none, none, {}, &t, CAX, 1.0f, 0, v, err));
    NEAR(v[1].azimuth_rad, PI / 2, 1e-5); NEAR(v[3].distance, 3.1192049980163574, 1e-5); NEAR(v[2].fov_rad, 0.3490658503988659, 1e-6);
    CHECK(resolve_view_cameras(4, {25}, {2.0f}, none, {}, &t, CAX, 1.0f, 0, v, err));
    NEAR(v[1].fov_rad, 25 * PI / 180, 1e-6); NEAR(v[1].distance, 2.0, 1e-6); NEAR(v[1].azimuth_rad, PI / 2, 1e-5);
    CHECK(!resolve_view_cameras(3, none, none, none, {}, &t, CAX, 1.0f, 0, v, err) && err.find("frames") != std::string::npos);
    // the camera handed to the projection carries azimuth, distance and ortho
    CHECK(resolve_view_cameras(2, {20}, {1.8f}, {0.55f}, {180}, nullptr, CAX, 1.0f, 0, v, err));
    const CameraParams cp = view_camera_params(v[1], 1.0f);
    NEAR(cp.azimuth, PI, 1e-6); NEAR(cp.distance, 1.8, 1e-6); NEAR(cp.ortho_half_width, 0.55, 1e-6); NEAR(cp.camera_angle_x, 20 * PI / 180, 1e-6);
}

static void test_framing() {
    using namespace trellis;
    // 8x8 view with a small 2x2 opaque patch in the lower-left, the rest transparent background.
    const int N = 8;
    std::vector<unsigned char> rgba((size_t)N * N * 4, 255);
    std::vector<float> alpha((size_t)N * N, 0.0f);
    for (int y = 5; y < 7; ++y) for (int x = 1; x < 3; ++x) {
        const size_t p = (size_t)y * N + x;
        alpha[p] = 1.0f; rgba[p * 4] = 200; rgba[p * 4 + 1] = 100; rgba[p * 4 + 2] = 50;
    }
    alpha[(size_t)5 * N + 3] = 0.5f;     // a soft edge pixel
    std::vector<unsigned char> out; int sz = 0; std::string err;
    CHECK(frame_view_rgba(rgba.data(), N, N, alpha, out, sz, err));
    CHECK(sz == N && out.size() == (size_t)N * N * 4);                 // not shrunk to the 2x2 bbox
    // the patch is exactly where it was: no recentring, no rescale
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x) {
        const size_t p = ((size_t)y * N + x) * 4;
        const bool in = (y == 5 || y == 6) && (x == 1 || x == 2);
        if (in) { CHECK(out[p] == 200 && out[p + 1] == 100 && out[p + 2] == 50 && out[p + 3] == 255); }
        else if (y == 5 && x == 3) { CHECK(out[p + 3] == 128); CHECK(out[p] == (unsigned char)(255 * 0.5f)); }   // premultiplied
        else CHECK(out[p] == 0 && out[p + 1] == 0 && out[p + 2] == 0 && out[p + 3] == 0);
    }
    // bbox centre of the foreground is unchanged (the single-view crop would move it to the middle)
    int xmin = N, xmax = -1, ymin = N, ymax = -1;
    for (int y = 0; y < N; ++y) for (int x = 0; x < N; ++x)
        if (out[((size_t)y * N + x) * 4 + 3] > 200) { xmin = std::min(xmin, x); xmax = std::max(xmax, x); ymin = std::min(ymin, y); ymax = std::max(ymax, y); }
    CHECK(xmin == 1 && xmax == 2 && ymin == 5 && ymax == 6);
    // an all-background view stays all-background and is not replaced by the whole frame
    std::vector<float> none((size_t)N * N, 0.0f);
    CHECK(frame_view_rgba(rgba.data(), N, N, none, out, sz, err));
    bool all0 = true; for (unsigned char b : out) all0 = all0 && b == 0; CHECK(all0);
    // non-square and size mismatch are refused
    std::vector<float> a2((size_t)8 * 6, 1.0f);
    std::vector<unsigned char> r2((size_t)8 * 6 * 4, 9);
    CHECK(!frame_view_rgba(r2.data(), 8, 6, a2, out, sz, err) && err.find("square") != std::string::npos);
    CHECK(!frame_view_rgba(rgba.data(), N, N, a2, out, sz, err));
}


static std::string tmp_png(const char* name, int W, int H, const std::vector<unsigned char>& rgba) {
    const std::string p = std::string("/tmp/") + name;
    stbi_write_png(p.c_str(), W, H, 4, rgba.data(), W * 4);
    return p;
}

static void test_build_cameras() {
    using namespace trellis;
    const float RAD = (float)(PI / 180.0);
    MvTransforms t; std::string err;
    CHECK(parse_transforms_json(kExample, t, err));
    ViewCamRequest rq; ViewCamResult rs;
    // --fov is honoured with no file (it used to be dropped in this mode)
    rq.n_views = 2; rq.fov_flag_deg = 30.0f; rq.extra_azim_deg = {90};
    CHECK(build_view_cameras(rq, rs, err));
    NEAR(rs.cams[0].camera_angle_x, 30 * RAD, 1e-6); NEAR(rs.cams[1].camera_angle_x, 30 * RAD, 1e-6);
    NEAR(rs.cams[1].distance, pixal3d_camera(30 * RAD, 1.0f, 512, 0).distance, 1e-5);      // closed form from the same fov
    CHECK(rs.warnings.empty());
    // no flag at all: Pixal3D's default
    rq.fov_flag_deg = 0.0f;
    CHECK(build_view_cameras(rq, rs, err));
    NEAR(rs.cams[0].camera_angle_x, CAX, 1e-6);
    // --view-fov beats --fov
    rq.fov_flag_deg = 30.0f; rq.view_fov_deg = {20, 25};
    CHECK(build_view_cameras(rq, rs, err));
    NEAR(rs.cams[0].camera_angle_x, 20 * RAD, 1e-6); NEAR(rs.cams[1].camera_angle_x, 25 * RAD, 1e-6);
    // mesh_scale: flag wins over the file; the file wins over the default
    ViewCamRequest f; f.n_views = 4; f.transforms = &t;
    t.mesh_scale = 1.7f;
    CHECK(build_view_cameras(f, rs, err));
    NEAR(rs.mesh_scale, 1.7, 1e-6); NEAR(rs.cams[2].mesh_scale, 1.7, 1e-6); CHECK(rs.warnings.empty());
    f.mesh_scale_set = true; f.mesh_scale_flag = 0.8f;
    CHECK(build_view_cameras(f, rs, err));
    NEAR(rs.mesh_scale, 0.8, 1e-6); NEAR(rs.cams[3].mesh_scale, 0.8, 1e-6); CHECK(rs.warnings.size() == 1);
    f.mesh_scale_flag = 1.7f;
    CHECK(build_view_cameras(f, rs, err) && rs.warnings.empty());      // same value: nothing to say
    // a file with no mesh_scale parses to 1 and an explicit --mesh-scale still wins
    CHECK(parse_transforms_json(kExample, t, err));
    f.mesh_scale_flag = 2.0f;
    CHECK(build_view_cameras(f, rs, err)); NEAR(rs.mesh_scale, 2.0, 1e-6);
    // file: fov/distance/azimuth come through untouched
    ViewCamRequest g; g.n_views = 4; g.transforms = &t;
    CHECK(build_view_cameras(g, rs, err) && rs.warnings.empty());
    NEAR(rs.cams[1].camera_angle_x, 0.3490658503988659, 1e-6); NEAR(rs.cams[1].distance, 3.1192049980163574, 1e-5); NEAR(rs.cams[1].azimuth, PI / 2, 1e-5);
    // fov override on a file keeps the file's distance and warns about it
    g.view_fov_deg = {30};
    CHECK(build_view_cameras(g, rs, err));
    NEAR(rs.cams[0].camera_angle_x, 30 * RAD, 1e-6); NEAR(rs.cams[0].distance, 3.1192049980163574, 1e-5);
    CHECK(rs.warnings.size() == 1 && rs.warnings[0].find("distance") != std::string::npos);
    g.view_distance = {2.5f};                                            // ... unless the distance is given too
    CHECK(build_view_cameras(g, rs, err) && rs.warnings.empty());
    NEAR(rs.cams[3].distance, 2.5, 1e-6);
    g.view_fov_deg.clear(); g.view_distance.clear(); g.fov_flag_deg = 30.0f;     // --fov overrides the file too
    CHECK(build_view_cameras(g, rs, err));
    NEAR(rs.cams[2].camera_angle_x, 30 * RAD, 1e-6); CHECK(rs.warnings.size() == 1);
    // orthographic on top of a file: ortho half-width set, the file's distance, azimuth and fov kept
    ViewCamRequest o; o.n_views = 4; o.transforms = &t; o.view_ortho = {0.55f};
    CHECK(build_view_cameras(o, rs, err) && rs.warnings.empty());
    for (int i = 0; i < 4; ++i) { NEAR(rs.cams[i].ortho_half_width, 0.55, 1e-6); NEAR(rs.cams[i].distance, 3.1192049980163574, 1e-5); }
    NEAR(rs.cams[3].azimuth * 180.0 / PI, -90.0, 1e-3);                   // 270 as -90
    float px, py;      // the ortho camera really is orthographic: depth no longer matters
    pixal3d_project_cell(2, 1, 1, 0, rs.cams[0], 512, px, py);
    float qx, qy; pixal3d_project_cell(2, 1, 1, 1, rs.cams[0], 512, qx, qy);
    NEAR(px, qx, 1e-4); NEAR(py, qy, 1e-4);
    // per-view ortho: main perspective, the others orthographic
    o.view_ortho = {0, 0.55f};
    o.n_views = 2; o.transforms = nullptr; o.extra_azim_deg = {180}; o.view_distance = {1.8f};
    CHECK(build_view_cameras(o, rs, err));
    CHECK(rs.cams[0].ortho_half_width == 0.0f); NEAR(rs.cams[1].ortho_half_width, 0.55, 1e-6); NEAR(rs.cams[1].azimuth, PI, 1e-6);
    // errors
    ViewCamRequest bad_r; bad_r.n_views = 3; bad_r.transforms = &t;       // 4 frames, 3 pictures
    CHECK(!build_view_cameras(bad_r, rs, err) && err.find("frames") != std::string::npos);
}

static void test_matte() {
    using namespace trellis;
    // auto: alpha if the picture has one, else BiRefNet, else refuse (never the white rule by accident)
    CHECK(choose_matte(true,  -1, true)  == Matte::Alpha);
    CHECK(choose_matte(true,  -1, false) == Matte::Alpha);
    CHECK(choose_matte(false, -1, true)  == Matte::BiRefNet);
    CHECK(choose_matte(false, -1, false) == Matte::Refuse);
    // explicit flags
    CHECK(choose_matte(false, 1, true)  == Matte::BiRefNet);
    CHECK(choose_matte(true,  1, true)  == Matte::BiRefNet);
    CHECK(choose_matte(false, 0, true)  == Matte::WhiteThreshold);
    CHECK(choose_matte(true,  0, true)  == Matte::Alpha);
    // the failure the review found: a main picture WITH alpha must not decide the matte of the others
    const bool has_alpha[3] = {true, false, false};
    for (bool a : has_alpha) CHECK(choose_matte(a, -1, true) == (a ? Matte::Alpha : Matte::BiRefNet));

    // threshold_alpha: own alpha when real, white rule otherwise
    std::vector<unsigned char> px = {255,255,255,255,  10,20,30,255,  240,240,250,255,  200,200,200,255};
    std::vector<float> al;
    CHECK(!threshold_alpha(px.data(), 4, al));
    CHECK(al[0] == 0.0f && al[1] == 1.0f && al[2] == 0.0f && al[3] == 1.0f);          // white and near-white are background
    std::vector<unsigned char> pa = {255,255,255,255,  10,20,30,128,  240,240,250,0,  200,200,200,249};
    CHECK(threshold_alpha(pa.data(), 4, al));
    NEAR(al[0], 1.0, 1e-6); NEAR(al[1], 128.0 / 255, 1e-6); NEAR(al[2], 0.0, 1e-6);   // a white pixel with alpha 255 stays foreground
    // grey MV-Adapter-style background (0.5 grey, no alpha) under the white rule becomes foreground: the reason Refuse exists
    std::vector<unsigned char> grey = {128,128,128,255, 128,128,128,255};
    CHECK(!threshold_alpha(grey.data(), 2, al) && al[0] == 1.0f && al[1] == 1.0f);
}

static void test_framed_files() {
    using namespace trellis;
    // 16x16 view, an opaque 4x4 patch near the top-left over a transparent background
    const int N = 16;
    std::vector<unsigned char> rgba((size_t)N * N * 4, 0);
    for (int y = 2; y < 6; ++y) for (int x = 3; x < 7; ++x) { size_t p = ((size_t)y * N + x) * 4; rgba[p] = 220; rgba[p + 1] = 80; rgba[p + 2] = 40; rgba[p + 3] = 255; }
    const std::string sq = tmp_png("mv_views_square.png", N, N, rgba);
    int W = 0, H = 0;
    CHECK(image_size(sq, W, H) && W == N && H == N);
    CHECK(!image_size("/tmp/mv_views_does_not_exist.png", W, H));
    CHECK(image_has_alpha(sq));
    int sz = 0;
    std::vector<unsigned char> out = threshold_framed(sq, sz);
    CHECK(sz == N && out.size() == (size_t)N * N * 4);
    CHECK(out[((size_t)2 * N + 3) * 4 + 3] == 255 && out[((size_t)2 * N + 3) * 4] == 220);        // patch still at (3,2)
    CHECK(out[((size_t)1 * N + 3) * 4 + 3] == 0 && out[((size_t)2 * N + 2) * 4 + 3] == 0);
    // the cropped path would have shrunk it: the framed one must not
    CHECK(sz != 6);
    // non-square: image_size reports it (the CLI refuses before loading any model) and the framing refuses it too
    std::vector<unsigned char> wide((size_t)16 * 8 * 4, 255);
    const std::string wp = tmp_png("mv_views_wide.png", 16, 8, wide);
    CHECK(image_size(wp, W, H) && W == 16 && H == 8);
    CHECK(threshold_framed(wp, sz).empty() && sz == 0);
    // BiRefNet path (stub matte): same size out, matte applied, nothing cropped. An opaque RGB picture (no alpha).
    std::vector<unsigned char> opaque((size_t)N * N * 4, 255);
    for (size_t i = 0; i < (size_t)N * N; ++i) { opaque[i * 4] = 128; opaque[i * 4 + 1] = 128; opaque[i * 4 + 2] = 128; }   // 0.5 grey everywhere
    const std::string gp = tmp_png("mv_views_grey.png", N, N, opaque);
    CHECK(!image_has_alpha(gp));
    Model bm;
    out = birefnet_framed(gp, bm, 0, sz);
    CHECK(sz == N && out.size() == (size_t)N * N * 4);
    CHECK(out[((size_t)8 * N + 5) * 4 + 3] == 255 && out[((size_t)8 * N + 5) * 4] == 128);       // inside the stub's foreground
    CHECK(out[((size_t)8 * N + 12) * 4 + 3] == 0 && out[((size_t)8 * N + 12) * 4] == 0);         // outside: transparent, not the grey
    CHECK(out[((size_t)1 * N + 1) * 4 + 3] == 0);
    // and under the white rule the same grey picture is entirely foreground (what the per-picture matte choice avoids)
    out = threshold_framed(gp, sz);
    CHECK(out[((size_t)1 * N + 1) * 4 + 3] == 255);
    std::remove(sq.c_str()); std::remove(wp.c_str()); std::remove(gp.c_str());
}

static void test_args() {
    trellis::TrellisParams p;
    CHECK(!p.views_as_framed && p.view_fov_deg.empty() && p.view_distance.empty() && p.view_ortho.empty() && p.views_transforms.empty());   // default behaviour untouched
    CHECK(parse({"a.png", "o.glb"}, p));
    CHECK(!p.views_as_framed);
    trellis::TrellisParams q;
    CHECK(parse({"a.png", "o.glb", "--model", "pixal3d", "--views", "b.png,c.png", "--view-azim", "90,180", "--views-as-framed",
                 "--view-fov", "20,20,20", "--view-distance", "1.8", "--view-ortho", "0.55"}, q));
    CHECK(q.views_as_framed && q.views.size() == 2 && q.view_azim.size() == 2);
    CHECK(q.view_fov_deg.size() == 3 && q.view_distance.size() == 1 && q.view_ortho.size() == 1);
    NEAR(q.view_ortho[0], 0.55, 1e-6);
    trellis::TrellisParams r;
    CHECK(parse({"a.png", "o.glb", "--views-transforms", "t.json"}, r));
    CHECK(r.views_as_framed && r.views_transforms == "t.json");
    trellis::TrellisParams z;
    CHECK(parse({"a.png", "o.glb", "--view-ortho", "0,0.55"}, z));     // 0 = perspective is allowed for ortho only
    CHECK(z.view_ortho.size() == 2 && z.view_ortho[0] == 0.0f);
    for (const char* flag : {"--view-fov", "--view-distance"})
        for (const char* bad_v : {"0", "-1", "nan", "inf", "", "1,,2", "x", "1,"}) {
            trellis::TrellisParams b;
            CHECK(!parse({"a.png", "o.glb", flag, bad_v}, b));
        }
    for (const char* bad_v : {"-1", "nan", "x"}) { trellis::TrellisParams b; CHECK(!parse({"a.png", "o.glb", "--view-ortho", bad_v}, b)); }
}

int main() {
    test_camera();
    test_ortho();
    test_transforms();
    test_resolve();
    test_framing();
    test_args();
    test_build_cameras();
    test_matte();
    test_framed_files();
    std::printf(bad ? "\n%d check(s) FAILED\n" : "all checks passed\n", bad);
    return bad ? 1 : 0;
}
