// "Views as framed" multi-view helpers (pure C++, no ggml, unit-testable on CPU).
//
// Upstream Pixal3D multi-view (inference_mv.py) never crops or rescales a view: transforms.json
// describes each picture's framing as given, with a per-view fov and camera distance. These
// helpers carry that contract into the engine: read a transforms.json, resolve one camera per
// view (from the file, from flags, or the single-view defaults), and frame a view without the
// single-view bounding-box crop.
#pragma once
#include "pixal3d.h"
#include <string>
#include <vector>

namespace trellis {

// One view's camera. azimuth is relative to the main view (frame 0), counter-clockwise from
// above, radians; 0 for the main view.
struct ViewCam {
    float fov_rad = 0.0f;
    float distance = 0.0f;
    float azimuth_rad = 0.0f;
    float ortho_half_width = 0.0f;     // 0 = perspective
};

struct MvTransforms {
    float mesh_scale = 1.0f;
    std::vector<std::string> files;    // file_path of each frame, in order (frame 0 = main)
    std::vector<ViewCam> cams;         // fov from camera_angle_x, distance = |translation|, azimuth relative to frame 0
};

// Parse an upstream-style transforms.json (Blender/NeRF convention: Z-up world, camera at
// translation of the camera-to-world matrix, camera_angle_x per frame or top level, optional
// mesh_scale). Cameras must lie on the equator (the engine has no elevation); a frame above or
// below it is an error. Returns false and sets `error` on anything malformed.
bool parse_transforms_json(const std::string& text, MvTransforms& out, std::string& error);

// Resolve one camera per view (n_views = 1 main + extras). Precedence per field: explicit list
// (a list of size 1 is broadcast, otherwise it must have n_views entries), then the transforms
// file (when given), then the single-view defaults: `default_fov_rad` and the closed-form
// distance derived from it. Azimuths come from `extra_azim_deg` (n_views - 1 entries, as
// --view-azim) or, when empty, from the transforms file; the main view is always 0.
bool resolve_view_cameras(size_t n_views,
                          const std::vector<float>& fov_deg,
                          const std::vector<float>& distance,
                          const std::vector<float>& ortho_half_width,
                          const std::vector<float>& extra_azim_deg,
                          const MvTransforms* transforms,
                          float default_fov_rad, float mesh_scale, int extend_pixel,
                          std::vector<ViewCam>& out, std::string& error);

CameraParams view_camera_params(const ViewCam& v, float mesh_scale);

// Everything that decides the cameras of a views-as-framed run, as plain values.
struct ViewCamRequest {
    size_t n_views = 1;                    // main + extras
    float fov_flag_deg = 0.0f;            // --fov (0 = not given). In this mode it behaves like --view-fov with one value.
    std::vector<float> view_fov_deg, view_distance, view_ortho, extra_azim_deg;
    bool  mesh_scale_set = false;         // --mesh-scale given
    float mesh_scale_flag = 1.0f;         // --mesh-scale (or its default)
    int   extend_pixel = 0;
    const MvTransforms* transforms = nullptr;
};
struct ViewCamResult {
    std::vector<CameraParams> cams;       // one per view, main first
    float mesh_scale = 1.0f;
    std::vector<std::string> warnings;
};
// Flags > transforms file > single-view defaults, per field:
//   mesh_scale : --mesh-scale, else the file's, else 1.
//   fov        : --view-fov, else --fov, else the file's, else Pixal3D's default 49.13 deg.
//   distance   : --view-distance, else the file's, else closed-form from that view's fov.
// An fov override on top of a file keeps the file's distance (the two are independent in
// transforms.json) and says so in `warnings`.
bool build_view_cameras(const ViewCamRequest& req, ViewCamResult& out, std::string& error);

// Which matte a picture gets in views-as-framed mode, decided per picture (not from the main one).
//   birefnet_flag: 1 = --bg-removal birefnet, 0 = threshold, -1 = auto.
enum class Matte { Alpha, BiRefNet, WhiteThreshold, Refuse };
Matte choose_matte(bool has_alpha, int birefnet_flag, bool birefnet_available);

// The threshold matte: the picture's own alpha when it has a real one (any alpha < 250),
// otherwise the white-background rule (min(RGB) < 232 is foreground). Returns whether the
// picture had an alpha.
bool threshold_alpha(const unsigned char* rgba, size_t npix, std::vector<float>& alpha);

// Premultiplied square RGBA of a view exactly as framed: same pixels, same size, no crop, no
// rescale, no recentring. `alpha` is [W*H] in 0..1. Non-square input is refused (it would have
// to be padded or squashed, and either changes the framing the camera describes).
bool frame_view_rgba(const unsigned char* rgba, int W, int H, const std::vector<float>& alpha,
                     std::vector<unsigned char>& out, int& sz, std::string& error);

} // namespace trellis
