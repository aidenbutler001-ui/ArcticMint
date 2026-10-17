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

// Premultiplied square RGBA of a view exactly as framed: same pixels, same size, no crop, no
// rescale, no recentring. `alpha` is [W*H] in 0..1. Non-square input is refused (it would have
// to be padded or squashed, and either changes the framing the camera describes).
bool frame_view_rgba(const unsigned char* rgba, int W, int H, const std::vector<float>& alpha,
                     std::vector<unsigned char>& out, int& sz, std::string& error);

} // namespace trellis
