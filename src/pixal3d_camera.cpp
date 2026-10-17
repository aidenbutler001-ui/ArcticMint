// Camera model for the Pixal3D view-aligned projection (pure C++, no ggml: unit-testable on CPU).
#include "pixal3d.h"

#include <cmath>
#include <stdexcept>

namespace trellis {

// Blender's sensor model, as the reference reimplements it: a 32 mm sensor with the focal
// length recovered from the horizontal FOV.
static float focal_pixels(float camera_angle_x, int resolution) {
    const float focal = 16.0f / std::tan(camera_angle_x * 0.5f);
    return focal * (float)resolution / 32.0f;
}

CameraParams pixal3d_camera(float camera_angle_x, float mesh_scale,
                            int image_resolution, int extend_pixel) {
    if (!std::isfinite(camera_angle_x) || camera_angle_x <= 0 || camera_angle_x >= 3.14159265358979323846 ||
        !std::isfinite(mesh_scale) || mesh_scale <= 0 || image_resolution <= 0 || extend_pixel < 0)
        throw std::invalid_argument("pixal3d: invalid camera parameters");
    CameraParams cam;
    cam.camera_angle_x = camera_angle_x;
    cam.mesh_scale     = mesh_scale;
    // Reference: the grid corner (-1, 0, 0) maps through the same rotation as every other grid
    // point, giving world x = -0.5/mesh_scale, and is required to land on the left image border
    // (target x = -extend_pixel). Solving the perspective divide for the camera distance:
    //   distance = f_pixels * x_world / x_ndc  with  x_ndc = -extend_pixel - resolution/2
    const float f_px = focal_pixels(camera_angle_x, image_resolution);
    const float x_world = -0.5f / mesh_scale;
    const float x_ndc   = -(float)extend_pixel - (float)image_resolution * 0.5f;
    cam.distance = f_px * x_world / x_ndc;
    if (!std::isfinite(cam.distance) || cam.distance <= 0)
        throw std::invalid_argument("pixal3d: camera distance is not finite and positive");
    return cam;
}

void pixal3d_project_cell(int R, int cx, int cy, int cz, const CameraParams& cam,
                          int image_resolution, float& px, float& py) {
    const float den = R > 1 ? (float)(R - 1) : 1.0f;
    const float gx = -1.0f + 2.0f * (float)cx / den;
    const float gy = -1.0f + 2.0f * (float)cy / den;
    const float gz = -1.0f + 2.0f * (float)cz / den;
    // ProjGrid rotates the grid into Blender axes with [[1,0,0],[0,0,-1],[0,1,0]] and halves it
    // by mesh_scale; the frontal view matrix then reduces the world->camera transform to
    // x_cam = wx, y_cam = wz, z_cam = -wy - distance.
    const float s = 1.0f / (2.0f * cam.mesh_scale);
    float wx = gx * s, wy = -gz * s;
    const float wz = gy * s;
    if (cam.azimuth != 0.0f) {          // a camera orbited by +a sees the grid rotated by -a about world Z
        const float ca = std::cos(cam.azimuth), sa = std::sin(cam.azimuth);
        const float rx = wx * ca + wy * sa, ry = -wx * sa + wy * ca;
        wx = rx; wy = ry;
    }
    if (cam.ortho_half_width > 0.0f) {
        // Orthographic camera (opt-in, MV-Adapter views): no divide by depth, a constant
        // pixels-per-world-unit scale in which the frame spans +-ortho_half_width.
        const float k = (float)image_resolution * 0.5f / cam.ortho_half_width;
        px = k * wx + (float)image_resolution * 0.5f;
        py = -k * wz + (float)image_resolution * 0.5f;
    } else {
    const float depth = wy + cam.distance;
    const float f_px  = focal_pixels(cam.camera_angle_x, image_resolution);
    const float inv   = 1.0f / (depth + 1e-8f);
    px = f_px * wx * inv + (float)image_resolution * 0.5f;
    py = -f_px * wz * inv + (float)image_resolution * 0.5f;   // image y grows downward
    }
    if (!std::isfinite(px) || !std::isfinite(py))
        throw std::invalid_argument("pixal3d: camera projects a cell to non-finite coordinates");
}


CameraParams pixal3d_camera_explicit(float camera_angle_x, float distance, float mesh_scale,
                                     float ortho_half_width) {
    if (!std::isfinite(camera_angle_x) || camera_angle_x <= 0 || camera_angle_x >= 3.14159265358979323846 ||
        !std::isfinite(distance) || distance <= 0 ||
        !std::isfinite(mesh_scale) || mesh_scale <= 0 ||
        !std::isfinite(ortho_half_width) || ortho_half_width < 0)
        throw std::invalid_argument("pixal3d: invalid explicit camera parameters");
    CameraParams cam;
    cam.camera_angle_x   = camera_angle_x;
    cam.distance         = distance;
    cam.mesh_scale       = mesh_scale;
    cam.ortho_half_width = ortho_half_width;
    return cam;
}

} // namespace trellis
