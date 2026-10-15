// FlexiDualGrid shape VAE decoder: shape SLAT latent -> dual-grid head [7, M] @ res 512.
#pragma once
#include <vector>
#include <array>
#include <cstdint>

namespace trellis {
struct Model;

struct ShapeOut {
    std::vector<float> feats7;                 // [7*M] channel-major (ne0=7): per voxel dual-grid fields
    std::vector<std::array<int,3>> coords;     // [M] voxel coords at final resolution
    int res = 512;
    std::vector<std::vector<uint8_t>> subs;    // per-C2S binarized subdiv masks (for the tex decoder)
};

// latent: shape SLAT feats [32*N0] channel-major; coords0: active voxels. resolution = final
// grid size (coords0 res * 16): 512 for res-32 input, 1024 for res-64 input (cascade HR).
ShapeOut shape_decode(const Model& m, const std::vector<float>& latent,
                      const std::vector<std::array<int,3>>& coords0, int resolution = 512);

// Cascade upsample: run from_latent + the 4 C2S up-blocks and return the grown voxel coords
// (input res * 16), discarding feats. Used by the 1024 cascade to get fine coords from the LR slat.
std::vector<std::array<int,3>> shape_upsample(const Model& m, const std::vector<float>& latent,
                                              const std::vector<std::array<int,3>>& coords0);

// Texture (PBR) decoder: SparseUnetVaeDecoder driven by the shape decoder's `subs` so it grows
// the IDENTICAL voxel tree. tex_latent: [32*N0]; returns 6-ch PBR [6*M] (channel-major) at the
// SAME final coords/order as shape_decode (base_color3, metallic, roughness, alpha) — pre *0.5+0.5.
std::vector<float> tex_decode(const Model& m, const std::vector<float>& tex_latent,
                              const std::vector<std::array<int,3>>& coords0,
                              const std::vector<std::vector<uint8_t>>& subs);

// Texture VAE encoder (SparseUnetVaeEncoder, tex_enc.gguf): the inverse of tex_decode. pbr6 is
// [6*N] channel-major (base colour RGB, metallic, roughness, alpha) in the decoder's output
// range [-1,1] at the full-resolution voxel coords; returns the posterior mean/logvar
// ([32*P] each) on the res/16 grid, with the grid's coords (sorted by x,y,z, NOT the shape
// latent's order: remap by coordinate before mixing with the flow's latent).
struct TexEncOut {
    std::vector<float> mean, logvar;
    std::vector<std::array<int,3>> coords;
};
TexEncOut tex_encode(const Model& m, const std::vector<float>& pbr6,
                     const std::vector<std::array<int,3>>& coords);

} // namespace trellis
