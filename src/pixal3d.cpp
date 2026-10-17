#include "pixal3d.h"
#include "naf.h"
#include "trellis_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace trellis {

static constexpr int D_DINO   = 1024;   // DINOv3 ViT-L/16 width
static constexpr int N_GLOBAL = 5;      // cls + 4 register tokens
static constexpr int PATCH    = 16;

// Bilinear sample of a channel-major [C, Hf*Wf] map at a pixel position expressed in the
// `S`-sized image frame. Mirrors F.grid_sample(align_corners=False, padding_mode="border").
static void sample_bilinear(const float* map, int Hf, int Wf, int C, int S,
                            float px, float py, float* dst) {
    float gx = (px + 0.5f) * (float)Wf / (float)S - 0.5f;
    float gy = (py + 0.5f) * (float)Hf / (float)S - 0.5f;
    gx = std::min(std::max(gx, 0.0f), (float)Wf - 1.0f);
    gy = std::min(std::max(gy, 0.0f), (float)Hf - 1.0f);
    const int x0 = (int)std::floor(gx), y0 = (int)std::floor(gy);
    const int x1 = std::min(x0 + 1, Wf - 1), y1 = std::min(y0 + 1, Hf - 1);
    const float ax = gx - x0, ay = gy - y0;
    const float w00 = (1 - ax) * (1 - ay), w10 = ax * (1 - ay);
    const float w01 = (1 - ax) * ay,       w11 = ax * ay;
    const float* p00 = map + (size_t)C * ((size_t)y0 * Wf + x0);
    const float* p10 = map + (size_t)C * ((size_t)y0 * Wf + x1);
    const float* p01 = map + (size_t)C * ((size_t)y1 * Wf + x0);
    const float* p11 = map + (size_t)C * ((size_t)y1 * Wf + x1);
    for (int j = 0; j < C; ++j)
        dst[j] = w00 * p00[j] + w10 * p10[j] + w01 * p01[j] + w11 * p11[j];
}

double pixal3d_cross_res_agreement(const std::vector<float>& dino_a, int Sa,
                                   const std::vector<float>& dino_b, int Sb,
                                   int grid_res, const CameraParams& cam,
                                   const std::vector<std::array<int, 3>>& coords) {
    if (coords.empty()) return 0.0;
    const int Ha = Sa / PATCH, Hb = Sb / PATCH;
    const float* pa = dino_a.data() + (size_t)D_DINO * N_GLOBAL;
    const float* pb = dino_b.data() + (size_t)D_DINO * N_GLOBAL;
    std::vector<float> va(D_DINO), vb(D_DINO);
    double acc = 0;
    // Every 37th point: the figure is a mean over thousands of samples either way, and the full
    // sweep would cost more than the flow step it is diagnosing.
    size_t n = 0;
    for (size_t t = 0; t < coords.size(); t += 37, ++n) {
        float xa, ya, xb, yb;
        pixal3d_project_cell(grid_res, coords[t][0], coords[t][1], coords[t][2], cam, Sa, xa, ya);
        pixal3d_project_cell(grid_res, coords[t][0], coords[t][1], coords[t][2], cam, Sb, xb, yb);
        sample_bilinear(pa, Ha, Ha, D_DINO, Sa, xa, ya, va.data());
        sample_bilinear(pb, Hb, Hb, D_DINO, Sb, xb, yb, vb.data());
        double d = 0, na = 0, nb = 0;
        for (int c = 0; c < D_DINO; ++c) { d += (double)va[c]*vb[c]; na += (double)va[c]*va[c]; nb += (double)vb[c]*vb[c]; }
        if (na > 0 && nb > 0) acc += d / std::sqrt(na * nb);
    }
    return n ? acc / (double)n : 0.0;
}

// The projection conditioning has no runtime failure mode: a wrong camera, a wrong grid or a
// broken upsampler all produce finite numbers and a plausible-looking mesh silhouette, with the
// damage showing up only as high-frequency noise in the generated surface. These three figures
// separate those causes.
//   - coverage: how many grid cells land inside the frame. The cube's corners legitimately fall
//     outside (the object is inscribed in it), but a badly wrong camera collapses this.
//   - the two halves of a 2048-channel vector: the low-resolution DINOv3 samples and the
//     NAF-upsampled ones describe the SAME points at different detail, so they must share a
//     scale and correlate strongly. A broken upsampler shows up here and nowhere else.
static void proj_stats(const ProjCond& pc, size_t N, const std::vector<float>& pts, int S) {
    if (N == 0) return;
    size_t inside = 0;
    for (size_t t = 0; t < N; ++t)
        if (pts[2*t] >= 0 && pts[2*t] < S && pts[2*t+1] >= 0 && pts[2*t+1] < S) ++inside;

    auto half = [&](int off, double& mean, double& sd) {
        double s = 0, s2 = 0;
        for (size_t t = 0; t < N; ++t)
            for (int c = 0; c < D_DINO; ++c) {
                const double v = pc.proj[(size_t)pc.proj_ch * t + off + c];
                s += v; s2 += v * v;
            }
        const double n = (double)N * D_DINO;
        mean = s / n; sd = std::sqrt(std::max(0.0, s2 / n - mean * mean));
    };
    double lm, ls;
    half(0, lm, ls);
    printf("      [stats] proj: %.0f%% of cells inside frame; lr mean=%.3f std=%.3f",
           100.0 * (double)inside / (double)N, lm, ls);
    if (pc.proj_ch == 2 * D_DINO) {
        double hm, hs;
        half(D_DINO, hm, hs);
        double dot = 0, nl = 0, nh = 0;
        for (size_t t = 0; t < N; ++t) {
            const float* p = &pc.proj[(size_t)pc.proj_ch * t];
            double d = 0, a = 0, b = 0;
            for (int c = 0; c < D_DINO; ++c) {
                d += (double)p[c] * p[D_DINO + c];
                a += (double)p[c] * p[c];
                b += (double)p[D_DINO + c] * p[D_DINO + c];
            }
            dot += d; nl += a; nh += b;
        }
        const double cs = (nl > 0 && nh > 0) ? dot / std::sqrt(nl * nh) : 0.0;
        printf("; hr mean=%.3f std=%.3f; cos(lr,hr)=%.3f", hm, hs, cs);
    }
    printf("\n");
}

ProjCond pixal3d_proj_cond(const std::vector<float>& dino, int S, int grid_res, int proj_ch,
                           const CameraParams& cam,
                           const std::vector<std::array<int, 3>>* coords,
                           const Model* naf, const std::vector<float>* img01, int naf_out) {
    if (proj_ch != D_DINO && proj_ch != 2 * D_DINO)
        throw std::runtime_error("pixal3d: unsupported proj_in_channels");
    const int Hp = S / PATCH;
    const size_t ntok = dino.size() / D_DINO;
    if (ntok != (size_t)N_GLOBAL + (size_t)Hp * Hp)
        throw std::runtime_error("pixal3d: DINOv3 token count does not match the image size");

    ProjCond out;
    out.n_global = N_GLOBAL;
    out.global.assign(dino.begin(), dino.begin() + (size_t)D_DINO * N_GLOBAL);

    // The DINOv3 output is channel-major per token and the patch tokens are row-major over the
    // patch grid, which is exactly the [C, Hf*Wf] layout the samplers want — no repacking, just
    // skip the global tokens.
    const float* patches = dino.data() + (size_t)D_DINO * N_GLOBAL;

    const size_t N = coords ? coords->size() : (size_t)grid_res * grid_res * grid_res;
    std::vector<float> pts(2 * N);
    for (size_t t = 0; t < N; ++t) {
        int cx, cy, cz;
        if (coords) { cx = (*coords)[t][0]; cy = (*coords)[t][1]; cz = (*coords)[t][2]; }
        else {
            const int R = grid_res;
            cx = (int)(t / ((size_t)R * R)); cy = (int)((t / R) % R); cz = (int)(t % R);
        }
        pixal3d_project_cell(grid_res, cx, cy, cz, cam, S, pts[2 * t], pts[2 * t + 1]);
    }

    out.proj_ch = proj_ch;
    out.proj.assign((size_t)proj_ch * N, 0.0f);

    for (size_t t = 0; t < N; ++t)
        sample_bilinear(patches, Hp, Hp, D_DINO, S, pts[2 * t], pts[2 * t + 1],
                        &out.proj[(size_t)proj_ch * t]);

    if (proj_ch == 2 * D_DINO) {
        if (naf) {
            if (!img01) throw std::runtime_error("pixal3d: NAF upsampling needs the raw [0,1] guide");
            std::vector<float> hr = naf_sample(*naf, *img01, S, patches, Hp, Hp, D_DINO, naf_out, pts);
            for (size_t t = 0; t < N; ++t)
                std::copy(hr.begin() + (size_t)D_DINO * t, hr.begin() + (size_t)D_DINO * (t + 1),
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO);
        } else {
            // --no-naf: repeat the low-resolution samples in place of the upsampled branch. The
            // stage runs and stays roughly on-distribution, but every high-frequency cue the
            // second branch was trained to carry is gone.
            for (size_t t = 0; t < N; ++t)
                std::copy(out.proj.begin() + (size_t)proj_ch * t,
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO,
                          out.proj.begin() + (size_t)proj_ch * t + D_DINO);
        }
    }

    proj_stats(out, N, pts, S);
    return out;
}

} // namespace trellis
