// trellis-test-tex-enc: reconstruction test for the texture VAE encoder port.
//   trellis-test-tex-enc <shape_dec.gguf> <tex_dec.gguf> <tex_enc.gguf> <hr_slat.bin> <tex_lat.f32> [gpu]
// hr_slat.bin: trellis-cli --dump-slat (set TRELLIS_DUMP_SLAT_PATH); tex_lat.f32: TRELLIS_TEX_LATDUMP file
// (normalised texture latent [N,32], same voxel order as the HR slat). Steps: shape_decode -> subs;
// tex_decode(latent) = reference PBR voxels; tex_encode(PBR) -> posterior mean; tex_decode(mean) with
// the same subs; report error and latent statistics. TRELLIS_S2C_PERM picks the octant order.
#include "trellis_model.h"
#include "shape_decoder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace trellis;

static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};

// error between two [6*M] voxel sets after the engine's clamp to [0,1]: mean abs error per channel, all-channel PSNR
static void report(const char* tag, const std::vector<float>& a, const std::vector<float>& b) {
    const size_t M = a.size() / 6;
    double sa[6] = {0}, se[6] = {0};
    auto cl = [](float v){ v = v * 0.5f + 0.5f; return v < 0 ? 0.f : (v > 1 ? 1.f : v); };
    for (size_t i = 0; i < M; ++i)
        for (int k = 0; k < 6; ++k) {
            const double d = (double)cl(a[6*i + k]) - cl(b[6*i + k]);
            sa[k] += std::fabs(d); se[k] += d * d;
        }
    double tot_a = 0, tot_e = 0;
    printf("  %s\n   per-channel MAE:", tag);
    for (int k = 0; k < 6; ++k) { printf(" %.4f", sa[k] / M); tot_a += sa[k]; tot_e += se[k]; }
    const double mse = tot_e / (6.0 * M);
    printf("\n   all MAE %.4f  PSNR %.2f dB  (voxels %zu)\n", tot_a / (6.0 * M), 10 * std::log10(1.0 / std::max(mse, 1e-12)), M);
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s shape_dec tex_dec tex_enc hr_slat.bin tex_lat.f32 [gpu]\n", argv[0]); return 1; }
    const int gpu = argc > 6 ? atoi(argv[6]) : 0;
    FILE* f = fopen(argv[4], "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[4]); return 1; }
    int N = 0, res = 0;
    if (fread(&N, 4, 1, f) != 1 || fread(&res, 4, 1, f) != 1) return 1;
    std::vector<std::array<int,3>> coords(N);
    for (int i = 0; i < N; ++i) { int xyz[3]; if (fread(xyz, 4, 3, f) != 3) return 1; coords[i] = {xyz[0], xyz[1], xyz[2]}; }
    std::vector<float> feats((size_t)32 * N);
    if (fread(feats.data(), 4, feats.size(), f) != feats.size()) return 1;
    fclose(f);
    std::vector<float> lat((size_t)32 * N);
    f = fopen(argv[5], "rb");
    if (!f || fread(lat.data(), 4, lat.size(), f) != lat.size()) { fprintf(stderr, "bad latent file (need %d*32 floats)\n", N); return 1; }
    fclose(f);
    for (int n = 0; n < N; ++n) for (int c = 0; c < 32; ++c) lat[(size_t)c + 32*n] = lat[(size_t)c + 32*n] * TEX_STD[c] + TEX_MEAN[c];
    printf("[test] N=%d res=%d perm=%s\n", N, res, getenv("TRELLIS_S2C_PERM") ? getenv("TRELLIS_S2C_PERM") : "xyz"); fflush(stdout);

    Model sd = Model::load(argv[1], gpu);
    ShapeOut so = shape_decode(sd, feats, coords, res); sd.free();
    printf("[test] shape decoded: %zu voxels\n", so.coords.size()); fflush(stdout);
    Model td = Model::load(argv[2], gpu);
    std::vector<float> pbr = tex_decode(td, lat, coords, so.subs);
    printf("[test] reference PBR from the flow latent: %zu voxels\n", pbr.size() / 6); fflush(stdout);

    for (int variant = 0; variant < 2; ++variant) {
        std::vector<float> in = pbr;
        if (variant == 1) for (float& v : in) { float u = v * 0.5f + 0.5f; u = u < 0 ? 0 : (u > 1 ? 1 : u); v = u * 2 - 1; }
        Model te = Model::load(argv[3], gpu);
        TexEncOut eo = tex_encode(te, in, so.coords); te.free();
        const int P = (int)eo.coords.size();
        printf("[test] %s input: encoded to %d latent voxels (shape slat had %d)\n", variant ? "clamped" : "raw", P, N);
        printf("  latent (normalised) per-channel mean/std:\n   ");
        double gm = 0, gs = 0;
        for (int c = 0; c < 32; ++c) {
            double m = 0, s = 0;
            for (int n = 0; n < P; ++n) m += (eo.mean[(size_t)c + 32*n] - TEX_MEAN[c]) / TEX_STD[c];
            m /= P;
            for (int n = 0; n < P; ++n) { const double z = (eo.mean[(size_t)c + 32*n] - TEX_MEAN[c]) / TEX_STD[c] - m; s += z * z; }
            s = std::sqrt(s / P);
            gm += std::fabs(m); gs += s;
            printf(" %+.2f/%.2f", m, s);
            if (c % 8 == 7) printf("\n   ");
        }
        printf("\n  mean |channel mean| %.3f, mean channel std %.3f (flow outputs: ~0 and ~1)\n", gm / 32, gs / 32);
        // The decoder wants the shape slat's order; look each slat coord up in the encoder's (sorted) coord set.
        std::vector<float> latmap((size_t)32 * N, 0.f);
        {
            auto key = [](const std::array<int,3>& c){ return ((uint64_t)c[0] << 40) | ((uint64_t)c[1] << 20) | (uint64_t)c[2]; };
            std::vector<std::pair<uint64_t,int>> idx(P);
            for (int p = 0; p < P; ++p) idx[p] = { key(eo.coords[p]), p };
            std::sort(idx.begin(), idx.end());
            int miss = 0;
            for (int n = 0; n < N; ++n) {
                auto it = std::lower_bound(idx.begin(), idx.end(), std::make_pair(key(coords[n]), -1));
                if (it == idx.end() || it->first != key(coords[n])) { ++miss; continue; }
                std::copy_n(eo.mean.data() + (size_t)32 * it->second, 32, latmap.data() + (size_t)32 * n);
            }
            printf("  slat coords missing from the encoder grid: %d of %d\n", miss, N);
        }
        double dl = 0, ll = 0;
        for (size_t i = 0; i < latmap.size(); ++i) { dl += std::fabs(latmap[i] - lat[i]); ll += std::fabs(lat[i]); }
        printf("  latent MAE vs the flow latent: %.3f (flow latent mean |v| %.3f)\n", dl / latmap.size(), ll / latmap.size());
        std::vector<float> pbr2 = tex_decode(td, latmap, coords, so.subs);
        report(variant ? "recon vs reference (clamped input)" : "recon vs reference (raw input)", pbr, pbr2);
        fflush(stdout);
    }
    td.free();
    return 0;
}
