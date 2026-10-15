// trellis-tex-start: build the picture-colour start latent for the texture flow (opt-in route 2).
//   trellis-tex-start dump-coords <shape_dec.gguf> <hr_slat.bin> <out_coords.i32> [gpu]
//       final voxel coords of the shape decode (int32 [M,3]) -> tools/tex_project.py input
//   trellis-tex-start encode <tex_enc.gguf> <hr_slat.bin> <coords.i32> <pbr6.f32> <out.f32> [gpu]
//       pbr6 [M,6] in [-1,1] (tools/tex_project.py) -> tex_encode -> remap to the shape slat's token order
//       -> normalise -> statistics match -> out.f32 [N,32], the TRELLIS_TEX_START file for trellis-cli.
// hr_slat.bin is trellis-cli's --dump-slat file (TRELLIS_DUMP_SLAT_PATH).
// TRELLIS_TEX_MATCH = none | scale (default) | full.
//   scale: one scalar so the mean per-channel std over tokens equals TRELLIS_TEX_MATCH_STD (default 1);
//          per-channel means are kept, because the offset carries the global colour (DC) the steering study found.
//   full : per channel to mean 0 / std MATCH_STD (drops the DC colour offset; use only to test that).
#include "trellis_model.h"
#include "shape_decoder.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace trellis;

static const float TEX_MEAN[32]={3.501659f,2.212398f,2.226094f,0.251093f,-0.026248f,-0.687364f,0.439898f,-0.928075f,0.029398f,-0.339596f,-0.869527f,1.038479f,-0.972385f,0.126042f,-1.129303f,0.455149f,-1.209521f,2.069067f,0.544735f,2.569128f,-0.323407f,2.293000f,-1.925608f,-1.217717f,1.213905f,0.971588f,-0.023631f,0.106750f,2.021786f,0.250524f,-0.662387f,-0.768862f};
static const float TEX_STD[32]={2.665652f,2.743913f,2.765121f,2.595319f,3.037293f,2.291316f,2.144656f,2.911822f,2.969419f,2.501689f,2.154811f,3.163343f,2.621215f,2.381943f,3.186697f,3.021588f,2.295916f,3.234985f,3.233086f,2.260140f,2.874801f,2.810596f,3.292720f,2.674999f,2.680878f,2.372054f,2.451546f,2.353556f,2.995195f,2.379849f,2.786195f,2.775190f};

static bool read_slat(const char* path, std::vector<std::array<int,3>>& coords, int& res) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int N = 0;
    bool ok = fread(&N, 4, 1, f) == 1 && fread(&res, 4, 1, f) == 1 && N > 0;
    if (ok) {
        coords.resize(N);
        for (int i = 0; i < N && ok; ++i) { int xyz[3]; ok = fread(xyz, 4, 3, f) == 3; coords[i] = {xyz[0], xyz[1], xyz[2]}; }
    }
    fclose(f);
    return ok;
}

static uint64_t key(const std::array<int,3>& c) { return ((uint64_t)c[0] << 40) | ((uint64_t)c[1] << 20) | (uint64_t)c[2]; }

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "dump-coords" && argc >= 5) {
        const int gpu = argc > 5 ? atoi(argv[5]) : 0;
        std::vector<std::array<int,3>> coords; int res = 0;
        if (!read_slat(argv[3], coords, res)) { fprintf(stderr, "cannot read slat %s\n", argv[3]); return 1; }
        FILE* f = fopen(argv[3], "rb");
        int N = 0, r2 = 0; if (fread(&N, 4, 1, f) != 1 || fread(&r2, 4, 1, f) != 1) return 1;
        fseek(f, 8 + 12L * N, SEEK_SET);
        std::vector<float> feats((size_t)32 * N);
        if (fread(feats.data(), 4, feats.size(), f) != feats.size()) { fprintf(stderr, "short slat file\n"); return 1; }
        fclose(f);
        Model sd = Model::load(argv[2], gpu);
        ShapeOut so = shape_decode(sd, feats, coords, res); sd.free();
        FILE* o = fopen(argv[4], "wb");
        if (!o) { fprintf(stderr, "cannot write %s\n", argv[4]); return 1; }
        for (auto& c : so.coords) { int xyz[3] = { c[0], c[1], c[2] }; fwrite(xyz, 4, 3, o); }
        fclose(o);
        printf("[tex-start] %zu voxels at res %d -> %s\n", so.coords.size(), res, argv[4]);
        return 0;
    }
    if (mode == "encode" && argc >= 7) {
        const int gpu = argc > 7 ? atoi(argv[7]) : 0;
        std::vector<std::array<int,3>> slat_coords; int res = 0;
        if (!read_slat(argv[3], slat_coords, res)) { fprintf(stderr, "cannot read slat %s\n", argv[3]); return 1; }
        const int N = (int)slat_coords.size();
        std::vector<std::array<int,3>> vc;
        { FILE* f = fopen(argv[4], "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", argv[4]); return 1; }
          fseek(f, 0, SEEK_END); const long sz = ftell(f); fseek(f, 0, SEEK_SET);
          vc.resize(sz / 12);
          for (auto& c : vc) { int xyz[3]; if (fread(xyz, 4, 3, f) != 3) return 1; c = {xyz[0], xyz[1], xyz[2]}; }
          fclose(f); }
        const size_t M = vc.size();
        std::vector<float> pbr(6 * M);
        { FILE* f = fopen(argv[5], "rb");
          if (!f || fread(pbr.data(), 4, pbr.size(), f) != pbr.size()) { fprintf(stderr, "pbr6 file must hold %zu*6 floats\n", M); return 1; }
          fclose(f); }
        Model te = Model::load(argv[2], gpu);
        TexEncOut eo = tex_encode(te, pbr, vc); te.free();
        const int P = (int)eo.coords.size();
        std::vector<std::pair<uint64_t,int>> idx(P);
        for (int p = 0; p < P; ++p) idx[p] = { key(eo.coords[p]), p };
        std::sort(idx.begin(), idx.end());
        std::vector<float> z((size_t)32 * N, 0.f);   // normalised latent in the slat's token order
        int miss = 0;
        for (int n = 0; n < N; ++n) {
            auto it = std::lower_bound(idx.begin(), idx.end(), std::make_pair(key(slat_coords[n]), -1));
            if (it == idx.end() || it->first != key(slat_coords[n])) { ++miss; continue; }
            for (int c = 0; c < 32; ++c)
                z[(size_t)c + 32*n] = (eo.mean[(size_t)c + 32*(size_t)it->second] - TEX_MEAN[c]) / TEX_STD[c];
        }
        double cm[32] = {0}, cs[32] = {0};
        for (int c = 0; c < 32; ++c) {
            for (int n = 0; n < N; ++n) cm[c] += z[(size_t)c + 32*n];
            cm[c] /= N;
            for (int n = 0; n < N; ++n) { const double d = z[(size_t)c + 32*n] - cm[c]; cs[c] += d * d; }
            cs[c] = std::sqrt(cs[c] / N);
        }
        double mean_std = 0; for (int c = 0; c < 32; ++c) mean_std += cs[c] / 32;
        const char* mm = getenv("TRELLIS_TEX_MATCH");
        const std::string match = mm ? mm : "scale";
        const double target = getenv("TRELLIS_TEX_MATCH_STD") ? atof(getenv("TRELLIS_TEX_MATCH_STD")) : 1.0;
        printf("[tex-start] encoded %d latent voxels, slat tokens %d (missing %d); before match: mean std %.3f, mean |channel mean| %.3f\n",
               P, N, miss, mean_std, [&]{ double a = 0; for (int c = 0; c < 32; ++c) a += std::fabs(cm[c]) / 32; return a; }());
        if (match == "scale" && mean_std > 1e-6) {
            const double s = target / mean_std;
            for (int c = 0; c < 32; ++c) for (int n = 0; n < N; ++n) z[(size_t)c + 32*n] = (float)(cm[c] + (z[(size_t)c + 32*n] - cm[c]) * s);
            printf("[tex-start] match=scale: deviations x%.3f, channel means kept\n", s);
        } else if (match == "full") {
            for (int c = 0; c < 32; ++c) { const double s = cs[c] > 1e-6 ? target / cs[c] : 1.0;
                for (int n = 0; n < N; ++n) z[(size_t)c + 32*n] = (float)((z[(size_t)c + 32*n] - cm[c]) * s); }
            printf("[tex-start] match=full: per channel to mean 0 / std %.2f\n", target);
        } else printf("[tex-start] match=none\n");
        FILE* o = fopen(argv[6], "wb");
        if (!o) { fprintf(stderr, "cannot write %s\n", argv[6]); return 1; }
        fwrite(z.data(), 4, z.size(), o); fclose(o);
        printf("[tex-start] wrote %s (%d tokens x 32); run trellis-cli with TRELLIS_TEX_START=%s TRELLIS_TEX_T0=0.7\n", argv[6], N, argv[6]);
        return 0;
    }
    fprintf(stderr, "usage:\n  %s dump-coords <shape_dec.gguf> <hr_slat.bin> <out_coords.i32> [gpu]\n"
                    "  %s encode <tex_enc.gguf> <hr_slat.bin> <coords.i32> <pbr6.f32> <out.f32> [gpu]\n", argv[0], argv[0]);
    return 1;
}
