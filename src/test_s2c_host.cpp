// CPU-only checks of the S2C host helpers (no model, no ggml).
#include "s2c_host.h"
#include <cmath>
#include <cstdio>
#include <random>
using namespace trellis;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    // random sparse fine coords in a 16^3 grid
    std::mt19937 rng(7);
    std::vector<std::array<int,3>> co;
    for (int x = 0; x < 16; ++x) for (int y = 0; y < 16; ++y) for (int z = 0; z < 16; ++z)
        if (rng() % 5 == 0) co.push_back({x, y, z});
    const int N = (int)co.size();
    S2CMap mp = build_s2c_map(co);
    const int P = (int)mp.parents.size();
    CHECK(P > 0 && P <= N);
    // parents sorted and unique
    for (int p = 1; p < P; ++p)
        CHECK(s2c_key(mp.parents[p-1][0], mp.parents[p-1][1], mp.parents[p-1][2]) <
              s2c_key(mp.parents[p][0], mp.parents[p][1], mp.parents[p][2]));
    // each child maps to its parent with the right octant
    for (int i = 0; i < N; ++i) {
        auto& pa = mp.parents[mp.child_parent[i]];
        int o = mp.child_oct[i];
        CHECK(pa[0]*2 + (o&1) == co[i][0] && pa[1]*2 + ((o>>1)&1) == co[i][1] && pa[2]*2 + ((o>>2)&1) == co[i][2]);
    }
    // stack: round trip against the C2S gather convention (channel = o*K + k)
    const int K = 3;
    std::vector<float> h((size_t)K * N);
    for (auto& v : h) v = (float)(rng() % 1000) / 100.f;
    auto st = s2c_stack(h, K, mp);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k)
            CHECK(st[(size_t)8*K*mp.child_parent[i] + mp.child_oct[i]*K + k] == h[(size_t)K*i + k]);
    // skip: mean over groups, total mass preserved (sum over Cout channels = sum(x)/G)
    const int Cin = 8, Cout = 16;
    std::vector<float> x((size_t)Cin * N);
    for (auto& v : x) v = (float)(rng() % 1000) / 100.f;
    auto sk = s2c_skip(x, Cin, Cout, mp);
    double sx = 0, ss = 0; for (float v : x) sx += v; for (float v : sk) ss += v;
    CHECK(std::fabs(ss - sx / (8.0 * Cin / Cout)) < 1e-3 * sx);
    // skip against the literal definition: stack then reshape [Cout,G] and mean
    auto stx = s2c_stack(x, Cin, mp);
    const int G = 8 * Cin / Cout;
    for (int p = 0; p < P; ++p)
        for (int co2 = 0; co2 < Cout; ++co2) {
            double a = 0; for (int g = 0; g < G; ++g) a += stx[(size_t)8*Cin*p + co2*G + g];
            CHECK(std::fabs(a / G - sk[(size_t)Cout*p + co2]) < 1e-4);
        }
    printf(fails ? "s2c host: %d FAILED\n" : "s2c host: OK (N=%d P=%d)\n", fails ? fails : N, P);
    return fails != 0;
}
