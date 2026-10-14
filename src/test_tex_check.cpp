#include "tex_check.h"
#include <iostream>
#include <stdexcept>
using namespace trellis;
static void check(bool b, const char* msg) { if (!b) throw std::runtime_error(msg); }
int main() { try {
    // Hollow 5x5x5 box: the outer shell is exposed; one inner voxel floating in the cavity is not.
    std::vector<std::array<int,3>> v;
    for (int x = 0; x < 5; ++x) for (int y = 0; y < 5; ++y) for (int z = 0; z < 5; ++z)
        if (x == 0 || x == 4 || y == 0 || y == 4 || z == 0 || z == 4) v.push_back({x, y, z});
    const size_t shell = v.size();
    v.push_back({2, 2, 2});   // enclosed cavity voxel
    v.push_back({1, 1, 1});   // inner shell, one step inside the corner
    auto vis = exposed_voxels(v);
    check(vis.size() == v.size(), "Wrong size");
    for (size_t i = 0; i < shell; ++i) check(vis[i], "Outer shell voxel not exposed");
    check(!vis[shell], "Enclosed cavity voxel counted as exposed");
    check(!vis[shell + 1], "Inner shell voxel counted as exposed");
    // Open top: the cavity floor is seen from above (it is the lowest voxel only if the lid is missing).
    std::vector<std::array<int,3>> cup;
    for (int x = 0; x < 5; ++x) for (int y = 0; y < 5; ++y) { cup.push_back({x, y, 0}); if (x == 0 || x == 4 || y == 0 || y == 4) cup.push_back({x, y, 3}); }
    auto cv = exposed_voxels(cup);
    for (size_t i = 0; i < cup.size(); ++i) check(cv[i], "Cup voxel hidden");
    // Empty input and negative coordinates.
    check(exposed_voxels({}).empty(), "Empty input");
    auto nv = exposed_voxels({{-3, 0, 0}, {-1, 0, 0}, {2, 0, 0}});
    check(nv[0] && nv[1] && nv[2], "Negative coordinates");
    std::cout << "PASS exposed voxels: hollow shell, enclosed cavity, open cup, empty, negative coords\n";
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
