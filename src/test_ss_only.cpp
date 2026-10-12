// CPU-only checks for `trellis-cli --ss-only`: the bounding-box helper, the JSON writer and the flag parsing.
// No models, no GPU: build `trellis-test-ss-only` and run it.
#include "ss_bbox.h"
#include "trellis_args.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

static int bad = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); ++bad; } } while (0)

static bool parse(std::vector<std::string> args, trellis::TrellisParams& p) {
    std::vector<char*> argv;
    static std::string prog = "trellis-cli";
    argv.push_back(&prog[0]);
    for (auto& a : args) argv.push_back(&a[0]);
    return trellis::parse_args((int)argv.size(), argv.data(), p);
}

int main() {
    using trellis::SsBox;

    // bounding box
    SsBox e = trellis::ss_bbox({});
    CHECK(e.voxels == 0 && e.extent[0] == 0 && e.extent[1] == 0 && e.extent[2] == 0);
    SsBox one = trellis::ss_bbox({{5, 7, 9}});
    CHECK(one.voxels == 1 && one.extent[0] == 1 && one.extent[1] == 1 && one.extent[2] == 1 && one.mn[2] == 9);
    SsBox b = trellis::ss_bbox({{2, 10, 0}, {30, 12, 31}, {16, 11, 5}, {2, 10, 0}});   // duplicates are counted as given
    CHECK(b.voxels == 4);
    CHECK(b.mn[0] == 2 && b.mx[0] == 30 && b.extent[0] == 29);
    CHECK(b.mn[1] == 10 && b.mx[1] == 12 && b.extent[1] == 3);
    CHECK(b.mn[2] == 0 && b.mx[2] == 31 && b.extent[2] == 32);

    // json
    CHECK(trellis::json_escape("C:\\a\\\"b\"\n") == "C:\\\\a\\\\\\\"b\\\"\\n");
    std::string j = trellis::ss_json("C:\\img\\w.png", "trellis", 24, 7.5f, 32, {{1u, b}, {2u, one}});
    CHECK(j.find("\"image\":\"C:\\\\img\\\\w.png\"") != std::string::npos);
    CHECK(j.find("\"steps\":24") != std::string::npos && j.find("\"gss\":7.5") != std::string::npos);
    CHECK(j.find("{\"seed\":1,\"voxels\":4,\"min\":[2,10,0],\"max\":[30,12,31],\"extent\":[29,3,32]}") != std::string::npos);
    CHECK(j.find("{\"seed\":2,\"voxels\":1,\"min\":[5,7,9],\"max\":[5,7,9],\"extent\":[1,1,1]}") != std::string::npos);
    CHECK(j.back() == '\n');
    int depth = 0, mx = 0;
    for (char c : j) { if (c == '{' || c == '[') mx = std::max(mx, ++depth); else if (c == '}' || c == ']') --depth; }
    CHECK(depth == 0 && mx >= 3);

    // flags
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb"}, p)); CHECK(!p.ss_only && p.ss_seeds == 1); }
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb", "--ss-only"}, p)); CHECK(p.ss_only && p.ss_seeds == 1); }
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb", "--ss-only", "--ss-seeds", "16", "--seed", "3", "--steps", "24"}, p));
      CHECK(p.ss_only && p.ss_seeds == 16 && p.seed == 3 && p.steps == 24); }
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb", "--ss-seeds", "4", "--ss-only"}, p) && p.ss_seeds == 4); }   // order does not matter
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb"}, p)); CHECK(p.steps_ss == 0); }
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb", "--steps", "12", "--steps-ss", "24"}, p)); CHECK(p.steps == 12 && p.steps_ss == 24); }
    { trellis::TrellisParams p; CHECK(parse({"a.png", "o.glb", "--steps-ss", "24", "--steps", "12"}, p)); CHECK(p.steps == 12 && p.steps_ss == 24); }
    { trellis::TrellisParams p; CHECK(!parse({"a.png", "o.glb", "--steps-ss"}, p)); }
    for (const char* v : {"0", "-1", "4097", "x", "2x", ""}) {
        trellis::TrellisParams p; CHECK(!parse({"a.png", "o.glb", "--ss-only", "--ss-seeds", v}, p));
    }
    { trellis::TrellisParams p; CHECK(!parse({"a.png", "o.glb", "--ss-seeds", "4"}, p)); }   // needs --ss-only
    { trellis::TrellisParams p; CHECK(!parse({"a.png", "o.glb", "--ss-only", "--ss-seeds"}, p)); }

    std::printf(bad ? "trellis-test-ss-only: %d FAILED\n" : "trellis-test-ss-only: all passed\n", bad);
    return bad ? 1 : 0;
}
