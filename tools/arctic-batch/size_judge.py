#!/usr/bin/env python3
"""Size judge for draft_refine (stdlib): reads a GLB's POSITION bounds and compares the SHAPE of the bounding box with a spec.

  python size_judge.py draft.glb --spec 1.3,0.8,0.85 [--veto 0.35]
  python size_judge.py draft.glb --spec-file specs.json --prop chest        (specs.json: {"chest": [1.3, 0.8, 0.85], ...})

Axis order and units do not matter: both boxes are sorted and divided by their largest side, then the score is minus the worst
relative deviation of the two smaller sides (0 = same proportions). Prints the score on the last line; exits 4 (veto) when the
deviation is above --veto. A screen for size and proportion only: it says nothing about look.
"""
import argparse, json, struct, sys


def glb_bounds(path):
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[:4] != b"glTF":
            raise ValueError("not a GLB")
        n, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("first chunk is not JSON")
        doc = json.loads(f.read(n).decode("utf-8"))
    lo, hi = [1e30] * 3, [-1e30] * 3
    got = False
    for m in doc.get("meshes", []):
        for prim in m.get("primitives", []):
            acc = doc["accessors"][prim["attributes"]["POSITION"]]
            if "min" in acc and "max" in acc:
                got = True
                lo = [min(a, b) for a, b in zip(lo, acc["min"])]
                hi = [max(a, b) for a, b in zip(hi, acc["max"])]
    if not got:
        raise ValueError("no POSITION min/max in the GLB")
    return [h - l for l, h in zip(lo, hi)]


def deviation(dims, spec):
    a, b = sorted(dims, reverse=True), sorted(spec, reverse=True)
    if a[0] <= 0 or b[0] <= 0:
        raise ValueError("empty box")
    ra, rb = [x / a[0] for x in a[1:]], [x / b[0] for x in b[1:]]
    return max(abs(x - y) / y for x, y in zip(ra, rb))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("glb"); p.add_argument("--spec", default="", help="W,D,H in any units and order")
    p.add_argument("--spec-file", default=""); p.add_argument("--prop", default="")
    p.add_argument("--veto", type=float, default=0.35)
    a = p.parse_args(argv)
    if a.spec_file:
        with open(a.spec_file, encoding="utf-8") as f:
            spec = [float(x) for x in json.load(f)[a.prop]]
    elif a.spec:
        spec = [float(x) for x in a.spec.split(",")]
    else:
        p.error("give --spec or --spec-file with --prop")
    if len(spec) != 3:
        p.error("--spec needs three numbers")
    dev = deviation(glb_bounds(a.glb), spec)
    print("deviation %.4f (veto above %.2f)" % (dev, a.veto))
    if dev > a.veto:
        return 4
    print("%.6f" % -dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
