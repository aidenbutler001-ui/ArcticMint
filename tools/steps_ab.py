#!/usr/bin/env python3
"""A/B the flow step count: arm A = --steps 24 (standing rule), arm B = --steps 12 --steps-ss 24.

For each piece and seed it runs trellis-cli once per arm, then records wall time, the
per-stage table from --stage-timing and a colour check: brightness of the VISIBLE surface of the
output GLB against the source picture's foreground, using the same ratio and 0.6..1/0.6 window as
the engine's --tex-reroll check (the dark-brown failure). Up to ~73% of the atlas can be hidden
interior faces, so the atlas mean is not used: faces are kept only if some outside view sees them
(point-splat depth test over 26 directions) and weighted by area. Nothing here changes the engine's
texture stage; the check reads the finished GLB, so --tex-reroll is left off in both arms.

This script launches trellis-cli. On the PC run it only from the queue, under the gpulock tooling
and after the usual go. Use --prefix to wrap each command (for example the gpulock launcher).

  python3 tools/steps_ab.py --cli C:/AI/ArcticMint/build-full/trellis-cli.exe \
      --images C:/AI/Expiramental3d/phase0/images --out ab_out --dry-run
"""
import argparse
import csv
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PIECES = ["SM_ChoirStall", "SM_StormStump", "SM_CargoStack", "SM_CrackedAltar", "SM_ClockworkDrive"]
ARMS = {"A": ["--steps", "24"], "B": ["--steps", "12", "--steps-ss", "24"]}
COMMON = ["--res", "1024", "--faces", "150000", "--atlas", "4096"]
EPS, LO = 0.02, 0.6   # same offset and window as the engine's tex-reroll check


def luminance(r, g, b):
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def mean_luma(path):
    """Alpha-weighted mean luminance (0..1) of a foreground picture (texels with alpha >= 128)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    im = Image.open(path).convert("RGBA")
    y = w = 0.0
    for r, g, b, a in im.getdata():
        if a < 128:
            continue
        y += (a / 255.0) * luminance(r, g, b) / 255.0
        w += a / 255.0
    return y / w if w else None


def load_glb(path):
    """First primitive of a GLB: positions, uvs, triangle indices, base colour texture (H,W,3 uint8)."""
    import io
    import struct
    import numpy as np
    from PIL import Image
    d = Path(path).read_bytes()
    if d[:4] != b"glTF":
        raise ValueError("not a GLB file")
    jl = struct.unpack_from("<I", d, 12)[0]
    J = json.loads(d[20:20 + jl])
    p = 20 + jl
    bl = struct.unpack_from("<I", d, p)[0]
    BIN = d[p + 8:p + 8 + bl]

    def acc(i):
        a = J["accessors"][i]
        bv = J["bufferViews"][a["bufferView"]]
        dt = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]]
        nc = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}[a["type"]]
        off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(BIN, dtype=dt, count=a["count"] * nc, offset=off).reshape(a["count"], nc)

    prim = J["meshes"][0]["primitives"][0]
    P = acc(prim["attributes"]["POSITION"]).astype(np.float64)
    UV = acc(prim["attributes"]["TEXCOORD_0"]).astype(np.float64)
    I = acc(prim["indices"]).reshape(-1, 3).astype(np.int64)
    bv = J["bufferViews"][J["images"][0]["bufferView"]]
    o = bv.get("byteOffset", 0)
    tex = np.array(Image.open(io.BytesIO(BIN[o:o + bv["byteLength"]])).convert("RGB"))
    return P, UV, I, tex


def _view_dirs():
    import numpy as np
    v = [(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
    v = np.array(v, float)
    return v / np.linalg.norm(v, axis=1, keepdims=True)   # 6 axes + 12 edges + 8 corners


def visible_faces(P, I, res=None, tol_cells=2.0):
    """Boolean per face: seen from outside in at least one of 26 directions.

    Face centroids are splatted into a res x res depth buffer per direction; a face is visible if its
    depth is within tol_cells pixels of the nearest depth in its 3x3 neighbourhood. Interior faces are
    always behind an outer shell, so they never pass."""
    import numpy as np
    C = P[I].mean(axis=1)
    vis = np.zeros(len(I), bool)
    if res is None:   # a few centroids per pixel on the outer shell, so the depth buffer has no holes
        res = int(min(512, max(32, math.sqrt(len(I)) / 3.0)))
    for d in _view_dirs():
        a = np.cross(d, (0, 0, 1) if abs(d[2]) < 0.9 else (1, 0, 0))
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        x, y, z = C @ a, C @ b, C @ d          # larger z = nearer the camera
        ext = max(np.ptp(x), np.ptp(y), 1e-9)
        cell = ext / (res - 1)
        ix = np.clip(((x - x.min()) / cell).astype(np.int64), 0, res - 1)
        iy = np.clip(((y - y.min()) / cell).astype(np.int64), 0, res - 1)
        flat = np.full((res + 2) * (res + 2), -np.inf)
        np.maximum.at(flat, (iy + 1) * (res + 2) + (ix + 1), z)
        zb = flat.reshape(res + 2, res + 2)
        near = zb.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near = np.maximum(near, np.roll(np.roll(zb, dy, 0), dx, 1))
        vis |= z >= near[iy + 1, ix + 1] - tol_cells * cell
    return vis


def visible_luma_arrays(P, UV, I, tex):
    """Area-weighted mean luminance (0..1) of the base texture over outside-visible faces."""
    import numpy as np
    vis = visible_faces(P, I)
    if not vis.any():
        return None
    Iv = I[vis]
    area = 0.5 * np.linalg.norm(np.cross(P[Iv[:, 1]] - P[Iv[:, 0]], P[Iv[:, 2]] - P[Iv[:, 0]]), axis=1)
    uv = UV[Iv].mean(axis=1)
    h, w = tex.shape[:2]
    px = tex[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    lum = (0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]) / 255.0
    return float((lum * area).sum() / area.sum()) if area.sum() > 0 else None


def visible_luma(glb):
    """Visible-surface brightness of a GLB's base texture; None if numpy/PIL or the GLB can't be read."""
    try:
        return visible_luma_arrays(*load_glb(glb))
    except (ImportError, OSError, ValueError, KeyError):
        return None


def colour_ratio(pic, tex):
    if pic is None or tex is None:
        return None
    return (tex + EPS) / (pic + EPS)


def in_window(ratio):
    return ratio is not None and LO <= ratio <= 1.0 / LO


STAGE_RE = re.compile(r"^\[timing\]\s{3}(\S+)\s+([\d.]+) s")


def parse_stages(log):
    return {m.group(1): float(m.group(2)) for m in map(STAGE_RE.match, log.splitlines()) if m}


def has_stage_timing(cli):
    """--stage-timing arrived in a later build; only pass it when --help lists it."""
    try:
        p = subprocess.run([cli, "--help"], capture_output=True, text=True, timeout=60)
        return "--stage-timing" in (p.stdout + p.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_cmd(cli, prefix, image, out_glb, seed, arm, extra, stage_timing=True):
    return (shlex.split(prefix) + [cli, str(image), str(out_glb), "--seed", str(seed)] + COMMON
            + (["--stage-timing"] if stage_timing else []) + ARMS[arm] + extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", required=True, help="path to trellis-cli")
    ap.add_argument("--images", required=True, help="folder with one source picture per piece")
    ap.add_argument("--image-pattern", default="{name}.png", help="picture file name per piece (default %(default)s)")
    ap.add_argument("--out", required=True, help="output folder (arm/piece/seed subfolders)")
    ap.add_argument("--pieces", nargs="+", default=PIECES)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="seeds; more seeds give a failure rate (default 42)")
    ap.add_argument("--prefix", default="", help="command prefix, e.g. the gpulock launcher")
    ap.add_argument("--extra", default="", help="extra trellis-cli arguments for both arms")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    a = ap.parse_args(argv)

    out = Path(a.out)
    rows = []
    stage_timing = True if a.dry_run else has_stage_timing(a.cli)
    if not stage_timing:
        print("note: this trellis-cli has no --stage-timing; per-stage columns will be empty", file=sys.stderr)
    for name in a.pieces:
        image = Path(a.images) / a.image_pattern.format(name=name)
        for seed in a.seeds:
            for arm in ARMS:
                glb = out / arm / name / f"s{seed}" / f"{name}.glb"
                cmd = build_cmd(a.cli, a.prefix, image, glb, seed, arm, shlex.split(a.extra), stage_timing)
                if a.dry_run:
                    print(" ".join(shlex.quote(c) for c in cmd))
                    continue
                if not image.exists():
                    sys.exit(f"missing source picture: {image}")
                glb.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                p = subprocess.run(cmd, capture_output=True, text=True)
                wall = time.time() - t0
                (glb.parent / "run.log").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr)
                pic = mean_luma(image)
                bright = visible_luma(glb) if glb.exists() else None
                ratio = colour_ratio(pic, bright)
                row = {"piece": name, "seed": seed, "arm": arm, "rc": p.returncode, "wall_s": round(wall, 1),
                       "pic_luma": pic, "tex_luma": bright, "ratio": ratio, "colour_ok": in_window(ratio),
                       "glb": str(glb), "stages": parse_stages(p.stdout)}
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "stages"}), flush=True)
    if a.dry_run:
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(rows, indent=1))
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["piece", "seed", "arm", "rc", "wall_s", "ratio", "colour_ok"])
        for r in rows:
            w.writerow([r[k] for k in ("piece", "seed", "arm", "rc", "wall_s", "ratio", "colour_ok")])
    summarize(rows)
    return 0


def summarize(rows):
    print("\narm  runs  mean wall (s)  colour in window")
    for arm in ARMS:
        rs = [r for r in rows if r["arm"] == arm and r["rc"] == 0]
        if not rs:
            continue
        ok = [r for r in rs if r["ratio"] is not None]
        print(f"{arm}    {len(rs):<4}  {sum(r['wall_s'] for r in rs)/len(rs):>12.1f}  "
              f"{sum(1 for r in ok if r['colour_ok'])}/{len(ok)}")
    print("\nCompare the two arms' GLBs with tools/glb_metrics.py and by eye (renders); this script does not judge shape quality.")


if __name__ == "__main__":
    sys.exit(main())
