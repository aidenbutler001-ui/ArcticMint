#!/usr/bin/env python3
"""A/B the flow step count: arm A = --steps 24 (standing rule), arm B = --steps 12 --steps-ss 24.

For each piece and seed it runs trellis-cli once per arm, then records wall time, the
per-stage table from --stage-timing, mesh/UV metrics (tools/glb_metrics.py) and a colour check:
visible-surface brightness of the baked base texture against the source picture's foreground,
using the same ratio and 0.6..1/0.6 window as the engine's --tex-reroll check (the dark-brown
failure). Nothing here changes the engine's texture stage; the colour check is read from the
output _base.png after the run, so --tex-reroll is left off in both arms.

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
    """Alpha-weighted mean luminance (0..1) over texels with alpha >= 128; None if unreadable."""
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
                tex = glb.with_name(glb.stem + "_base.png")
                pic, bright = mean_luma(image), mean_luma(tex) if tex.exists() else None
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
