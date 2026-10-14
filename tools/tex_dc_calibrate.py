#!/usr/bin/env python3
"""Calibrate TRELLIS_TEX_STEER: the texture latent's global mean (DC) along one direction predicts the decoded brightness.

Input: draws of one or more pictures made with `TRELLIS_TEX_DRAWS=N TRELLIS_TEX_LATDUMP=<prefix> trellis-cli ...`: the files
<prefix>_000.f32 ... (normalised texture latent, f32 [tokens, 32], channel fastest) and the run log with its
`[tex-draw] i ... decoded_y Y` lines. Writes the calibration file the engine reads:

    d <32 floats, unit vector, larger coordinate = brighter>
    fit <a> <b>          # decoded luma ~ a + b * coordinate, coordinate = mean over tokens of d . latent

Direction = first principal component of the per-draw DC vectors (centred per set, so picture-to-picture differences do not
enter), sign chosen so it correlates positively with luma. Prints the leave-one-out error and the correlation.

  python tools/tex_dc_calibrate.py --set <dir-with-lat_files-and-log>:<lat-prefix> [--set ...] --out cal.txt
"""
import argparse, os, re, sys
import numpy as np


def load_set(spec):
    d, prefix = spec.rsplit(":", 1)
    ys = []
    for name in sorted(os.listdir(d)):
        if name.endswith(".log"):
            for line in open(os.path.join(d, name), errors="ignore"):
                m = re.search(r"\[tex-draw\] (\d+) .*decoded_y ([0-9.]+)", line)
                if m:
                    ys.append((int(m.group(1)), float(m.group(2))))
            if ys:
                break
    ys = [y for _, y in sorted(ys)]
    dcs = []
    for i in range(len(ys)):
        f = os.path.join(d, "%s_%03d.f32" % (prefix, i))
        if not os.path.exists(f):
            break
        dcs.append(np.fromfile(f, dtype="<f4").reshape(-1, 32).mean(0))
    n = min(len(dcs), len(ys))
    return np.array(dcs[:n], dtype=np.float64), np.array(ys[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sets = [load_set(s) for s in a.set]
    cen = np.concatenate([dc - dc.mean(0) for dc, _ in sets])
    _, S, Vt = np.linalg.svd(cen, full_matrices=False)
    d = Vt[0]
    allc = np.concatenate([dc for dc, _ in sets]); ally = np.concatenate([y for _, y in sets])
    c = allc @ d
    if np.corrcoef(c, ally)[0, 1] < 0:
        d = -d; c = -c
    b, a0 = np.polyfit(c, ally, 1)
    errs = []
    for i in range(len(c)):
        m = np.ones(len(c), bool); m[i] = False
        bb, aa = np.polyfit(c[m], ally[m], 1)
        errs.append(aa + bb * c[i] - ally[i])
    print("draws %d  corr %.3f  fit luma = %.4f + %.4f * c   leave-one-out rmse %.3f (luma sd %.3f)   PC1 share %.2f"
          % (len(c), np.corrcoef(c, ally)[0, 1], a0, b, float(np.sqrt(np.mean(np.square(errs)))), ally.std(), (S[0] ** 2 / (S ** 2).sum())))
    with open(a.out, "w") as f:
        f.write("d " + " ".join("%.7f" % x for x in d) + "\n")
        f.write("fit %.6f %.6f\n" % (a0, b))
    print("wrote", a.out)


if __name__ == "__main__":
    sys.exit(main())
