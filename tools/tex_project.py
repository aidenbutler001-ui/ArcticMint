#!/usr/bin/env python3
"""Project the source picture's colours onto the shape's voxels (route 2, step 1).

  python tools/tex_project.py --coords coords.i32 --image cutout.png --out pbr6.f32 [--res 1024] [--view auto]

coords.i32: int32 [M,3] from `trellis-tex-start dump-coords` (the final voxel coords). image: RGBA cutout
(alpha = foreground) or RGB (then everything non-white counts). Writes float32 [M,6] in [-1,1] in the
decoder's own channel order and range (base colour RGB, metallic, roughness, alpha), ready for
`trellis-tex-start encode`.

Orthographic only (no camera fit): the voxel set is projected along one axis, scaled to fit the cutout's
bounding box, and a voxel takes the picture colour if it is the front-most in its pixel column (within
--depth-tol voxels). Hidden voxels take the colour of the nearest visible voxel. --view auto picks the
axis permutation/signs (48 candidates) whose projected silhouette best matches the alpha (IoU) and prints
the top candidates; pass an explicit --view 'u,v,d' such as '+x,+z,-y' (image right, image up, away from
the viewer) once known. Metallic/roughness/alpha use neutral values (the picture says nothing about them).
numpy + Pillow only.
"""
import argparse
import itertools
import sys

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


def parse_view(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise ValueError("view must be 'u,v,d' like +x,+z,-y")
    out = []
    for p in parts:
        sign = -1 if p[0] == "-" else 1
        out.append((AXES[p.lstrip("+-")], sign))
    if len({a for a, _ in out}) != 3:
        raise ValueError("view axes must be a permutation of x,y,z")
    return out


def view_name(v):
    return ",".join(("-" if s < 0 else "+") + "xyz"[a] for a, s in v)


def project(coords, view, W, H, bbox):
    """Fit the voxel extent into bbox=(x0,y0,x1,y1) (pixels, aspect kept, centred); return (px, py, depth)."""
    u = coords[:, view[0][0]].astype(np.float64) * view[0][1]
    v = coords[:, view[1][0]].astype(np.float64) * view[1][1]
    d = coords[:, view[2][0]].astype(np.float64) * view[2][1]
    u0, u1, v0, v1 = u.min(), u.max() + 1, v.min(), v.max() + 1
    x0, y0, x1, y1 = bbox
    sc = min((x1 - x0) / (u1 - u0), (y1 - y0) / (v1 - v0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    px = cx + (u + 0.5 - (u0 + u1) / 2) * sc
    py = cy - (v + 0.5 - (v0 + v1) / 2) * sc          # image y grows downwards, "up" axis grows upwards
    return px, py, d, sc


def silhouette_iou(px, py, sc, alpha, W, H):
    r = max(1, int(np.ceil(sc / 2)))                   # splat each voxel over its footprint
    m = np.zeros((H, W), bool)
    ix, iy = np.round(px).astype(int), np.round(py).astype(int)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            xx, yy = ix + dx, iy + dy
            ok = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
            m[yy[ok], xx[ok]] = True
    inter = (m & alpha).sum()
    union = (m | alpha).sum()
    return inter / max(union, 1)


def load_image(path):
    from PIL import Image
    im = Image.open(path)
    if im.mode == "RGBA":
        a = np.asarray(im)[..., 3] > 127
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    else:
        rgb = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        a = rgb.min(axis=2) < 0.97
    return rgb, a


def nearest_fill(coords, known):
    """Index of the nearest known voxel for every voxel (chunked brute force on a subsample grid)."""
    kidx = np.flatnonzero(known)
    out = np.arange(len(coords))
    unk = np.flatnonzero(~known)
    if len(unk) == 0 or len(kidx) == 0:
        return out
    kc = coords[kidx].astype(np.float32)
    # Hidden voxels are searched per 4x4x4 cell (the cell's first voxel stands for it): a brute-force search per
    # voxel is hours at 2M voxels, and hidden colours only need to be locally smooth.
    cell = coords[unk] >> 2
    key = (cell[:, 0].astype(np.int64) << 40) | (cell[:, 1].astype(np.int64) << 20) | cell[:, 2].astype(np.int64)
    _, first, inv = np.unique(key, return_index=True, return_inverse=True)
    res = np.empty(len(first), np.int64)
    for s in range(0, len(first), 2048):
        q = coords[unk[first[s:s + 2048]]].astype(np.float32)
        d2 = ((q[:, None, :] - kc[None, :, :]) ** 2).sum(-1)
        res[s:s + 2048] = kidx[d2.argmin(1)]
    out[unk] = res[inv.ravel()]
    return out


def project_colours(coords, rgb, alpha, view, depth_tol=2.0, max_known_for_fill=40000, rng=None):
    H, W = alpha.shape
    ys, xs = np.nonzero(alpha)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    px, py, d, sc = project(coords, view, W, H, bbox)
    ix = np.clip(np.floor(px).astype(int), 0, W - 1)
    iy = np.clip(np.floor(py).astype(int), 0, H - 1)
    cell = max(1, int(round(sc)))
    cid = (iy // cell) * (W // cell + 2) + (ix // cell)
    order = np.lexsort((d, cid))
    first = np.ones(len(order), bool)
    first[1:] = cid[order][1:] != cid[order][:-1]
    dmin = np.empty(cid.max() + 1, np.float64)
    dmin[cid[order][first]] = d[order][first]
    visible = d <= dmin[cid] + depth_tol
    col = rgb[iy, ix]
    inside = alpha[iy, ix]
    visible &= inside
    known = visible
    if known.sum() > max_known_for_fill:               # keep the fill cheap: use a random subset of known voxels as sources
        r = rng or np.random.default_rng(0)
        keep = np.zeros_like(known)
        keep[r.choice(np.flatnonzero(known), max_known_for_fill, replace=False)] = True
        src = nearest_fill(coords, keep)
    else:
        src = nearest_fill(coords, known)
    colours = col[src]
    return colours, known, (px, py, sc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coords", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="auto")
    ap.add_argument("--depth-tol", type=float, default=2.0)
    a = ap.parse_args(argv)
    coords = np.fromfile(a.coords, dtype=np.int32).reshape(-1, 3)
    rgb, alpha = load_image(a.image)
    H, W = alpha.shape
    ys, xs = np.nonzero(alpha)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    if a.view == "auto":
        cands = []
        for perm in itertools.permutations(range(3)):
            for signs in itertools.product((1, -1), repeat=3):
                v = [(perm[i], signs[i]) for i in range(3)]
                px, py, d, sc = project(coords, v, W, H, bbox)
                cands.append((silhouette_iou(px, py, sc, alpha, W, H), v))
        cands.sort(key=lambda t: -t[0])
        print("best views by silhouette IoU:")
        for iou, v in cands[:6]:
            print(f"  {view_name(v):14s} IoU {iou:.3f}")
        print("note: the sign of the depth axis (which side faces the camera) is NOT decided by the silhouette;"
              " the first candidate is used. Check the projected colours against the picture.")
        view = cands[0][1]
    else:
        view = parse_view(a.view)
    colours, known, _ = project_colours(coords, rgb, alpha, view, a.depth_tol)
    M = len(coords)
    out = np.zeros((M, 6), np.float32)
    out[:, 0:3] = colours * 2 - 1
    out[:, 3] = -1.0     # metallic 0
    out[:, 4] = 0.0      # roughness 0.5
    out[:, 5] = 1.0      # alpha 1
    out.tofile(a.out)
    print(f"view {view_name(view)}: {int(known.sum())} of {M} voxels coloured from the picture, rest filled from the nearest; wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
