#!/usr/bin/env python3
"""CPU check of tools/tex_project.py on a synthetic coloured box seen from a known view."""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import tex_project as tp  # noqa: E402


def main():
    res = 40
    # hollow box shell, 30 x 14 x 20 (x right, y depth, z up); the front face (min y, seen from -y) is red, back blue
    pts = []
    for x in range(5, 35):
        for y in range(10, 24):
            for z in range(8, 28):
                if x in (5, 34) or y in (10, 23) or z in (8, 27):
                    pts.append((x, y, z))
    coords = np.array(pts, np.int32)
    # picture: camera looks along +y; image right = +x, up = +z; front face (y=10) red, everything else grey
    W, H, s = 160, 160, 4
    img = np.full((H, W, 4), 0, np.uint8)
    x0, y0 = 20, 20
    img[y0:y0 + 20 * s, x0:x0 + 30 * s, :3] = (220, 30, 30)
    img[y0:y0 + 20 * s, x0:x0 + 30 * s, 3] = 255
    d = tempfile.mkdtemp()
    cp, ip, op = (os.path.join(d, n) for n in ("c.i32", "i.png", "o.f32"))
    coords.tofile(cp)
    Image.fromarray(img).save(ip)
    rc = tp.main(["--coords", cp, "--image", ip, "--out", op])
    assert rc == 0
    out = np.fromfile(op, np.float32).reshape(-1, 6)
    assert out.shape == (len(coords), 6)
    # explicit correct view: front voxels (y == 10) must come out red, back-face voxels get filled from visible neighbours (also red)
    view = tp.parse_view("+x,+z,+y")
    rgb, alpha = tp.load_image(ip)
    col, known, _ = tp.project_colours(coords, rgb, alpha, view)
    front = coords[:, 1] == 10
    assert known[front].mean() > 0.95, known[front].mean()
    assert np.allclose(col[front].mean(0), np.array([220, 30, 30]) / 255, atol=0.02), col[front].mean(0)
    # the wrong depth sign would colour the BACK face as visible; with +y the front face (y=10) is nearest
    assert not known[coords[:, 1] == 23].any()
    # auto must find an x/z-plane view with high IoU
    auto_iou = None
    from itertools import permutations, product
    H_, W_ = alpha.shape
    ys, xs = np.nonzero(alpha)
    bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    px, py, dd, sc = tp.project(coords, view, W_, H_, bbox)
    auto_iou = tp.silhouette_iou(px, py, sc, alpha, W_, H_)
    assert auto_iou > 0.9, auto_iou
    # value range / neutral channels
    assert out[:, :3].min() >= -1 and out[:, :3].max() <= 1
    assert (out[:, 3] == -1).all() and (out[:, 4] == 0).all() and (out[:, 5] == 1).all()
    print("tex_project: OK (front-face visibility %.2f, IoU %.3f)" % (known[front].mean(), auto_iou))


if __name__ == "__main__":
    main()
