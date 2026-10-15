#!/usr/bin/env python3
"""Tests for tools/steps_ab.py (no engine, no GPU): python3 tools/test_steps_ab.py"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import steps_ab as ab


class StepsAb(unittest.TestCase):
    def test_arms(self):
        a = ab.build_cmd("cli", "", "i.png", "o.glb", 7, "A", [])
        b = ab.build_cmd("cli", "", "i.png", "o.glb", 7, "B", [])
        self.assertIn("--steps", a); self.assertNotIn("--steps-ss", a)
        self.assertEqual(b[b.index("--steps") + 1], "12"); self.assertEqual(b[b.index("--steps-ss") + 1], "24")
        self.assertEqual(a[a.index("--seed") + 1], "7")

    def test_prefix_and_no_stage_timing(self):
        c = ab.build_cmd("cli", "lock run --", "i", "o", 1, "A", [], stage_timing=False)
        self.assertEqual(c[:2], ["lock", "run"]); self.assertNotIn("--stage-timing", c)

    def test_colour_window(self):
        self.assertTrue(ab.in_window(ab.colour_ratio(0.4, 0.38)))
        self.assertFalse(ab.in_window(ab.colour_ratio(0.4, 0.1)))     # dark-brown failure
        self.assertFalse(ab.in_window(ab.colour_ratio(0.1, 0.5)))     # far too bright
        self.assertFalse(ab.in_window(ab.colour_ratio(None, 0.5)))

    def test_mean_luma_ignores_transparent(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL missing")
        with tempfile.TemporaryDirectory() as d:
            im = Image.new("RGBA", (2, 1)); im.putpixel((0, 0), (255, 255, 255, 255)); im.putpixel((1, 0), (0, 0, 0, 0))
            p = Path(d) / "t.png"; im.save(p)
            self.assertAlmostEqual(ab.mean_luma(p), 1.0, places=3)

    def test_visible_surface_ignores_hidden_interior(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy missing")

        def box(c, h):
            v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * h + c
            q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
            f = [t for a, b, c2, d in q for t in ((a, b, c2), (a, c2, d))]
            return v, np.array(f)

        def subdivide(v, f, n=16):   # many small faces so centroids sample the surface densely
            out_v, out_f = [], []
            for t in f:
                a, b, c = v[t]
                base = len(out_v)
                for i in range(n + 1):
                    for j in range(n + 1 - i):
                        out_v.append(a + (b - a) * i / n + (c - a) * j / n)
                idx = {}
                k = base
                for i in range(n + 1):
                    for j in range(n + 1 - i):
                        idx[(i, j)] = k; k += 1
                for i in range(n):
                    for j in range(n - i):
                        out_f.append((idx[(i, j)], idx[(i + 1, j)], idx[(i, j + 1)]))
                        if j < n - i - 1:
                            out_f.append((idx[(i + 1, j)], idx[(i + 1, j + 1)], idx[(i, j + 1)]))
            return np.array(out_v), np.array(out_f)

        ov, of = subdivide(*box(np.zeros(3), 1.0))
        iv, if_ = subdivide(*box(np.zeros(3), 0.4))
        P = np.vstack([ov, iv]); I = np.vstack([of, if_ + len(ov)])
        UV = np.zeros((len(P), 2)); UV[:len(ov)] = (0.25, 0.5); UV[len(ov):] = (0.75, 0.5)
        tex = np.zeros((4, 4, 3), np.uint8); tex[:, :2] = 255        # left half white (outer), right half black (inner)
        self.assertTrue(ab.visible_faces(P, I)[:len(of)].all())
        self.assertFalse(ab.visible_faces(P, I)[len(of):].any())
        y = ab.visible_luma_arrays(P, UV, I, tex)
        self.assertAlmostEqual(y, 1.0, places=2)                       # whole-atlas mean would be ~0.5

    def test_parse_stages(self):
        log = "[timing] per-stage wall clock\n[timing]   shape_flow                 166.20 s   41.0%\n[timing]   total (marked)   400.00 s\n"
        self.assertEqual(ab.parse_stages(log), {"shape_flow": 166.2})


if __name__ == "__main__":
    unittest.main()
