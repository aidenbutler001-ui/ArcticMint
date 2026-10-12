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

    def test_parse_stages(self):
        log = "[timing] per-stage wall clock\n[timing]   shape_flow                 166.20 s   41.0%\n[timing]   total (marked)   400.00 s\n"
        self.assertEqual(ab.parse_stages(log), {"shape_flow": 166.2})


if __name__ == "__main__":
    unittest.main()
