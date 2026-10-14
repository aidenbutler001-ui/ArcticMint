# Step-count A/B (`tools/steps_ab.py`)

Question: can shape and texture flows run at 12 steps (with the structure flow kept at 24 via `--steps-ss`) without losing quality?

- Arm A: `--steps 24` (the standing rule). Arm B: `--steps 12 --steps-ss 24`.
- Pieces: five TRELLIS-type M2 props from `phase0/m2_specs.csv` (default SM_ChoirStall, SM_StormStump, SM_CargoStack, SM_CrackedAltar, SM_ClockworkDrive; override with `--pieces`). Source pictures are `<name>.png` in `--images` (override with `--image-pattern`).
- Each run: `--res 1024 --faces 150000 --atlas 4096`, same seed in both arms. More `--seeds` give a colour failure rate.
- Output: `results.csv` / `results.json` (wall time, per-stage seconds when the build has `--stage-timing`, colour ratio) and one `run.log` per run.
- Colour check: area-weighted luminance of the base texture over the faces visible from outside (a face counts if any of 26 directions sees its centroid in a depth buffer; hidden interior faces, up to ~73% of the atlas, are dropped), against the source picture's foreground. Same ratio and 0.6..1/0.6 window as `--tex-reroll`. Read from the finished GLB after the run (numpy + PIL needed; the column is empty without them); no engine change, and `--tex-reroll` stays off in both arms. If the colour root-cause thread lands its own visible-surface measure, swap it in for `visible_luma`.
- Shape quality is not scored. Compare arms with `tools/glb_metrics.py` and renders.
- Needs the PC: run it from the queue under the gpulock tooling (`--prefix` wraps each command) and after the usual go. `--dry-run` prints the commands only.

Tests: `python3 tools/test_steps_ab.py` (no engine or GPU).
