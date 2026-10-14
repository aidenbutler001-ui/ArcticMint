# `trellis-cli --ss-only`: screen seeds by structure before the expensive stages

The sparse-structure flow decides an object's proportions (the active-voxel box on the 32^3 grid), and it is cheap: 17 s of a
660 s chest run at 24 steps. `--ss-only` runs preprocess, DINOv3 and that stage only, then writes the box per seed.

```
trellis-cli IMAGE.png OUT.glb --ss-only [--ss-seeds N] --seed S --steps 24 --gpu 1 --models DIR --require-gpu
```

- Writes `OUT_ss.json` (no mesh, no `OUT.glb`): for each seed S .. S+N-1 the active-voxel count and the box
  (`min`, `max`, `extent` in cells on voxel axes x, y, z).
- Seed S reproduces the structure of a normal run with `--seed S` at the same `--steps` (the RNG is re-seeded per seed, so
  the first noise draw is the one a normal run makes). Check on one run: the log line `active voxels @res32 = N` of the full
  run equals `voxels` for that seed. GPU kernels can differ in the last bits, so a different count by a few voxels is a
  finding to report, not noise to ignore.
- Axes: voxel x = width, y = depth, z = up. `mesh_glb.cpp` rotates (x,y,z) -> (x,z,-y) for the GLB, so after loading the GLB the
  voxel x axis is the model's width and y its depth, with z up.
- Resolution: one cell is 1/32 of the long axis, about 3% of it; a thin axis (a winch's 1.0 m depth on a 2.8 m width is
  ~11 cells) is good to about +-1 cell, so read small differences between seeds with that in mind. The shape and texture
  stages refine the surface inside this box; they do not move it by more than a cell or so [E, unchecked].
- Same `--steps` and `--gss` as production, or the screened structure is not the production structure. To make the later flows cheaper without
  moving the structure, run the sweep with `--steps 24` and the full run with `--steps 12 --steps-ss 24`: `--steps-ss` sets the sparse-structure flow
  only (default: the `--steps` value), so seed S gives the sweep's structure and only the shape and texture flows run at 12 steps.
- A sweep driver that ranks seeds against an expected size is not part of this repository; `tools/arctic-batch/size_judge.py` does a similar proportion check on drafts.

Tests: `trellis-test-ss-only` (CPU only, no models): box maths, JSON writer, flag parsing.
