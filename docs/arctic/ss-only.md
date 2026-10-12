# `trellis-cli --ss-only`: screen seeds by structure before the expensive stages

The sparse-structure flow decides an object's proportions (the active-voxel box on the 32^3 grid), and it is cheap: 17 s of a
660 s chest run at 24 steps (FS2 log). `--ss-only` runs preprocess, DINOv3 and that stage only, then writes the box per seed.

```
trellis-cli IMAGE.png OUT.glb --ss-only [--ss-seeds N] --seed S --steps 24 --gpu 1 --models DIR --require-gpu
```

- Writes `OUT_ss.json` (no mesh, no `OUT.glb`): for each seed S .. S+N-1 the active-voxel count and the box
  (`min`, `max`, `extent` in cells on voxel axes x, y, z).
- Seed S reproduces the structure of a normal run with `--seed S` at the same `--steps` (the RNG is re-seeded per seed, so
  the first noise draw is the one a normal run makes). Check on one run: the log line `active voxels @res32 = N` of the full
  run equals `voxels` for that seed. GPU kernels can differ in the last bits, so a different count by a few voxels is a
  finding to report, not noise to ignore.
- Axes: voxel x = width, y = depth, z = up. `mesh_glb.cpp` rotates (x,y,z) -> (x,z,-y) for the GLB, the Blender import undoes
  it, and finish.py's `--rotate-z 90` then makes x the spec's W (Blender Y) and y the spec's D (Blender X). z is H.
- Resolution: one cell is 1/32 of the long axis, about 3% of it; a thin axis (a winch's 1.0 m depth on a 2.8 m width is
  ~11 cells) is good to about +-1 cell, so read small differences between seeds with that in mind. The shape and texture
  stages refine the surface inside this box; they do not move it by more than a cell or so [E, unchecked].
- Same `--steps` and `--gss` as production, or the screened structure is not the production structure.
- `tools/ss_sweep.py` in the Expiramental3d repo runs this over a folder of images and ranks the seeds against a spec size.

Tests: `trellis-test-ss-only` (CPU only, no models): box maths, JSON writer, flag parsing.
