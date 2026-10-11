# arctic-post: CPU post-processing harness (Arctic fork only)

Tools for profiling and speeding up trellis.cpp's CPU post-processing (weld → fill holes → remesh → clean → decimate →
UV/bake → GLB) without a GPU and without ggml. The profile and the plan are in the project repo
(`aidenbutler001-ui/expiramental3d`, branch `claude/lucid-hopper-9wbryp`): `research/notes/arctic-post-profile.md`
and `research/notes/phase0-cpu-bottleneck.md`.

- `CMakeLists.txt`: a standalone Release build of the post stages, `post-replay` and `edgebench`. It doesn't need the
  `thirdparty/ggml` submodule.
  ```
  cmake -S tools/arctic-post -B build-post -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-post
  ```
- `gen_dump.py` (numpy): writes a synthetic `--dump-post` file. It meshes a noisy, chest-like SDF on an N³ grid with
  surface nets (the same topology class as FlexiDualGrid), then injects near-duplicate vertices, holes, flipped
  blocks and floaters, plus a sparse PBR volume. At N=1024 it gives a raw mesh of ~17M faces, close to the PC's 16.5M.
  ```
  python3 tools/arctic-post/gen_dump.py d.bin --N 1024 --pbr-res 512
  build-post/post-replay d.bin out.glb --geometry-res 1024 --band 2 --faces 300000
  ```
  `--geometry-res` must be 512 or 1024 (or 1536). Use `--band 2` at 1024 to match the CLI default.
- `edgebench.cpp`: a microbenchmark of edge counting with `unordered_map` versus sort-based methods, used for the profile.

The synthetic meshes are box-like, and their defect rates are invented. Validate every change on real dumps from
the PC (`trellis-cli ... --dump-post PATH`) before trusting the timings.
