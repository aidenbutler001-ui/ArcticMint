# PR history from the old public fork

These pull requests were merged in an earlier fork, `aidenbutler001-ui/ArcticMint_trellis.cpp`, before the engine moved to
this repository on 2026-10-11. Their commits are in this repo's history; the descriptions are kept here because they
lived only on GitHub.

## #1 Add arctic-post: CPU post-processing profiling tools (merged 2026-10-11)

Commits `8617644` (install.ps1: ASCII dashes, so Windows PowerShell 5.1 can parse it; the bytes of the em-dashes
decoded as a curly quote in the ANSI code page) and `89f692d` (tools/arctic-post: standalone CPU-only build of the post
stages, `gen_dump.py` synthetic dump generator, `edgebench.cpp` microbenchmark).

## #2 Arctic: multithreaded CPU post-processing with byte-identical output (merged 2026-10-11, squash `58b915e`)

Multithreaded weld_vertices, clean_mesh, drop_small_components, boundary edges, TriBvh::build and remesh_dc's serial
parts. They produce byte-identical output to the legacy single-thread code at any thread count.

- **Infrastructure** (`include/post_parallel.h`, `src/post_parallel.cpp`): `parallel_for` with dynamic chunks,
  `parallel_tasks`, `fixed_chunks` (deterministic, independent of the thread count), a stable parallel radix sort for
  64-bit keys, the `FlatU64Map` open-addressing map, `build_edge_csr`, and per-stage hashing (`TRELLIS_POST_HASH`).
- **Mesh stages** (`src/post_mesh.cpp`):
  - weld: stable sort of (cell key, vertex), parallel conflict flagging, serial resolution in index order,
    parallel remap;
  - clean_mesh: parallel edge CSR, then the legacy DFS on flat arrays;
  - drop_small_components: parallel union-find, deterministic compaction by prefix sum.
- **Existing files:**
  - `tri_bvh.cpp`: parallel centroids; concurrent node build into pre-assigned ID ranges.
  - `remesh_dc.cpp`: parallel candidate and active-voxel passes, plus step timing.
  - `uv_bake.cpp`: one-line dispatch hooks.
  - `trellis_cli.cpp`: hash reporting.
- **Tests:** `tools/arctic-post/post_equiv_test.cpp` (randomised legacy-vs-new at 1, 2, 3, 4, 7 and 16 threads);
  golden hashes in `docs/arctic/post-mt-golden.txt`; design notes in `docs/arctic/post-mt.md`.
- **Environment variables:**
  - `TRELLIS_POST_THREADS` (default: hardware concurrency);
  - `TRELLIS_POST_LEGACY=1` (old single-thread path);
  - `TRELLIS_POST_HASH=1`;
  - `TRELLIS_POST_TIMING=1`;
  - `TRELLIS_POST_CANON_HOLES=1` (opt-in sorted hole order; changes output).
