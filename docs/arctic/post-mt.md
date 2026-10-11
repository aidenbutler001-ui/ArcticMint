# Arctic: multithreaded CPU post-processing (`claude/arctic-post-mt`)

Goal: make weld / hole fill / BVH / remesh / clean much faster using sort-based flat data structures and `std::thread`,
with output byte-identical to the original code. Plan and profile: project repo
`research/notes/arctic-post-profile.md`.

## Switches

| Env | Effect |
|---|---|
| `TRELLIS_POST_THREADS=N` | worker threads (default `hardware_concurrency`) |
| `TRELLIS_POST_LEGACY=1` | run the original code paths (A/B checks) |
| `TRELLIS_POST_HASH=1` | print a 64-bit hash of the vertex/face buffers after each stage (`[post-hash]` lines, also in `trellis-cli`) |

`post-replay` gained `--fill-holes` (run the CLI's centroid `fill_holes` first), `--stop-before-qem`, `--no-audit`.

## Code map

- `include/post_parallel.h`, `src/post_parallel.cpp`: `parallel_for` (dynamic chunks), `parallel_tasks`, `fixed_chunks`
  (chunking independent of thread count), stable parallel radix sort, `FlatU64Map` (open addressing, parallel build),
  `build_edge_csr` (sorted undirected-edge adjacency), hash debug helpers.
- `src/post_mesh.cpp`: `weld_vertices`, `clean_mesh`, `drop_small_components`. Dispatched from `uv_bake.cpp` (3 one-line
  hooks).
- `src/tri_bvh.cpp`: parallel `TriBvh::build`; `src/remesh_dc.cpp`: flat tables + parallel assembly.
- `tools/arctic-post/post_equiv_test.cpp`: randomized legacy-vs-new equivalence test (thread counts 1,2,3,4,7,16).

## Why the output is identical

- **weld**: legacy scan = vertex joins the first earlier *kept* vertex within eps in the 27 cell buckets (dz,dy,dx order).
  New: stable sort of (cell key, index); parallel flag of vertices that have *any* earlier vertex within eps (everything
  else is unconditionally kept); the flagged ones are resolved serially in index order with the legacy scan; kept ids by
  prefix sum. Cell keys are the same XOR-packed keys (collisions behave as before).
- **clean_mesh**: the legacy DFS only needs "the other face of a 2-face edge". That table comes from a sorted edge CSR
  (parallel); the DFS itself is unchanged and serial, on flat arrays, so even non-orientable regions come out the same.
- **drop_small_components**: same union-find; face counts in a flat array; compaction + first-appearance vertex
  renumbering by atomic-min of first index + prefix sum.
- **TriBvh::build**: legacy pops spans LIFO and numbers children when the parent is processed. A node processed with id
  counter c gets children (c, c+1); its right subtree is processed next (counter c+2), then the left one
  (counter c+1+count(right)). Node counts are closed-form, so subtrees can be built concurrently into pre-assigned ids.
  Same `nth_element` calls on the same ranges, so identical nodes and primitive order.
- **remesh_dc**: active voxels come out in sorted key order; legacy vertex numbering (first visit in voxel order) is
  reproduced exactly: a corner is new iff no voxel at one of 13 lexicographically-negative offsets owns it. Quad lists
  are built per fixed chunk and concatenated in chunk order. UDF passes use dynamic chunks.

## Hole filling (`fill_holes`, `fill_small_holes`): canonical order, opt-in

Both pick loop start vertices by iterating `std::unordered_map`s, so loop order, the fan apex and the new centroid
vertex ids depend on the standard library's hash order (already different between libstdc++ and MSVC). The expensive
part is the edge use-count map (25-35 s at 16.5M faces); the boundary set itself is tiny.

`TRELLIS_POST_CANON_HOLES=1` finds the boundary edges with a parallel sorted edge CSR (`post::boundary_edges`) and
processes loops in ascending vertex id (so a loop starts at its smallest vertex). Default is still the legacy
order and is byte-identical to before. **Re-baselining to the canonical order needs Aiden's approval.**

Measured on d1024 (4.98M raw faces, same hole set both ways):

| | legacy | canonical |
|---|---|---|
| `fill_holes` centroid fans | 3.2 s, 2889 filled | 0.3-0.5 s, 2889 filled |
| `fill_small_holes` | 5.5 s, +145 faces | 0.4 s, +145 faces |
| area after `fill_holes` | 1.77187933 | 1.77187933 |
| area after `fill_small_holes` | 1.76823585 | 1.7682359 (rel. diff 3e-8) |

What differs: the fan apex of each filled loop and the order of appended faces / centroid vertices, so later hashes
differ. The remesh output keeps identical faces (`faces=7da7497e77431f32` in both); only the positions of a few
vertices near filled holes change, because the projection-back snaps to the (slightly different) fan patch.
On non-manifold input (a boundary path that runs into a cycle) the *set* of loops `fill_holes` fills also depends on
visiting order in the legacy code (a path start marks the cycle visited), so the canonical order can fill a
different loop there; this is order-dependence the legacy code already has between standard libraries.
`post-equiv-test` checks canonical mode is identical at 1,2,3,4,7,16 threads and, for `fill_small_holes`, that it
fills the same number of loops and leaves the same boundary set as legacy.

## Follow-up for the PC: CUDA device for `decimate_qem_gpu` (task B0)

`src/decimate_qem.cu` never calls `cudaSetDevice`; it uses the calling thread's current device.
- `trellis-cli` runs QEM on the main thread after the texture/shape decode. ggml's CUDA backend is process-wide
  (`Model::free()` never frees it) and calls `ggml_cuda_set_device(<chosen device>)` for every graph compute and
  buffer operation, so the current device is still the one picked by `--gpu` at QEM time. Low risk.
- Real cases that land on CUDA device 0 (the 3060 Ti with `CUDA_DEVICE_ORDER=PCI_BUS_ID`): `post-replay` (never
  touches ggml) and any run where no ggml GPU op ran on that thread (e.g. all models on CPU).
- Not changed here: no nvcc in the cloud container, so a `.cu` edit could not be compile-checked. If wanted, add a
  small `cudaSetDevice(g_post_cuda_device)` hook in `decimate_qem_gpu`, set from `cfg.gpu`.
- To confirm on the PC, log `int d; cudaGetDevice(&d); fprintf(stderr, "[decimate_qem_gpu] device %d\n", d);` at
  the top of `decimate_qem_gpu` and check it equals the `--gpu` device (peak allocation at 30M faces is ~3-5 GB, so
  it still fits the 3060 Ti if it does land there).

## Golden references (legacy code, TRELLIS_POST_HASH=1, 4 vCPU)

Dumps (synthetic, `tools/arctic-post/gen_dump.py`): `d512a` (N=512 seed 1), `d512b` (N=512 seed 7 shell 0.03 amp 1.5),
`d1024` (N=1024 seed 3). Run with `--fill-holes --no-audit --faces 300000`; geometry-res = N, band = N/512.

| Stage | d512a verts / faces hash | d1024 (front chain) verts / faces hash |
|---|---|---|
| fill_holes | a86344f86f89a753 / a6bd649323995fea | 1a4c2de2fc5d144b / 97826c21ba754475 |
| weld | 66cc206951d51681 / 627ff65a72f6298b | 011ecf01e4a743f1 / 2931a669d458d2cb |
| fill_small_holes | 66cc206951d51681 / 951fe9031b74670d | 011ecf01e4a743f1 / 331a7896cd21e62a |
| bvh (nodes+prims) | bd7ec523ed42eb54 | d57a49f8011df891 |
| remesh | a1c95f0f19bb550c / 61796f0f2a6bfa0d | f177eabec6fcda54 / 7da7497e77431f32 |
| clean_mesh | (same as remesh, flipped 0) | (same as remesh) |
| drop_components | b902f0840aa4139f / b2e48a8f4e163f82 | 981d7b21fc10c5ea / 94babccf23777da4 |
| final GLB sha256[:16] | 4fe85a54c0cef47b | (front chain only) |

d512b final GLB 7999c642c96bbadd. Full tables: see `docs/arctic/post-mt-golden.txt`.

## Timings (seconds, 4 vCPU container, d1024 = 4.98M raw faces -> 9.1M after remesh)

| Stage | legacy | new, 4 threads | identical |
|---|---|---|---|
| fill_holes | 3.2 | 3.0 (unchanged) | n/a |
| weld | 4.2 | 1.5 | yes |
| fill_small_holes | 5.5 | 6.2 (unchanged) | n/a |
| bvh | 2.1 | 0.7 | yes |
| remesh | 32.3 | 20.7 | yes |
| clean_mesh | 18.5 | 0.7 | yes |
| drop_components | 0.6 | 0.6 | yes |
| front chain wall / peak RSS | 66.7 s / 1.73 GB | 33.6 s / 1.47 GB | yes |

## Results at the PC's size: 16.45M raw faces -> 29.4M after remesh (`dC.bin`, 4 vCPU container)

Generated with `gen_dump.py dC.bin --N 1024 --pbr-res 512 --seed 5 --shell 0.03 --amp 2.0 --scale 1.2`; run with
`--geometry-res 1024 --band 2 --fill-holes --no-audit --stop-before-qem`. Per-stage vertex and face hashes
identical to the legacy binary at 1 and 4 threads (`input`..`drop_components`; the legacy binary just lacks the
`area=` column). Seconds:

| Stage | legacy (1 thread, hash maps) | new, 4 threads | new, 1 thread | new 4 thr + canonical holes |
|---|---|---|---|---|
| fill_holes (centroid) | 16.9 | 19.4 (legacy order, unchanged) | 19.7 | **1.0** |
| weld_vertices | 17.4 | **4.6** | 15.1 | 4.9 |
| fill_small_holes | 25.4 | 28.2 (legacy order, unchanged) | 29.5 | **1.9** |
| TriBvh::build | 7.5 | **2.6** | 7.4 | 2.8 |
| remesh_dc | 92.3 | **75.1** | 284.1 | 77.3 |
| clean_mesh | 73.5 | **3.0** | 5.0 | 3.0 |
| clean + drop_components | 75.7 | **5.4** | 7.8 | 5.6 |
| front chain wall | 235.9 | **136.3** | 364.7 | **94.5** |
| peak RSS | 5.47 GB | 4.51 GB | 4.53 GB | 4.24 GB |

(Legacy remesh already used all 4 threads for its UDF passes; the 1-thread column is the new code with
`TRELLIS_POST_THREADS=1`.) With the canonical hole order the front chain is 2.5x faster than legacy at 4 threads and
memory is lower. Canonical order also changes the remesh by 8 faces here (29,383,316 vs 29,383,308) because the
fill patches differ.

Remesh breakdown at 4 threads (`TRELLIS_POST_TIMING=1`): candidate bitset 2.7, active voxels (UDF) 28.0, voxel/vertex
tables 10.0, UDF at grid vertices 26.0, dual vertices 3.3, quads + compaction 4.9. The two UDF passes (54 s, BVH
closest-point queries) are ~72% of what is left and scale with cores: at 24 threads expect ~9 s, so the remaining
serial-ish parts (tables 10 s at 4 threads) become the next target. `dual_grid_to_mesh` (stage 0, hash lookups) and the
`audit()` calls in `post-replay` were not touched.

### Update: remesh tables without hashing (dense occupancy + rank)

`remesh_dc` now keeps active voxels and grid vertices in dense occupancy bitsets with per-word prefix popcounts
(`DenseSet`): lookup = bounds check, bit test, popcount. A voxel's id is its rank among set cells (same as its index
in `acoord`, which is in linear order). Grid vertices are numbered by rank in linear vertex order instead of first
visit; vertex ids only index `fvert` (and, in the debug TETS branch, a dedupe key), so the output is unchanged.
This removes the 13-probe first-visit test and both flat tables (up to ~1.5 GB at 16M faces).

16.45M-face dump, 4 vCPU, hashes identical to legacy at 1 and 4 threads:

| | before (flat hash tables) | dense tables |
|---|---|---|
| voxel/vertex tables (4 thr) | 10.0 s | 0.54 s |
| dual vertices (4 thr) | 3.3 s | 2.1 s |
| remesh_dc total (4 thr) | 75.1 s | 64.5 s |
| remesh_dc total (1 thr) | 284.1 s | 236.2 s |
| front chain wall, legacy hole order, 4 thr | 136.3 s | 124.9 s |
| peak RSS | 4.51 GB | 3.29 GB |

### Update: `dual_grid_to_mesh` (stage 0, runs inside trellis-cli before the post stages)

Called once from `trellis_cli.cpp` (and nowhere else), so the function-level change covers the CLI path. Vertex
arithmetic is the same code (a lambda shared by both paths); the coordinate map is a `FlatU64Map` filled with
`insert_max` (legacy `cmap[key] = i` keeps the LAST duplicate, the largest index does too); faces are built per fixed
chunk of voxels and concatenated in chunk order. `tools/arctic-post/dualgrid_bench.cpp` builds a synthetic `ShapeOut`
from a dump (parent-major order like the C2S output, optional duplicate coordinates) and compares byte for byte at
1,2,4,7,16 threads: identical. 7.97M voxels at res 1024: legacy 3.3 s, new 4.5 s (1 thread) / 2.2 s (2) / 1.2 s (4).
Small because the stage is cheap; forced single-thread runs are ~1 s slower than legacy (`TRELLIS_POST_LEGACY=1`
restores it). Real shape-decoder features were not available in the cloud, so the features are random.
