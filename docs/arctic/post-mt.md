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

## Hole filling (`fill_holes`, `fill_small_holes`): needs a decision

Both iterate `std::unordered_map`s over the (tiny) boundary set to pick loop start vertices, so loop order and the fan
apex depend on the standard library's hash order (differs between libstdc++ and MSVC today). The big cost is the edge
use-count map (25-35 s at 17M faces). The boundary *set* can be found with a sort in a few seconds, but reproducing
the *order* means simulating a particular library's hash table. See the proposal section below once measured.

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
