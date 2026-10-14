# arctic-post

A standalone harness for ArcticMint's CPU post-processing (weld, fill holes, remesh, clean, decimate, UV/bake, GLB). It
builds and runs without a GPU and without ggml, so post-processing changes can be profiled and tested on any machine.
Design notes for the multithreaded stages are in [`docs/arctic/post-mt.md`](../../docs/arctic/post-mt.md).

## Contents

- [Standalone build and tools](#standalone-build-and-tools)
- [PC verification](#pc-verification-pc-verifyps1-windows-msvc--cuda)

## Standalone build and tools

- `CMakeLists.txt`: a standalone Release build of the post stages, `post-replay` and `edgebench`. It doesn't need the
  `thirdparty/ggml` submodule.
  ```
  cmake -S tools/arctic-post -B build-post -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build-post
  ```
- `gen_dump.py` (numpy): writes a synthetic `--dump-post` file. It meshes a noisy, chest-like SDF on an N³ grid with
  surface nets (the same topology class as FlexiDualGrid), then injects near-duplicate vertices, holes, flipped
  blocks and floaters, plus a sparse PBR volume. At N=1024 it gives a raw mesh of ~17M faces, close to the 16.5M faces seen on a real 1024 run.
  ```
  python3 tools/arctic-post/gen_dump.py d.bin --N 1024 --pbr-res 512
  build-post/post-replay d.bin out.glb --geometry-res 1024 --band 2 --faces 300000
  ```
  `--geometry-res` must be 512 or 1024 (or 1536). Use `--band 2` at 1024 to match the CLI default.
- `edgebench.cpp`: a microbenchmark of edge counting with `unordered_map` versus sort-based methods, used for the profile.

The synthetic meshes are box-like, and their defect rates are invented. Validate every change on real dumps from
a real machine (`trellis-cli ... --dump-post PATH`) before trusting the timings.

## PC verification: `pc-verify.ps1` (Windows, MSVC + CUDA)

One script that builds ArcticMint on a Windows machine, makes two real dumps, replays them three ways and says PASS/FAIL per asset.

**Not yet run on real hardware.** It was written without a Windows machine, an NVIDIA GPU or PowerShell 5.1. It was only
parse-checked, and its helper functions (argument quoting, hash/timing parsing, timestamp-masked GLB hash, the logged
process runner with timeout and peak-RSS sampling) were exercised on PowerShell 7 under Linux against real logs. The
build, `nvidia-smi` and GPU parts have never executed, so expect to fix small things on the first run. A static review found no definite breakers; its might-break items
(x64 MSVC check, null `ExitCode`, `-Ref` with untracked output, unreadable `nvidia-smi` values, trailing-backslash paths,
CUDA bin dir with `-SkipBuild`, timeout killing the process tree) were fixed. It stops at the
first failure and prints which log to read.

```
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1 -Ref my-branch -FrontOnly
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1 -SkipBuild -SkipDump -OutDir out\pv-last
```

What it does, in order:
1. Preflight: git, cmake, ninja, nvcc, nvidia-smi; imports the MSVC x64 environment (vswhere + vcvars64) if `cl.exe`
   is not on PATH; checks the models directory (`-ModelsDir`) and the test images (`-ImagesDir`, names given with `-Images`; png, jpg
   or webp). Both are parameters at the top of the script; set them for your machine.
2. Optional `-Ref` checkout (needs a clean tree), `git submodule update --init --recursive`.
3. Configure + build like the Windows CUDA job in `.github/workflows/release.yml`
   (`-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89 -DGGML_OPENMP=OFF`, override
   with `-CudaArch`; the release workflow builds `75;80;86;89;90;120`): `trellis-cli`, `post-replay` and the CPU tests; runs the tests; builds `tools/arctic-post` with
   MSVC and runs `post-equiv-test` (legacy vs multithreaded, byte for byte). The `decimate_qem.cu` `cudaGetDevice` log
   (a separate commit) gets its first compile check here.
4. Refuses to start if anything else is using GPU 1 (default; `nvidia-smi` order = `CUDA_DEVICE_ORDER=PCI_BUS_ID`):
   a `trellis-cli`/`trellis-server`/`post-replay` process, a compute app on that GPU, more than `-BusyMemMiB` (3000) MiB
   used, or more than `-BusyUtilPct` (15) % utilization. Windows (WDDM) often hides other clients from `nvidia-smi`,
   so memory and utilization are checked too; raise the two thresholds if the monitor on that card trips them. The check
   repeats before every run.
5. `trellis-cli -i <img> -o <out.glb> -m <models> --res 1024 --gpu 1 --seed 1 -v` per image with
   `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `TRELLIS_DUMP_POST=<dump>` (default `-DumpMode Raw`): the dump is the raw decoded
   mesh (after the centroid `fill_holes`, before weld), the full CLI run continues to its GLB (several minutes per
   asset), and with `TRELLIS_POST_HASH=1` the CLI logs its own stage hashes too. `-DumpMode Cleaned` uses
   `--dump-post` instead: it stops after the CLI's remesh, so the replay re-remeshes an already remeshed mesh
   (faster, but not the real chain).
6. `post-replay <dump> <glb> --geometry-res 1024 --band 2 --faces 300000 --no-audit` per image, with
   `TRELLIS_POST_HASH=1 TRELLIS_POST_TIMING=1 TRELLIS_VERBOSE=1` and `CUDA_VISIBLE_DEVICES=<gpu>`, three times:
   `TRELLIS_POST_THREADS=1`, `=24` (`-HighThreads`), and `TRELLIS_POST_LEGACY=1`. `post-replay` never runs ggml, so
   without `CUDA_VISIBLE_DEVICES` its GPU QEM would run on CUDA device 0 (the 3060 Ti); the `[decimate_qem_gpu] current
   device` line from the new log is copied into the summary.
7. Writes `summary.md`, `summary.csv` (per-run stage times, peak RSS, QEM device, GLB hash), `hashes.csv` (every stage
   hash) and `logs\` under `pcverify-out\<timestamp>\`.

Verdict per asset: **PASS** if the vertex/face hashes of `input, weld, fill_small_holes, bvh, remesh, clean_mesh,
drop_components` are identical in all three runs and, unless `-FrontOnly`, so are `decimate_qem, weld2, fill2, drop2`
and the GLB SHA-256. **FAIL** otherwise (`FAIL (tail only)` if only the post-QEM part differs: then look at the GPU QEM,
xatlas and the GLB writer before the post-MT code). The GLB is compared with its `"generated"` timestamp zeroed: that
field changes on every run, even for the untouched original code (checked on a Linux build: with it masked, full-run GLBs
are byte-identical at 1 and 4 threads and with `TRELLIS_POST_LEGACY=1`).

Tail tolerance: if only the tail differs (GPU QEM is non-deterministic), each tail stage is compared with the legacy run on face count (`-TailFaceTolPct`, default 0.5%) and surface area (`-TailAreaTolPct`, default 0.1%): within both = `PASS (front exact; tail within ...)`, outside = `FAIL (tail outside tolerance)`. The `drop2` stage has its own limits (`-Drop2FaceTolPct` 7%, `-Drop2AreaTolPct` 2%): it drops components under 3% of the largest one, so a 0.01% QEM difference moves whole components across the threshold (log cabin, same card: two identical runs differ by 0.59% faces / 0.40% area there, and 1.14% / 1.08% against legacy, with every earlier stage under 0.1% / 0.005%). Hausdorff distance is not checked by the script.

## Requirements

Windows 10/11, Visual Studio Build Tools (C++ x64), CMake, Ninja (or the one VS ships), a CUDA toolkit with its
bin dir on PATH or `CUDA_PATH`, git, ~15 GB free disk (build + ggml + two dumps of 0.3-0.6 GB), and an idle GPU 1.
Typical run time (one machine): build 20-40 min, then roughly 25-40 min per asset in the default `Raw` mode (full CLI run plus three replays).
