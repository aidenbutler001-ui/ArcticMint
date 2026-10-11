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

## PC verification: `pc-verify.ps1` (Windows, MSVC + CUDA)

One script that builds the fork on the PC, makes two real dumps, replays them three ways and says PASS/FAIL per asset.

**Not run in the cloud.** It was written without a Windows machine, an NVIDIA GPU or PowerShell 5.1. It was only
parse-checked, and its helper functions (argument quoting, hash/timing parsing, timestamp-masked GLB hash, the logged
process runner with timeout and peak-RSS sampling) were exercised on PowerShell 7 under Linux against real logs. The
build, `nvidia-smi` and GPU parts have never executed, so expect to fix small things on the first run. A static review by a Sonnet subagent found no definite breakers; its might-break items
(x64 MSVC check, null `ExitCode`, `-Ref` with untracked output, unreadable `nvidia-smi` values, trailing-backslash paths,
CUDA bin dir with `-SkipBuild`, timeout killing the process tree) were fixed. It stops at the
first failure and prints which log to read.

```
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1 -Ref claude/arctic-post-mt -FrontOnly
powershell -ExecutionPolicy Bypass -File tools\arctic-post\pc-verify.ps1 -SkipBuild -SkipDump -OutDir D:\pv\last
```

What it does, in order:
1. Preflight: git, cmake, ninja, nvcc, nvidia-smi; imports the MSVC x64 environment (vswhere + vcvars64) if `cl.exe`
   is not on PATH; checks the models dir (`C:\AI\trellis\models`) and the phase0 images
   (`C:\AI\Expiramental3d\phase0\images`: `hard_03_log_cabin`, `hard_02_treasure_chest`, png/jpg/webp).
2. Optional `-Ref` checkout (needs a clean tree), `git submodule update --init --recursive`.
3. Configure + build like the Windows CUDA job in `.github/workflows/release.yml`
   (`-G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86;89 -DGGML_OPENMP=OFF`, override
   with `-CudaArch`): `trellis-cli`, `post-replay` and the CPU tests; runs the tests; builds `tools/arctic-post` with
   MSVC and runs `post-equiv-test` (legacy vs multithreaded, byte for byte). The `decimate_qem.cu` `cudaGetDevice` log
   (a separate commit) gets its first compile check here.
4. Refuses to start if anything else is using GPU 1 (default; `nvidia-smi` order = `CUDA_DEVICE_ORDER=PCI_BUS_ID`):
   a `trellis-cli`/`trellis-server`/`post-replay` process, a compute app on that GPU, more than `-BusyMemMiB` (3000) MiB
   used, or more than `-BusyUtilPct` (15) % utilization. Windows (WDDM) often hides other clients from `nvidia-smi`,
   so memory and utilization are checked too; raise the two thresholds if the monitor on that card trips them. The check
   repeats before every run.
5. `trellis-cli -i <img> -o <out.glb> -m <models> --res 1024 --gpu 1 --seed 1 -v --dump-post <dump>` per image.
   `-DumpMode Raw` uses `TRELLIS_DUMP_POST` instead (raw mesh, the full CLI run continues to the GLB, much slower,
   but replays the real chain and logs the CLI's own stage hashes). Default `Cleaned` stops after the CLI's remesh, so
   the replay re-remeshes an already remeshed mesh; fine for a same-output check, not a faithful timing.
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
field changes on every run, even for the untouched original code (checked in the cloud: with it masked, full-run GLBs
are byte-identical at 1 and 4 threads and with `TRELLIS_POST_LEGACY=1`).

Needs: Windows 10/11, Visual Studio Build Tools (C++ x64), CMake, Ninja (or the one VS ships), a CUDA toolkit with its
bin dir on PATH or `CUDA_PATH`, git, ~15 GB free disk (build + ggml + two dumps of 0.3-0.6 GB), and an idle GPU 1.
Typical run time: build 20-40 min, then ~10-15 min per asset in `Cleaned` mode.
