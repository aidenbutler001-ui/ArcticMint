# arctic-batch: overlap GPU generation with CPU post (ArcticMint, experimental branch)

Changes on `claude/arctic-parallel` (from Fable's parallel-processing review, plan `research/notes/parallel-implementation-plan.md`
in Expiramental3d): `--post-workers` now defaults to **1**; `--dump-mode raw`; `batch.csv` has UTC stage timestamps,
`warnings`, `front_hash`, `dump_ok`; a one-page `batch_report.md`; `--preflight` (default on); `--gpulock`; `--phase-lock`;
one finish retry after an out-of-memory; `TRELLIS_REMESH_MEM_GIB`. See "Parallel-processing options" at the end.

One asset today = GPU generation (~150 s) then CPU post (~65 s @24 threads); each side idles during the other.
`arctic_batch.py` (Python stdlib only) runs the 4070 SUPER back to back and post-processes finished dumps in a pool.

- **GPU worker**: `trellis-cli -i IMG -o DUMPDIR\NAME.unused.glb -m MODELS --res 1024 --gpu 1 --seed S --dump-post DUMPDIR\NAME.bin`
  with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. `--dump-post` runs the CLI's weld -> fill_small_holes -> DC remesh -> clean -> drop
  floaters, writes the *already remeshed* mesh + sparse PBR volume and exits (no decimate/UV/bake/GLB).
- **Post pool** (N workers, default 1; the parallel review found a second worker mostly contends for the 3060 Ti and the CPU, so raise it only after `batch_report.md` says it may help): `post-replay DUMP OUT.glb --geometry-res 1024 --band 2 --faces 300000 --no-audit`
  with `TRELLIS_POST_THREADS = total_threads / N` (24 with one worker, 12 with two). Note post-replay re-remeshes the cleaned dump
  (as `pc-verify.ps1 -DumpMode Cleaned` does); that is the dump semantics requested, and it is the same input in serial and overlap mode.
- **QEM device**: post-replay never runs ggml, so its GPU QEM would use CUDA device 0. The runner sets
  `CUDA_VISIBLE_DEVICES=<--post-gpu>` (default `0` = 3060 Ti, PCI order), keeping the 4070 SUPER free.
  `--qem cpu` sets `CUDA_VISIBLE_DEVICES=-1`: `decimate_qem()` then logs "GPU decimation unavailable" and uses its CPU
  path (slower; stated by the code as output-unaffected, but GPU QEM was seen to be non-deterministic run to run, so
  expect CPU-vs-GPU QEM GLBs to possibly differ).
- Backpressure: at most `--max-queue` (2) finished dumps wait for a post worker (dumps are 0.3-0.6 GB); gen blocks beyond that.
  Dumps are deleted after a successful post unless `--keep-dumps`.
- `--serial`: gen then post, one asset at a time, post gets all `--total-threads` (today's behaviour) for A/B.

## Run (PowerShell, from the repo root, after building trellis-cli and post-replay)

```
$b = "D:\ArcticMint\build"   # wherever trellis-cli.exe / post-replay.exe live
powershell -ExecutionPolicy Bypass -File tools\arctic-batch\arctic-batch.ps1 run C:\AI\Expiramental3d\phase0\images -o D:\batch\overlap `
  --cli $b\trellis-cli.exe --replay $b\post-replay.exe -m C:\AI\trellis\models --post-workers 1 --total-threads 24
powershell -ExecutionPolicy Bypass -File tools\arctic-batch\arctic-batch.ps1 run C:\AI\Expiramental3d\phase0\images -o D:\batch\serial --serial `
  --cli $b\trellis-cli.exe --replay $b\post-replay.exe -m C:\AI\trellis\models
```

Output: `OUT\*.glb`, `OUT\logs\NAME.{gen,post}.log`, `OUT\batch.csv` with per asset `gen_s, queue_wait_s, post_s,
asset_total_s, dump_mb, glb_sha256_masked, status`, plus a final `WALL_TOTAL` row (wall seconds, `assets_per_hour`).
Other flags: `--gen-gpu 1 --post-gpu 0 --res 1024 --seed 1 --faces 300000 --cli-extra "..." --replay-extra "..."`.

## Same-output rule and the PC check

Overlap changes only *when* post runs. Both modes feed the same generation command (same seed, same GPU) and the same
post-replay args; post thread count does not change output (hash-verified by pc-verify.ps1: 1 vs 24 threads vs legacy).
Residual risk: GPU QEM non-determinism, and generation on the 4070 being perturbed by concurrent load (unlikely to change
results; that is what the check measures). Check on the PC:

```
python tools\arctic-batch\arctic_batch.py compare D:\batch\serial D:\batch\overlap
```
It hashes every GLB with the `"generated"` timestamp masked and prints PASS/FAIL per asset. If FAIL only with `--qem gpu`,
repeat both runs with `--qem cpu` (and run `serial` twice to see whether serial-vs-serial already differs). Also
watch `nvidia-smi` on the 4070 during the overlap run: peak should stay at generation's ~11.1 GB with no QEM spike.

## Tests (no GPU)

`python -m unittest discover -s tools/arctic-batch/tests` uses a fake trellis-cli/post-replay (`tests/fake_engine.py`):
overlap faster than serial, identical masked GLBs, env (threads, `CUDA_VISIBLE_DEVICES`) per mode, failure isolation, compare.

## Parallel-processing options (claude/arctic-parallel)

- **`--dump-mode raw`**: the generation process runs `TRELLIS_DUMP_POST=<dump> TRELLIS_STOP_AFTER_POST_DUMP=1` (no `--dump-post`)
  and exits right after the texture decode, so weld / hole fill / BVH / remesh / clean / drop run in the post pool and
  `post-replay` runs the serial CLI's exact chain (no second remesh). The dump is the pre-weld mesh after `fill_holes`, about
  +0.25 GB over a cleaned dump. Do not pass `--no-remesh`, `--fill-holes`, `--no-weld` or `--no-fill` in `--replay-extra`
  (refused). Default stays `cleaned` until the PC check (T6) passes. Pass option values that start with `--` as
  `--replay-extra=--atlas`; a value with a space (`"--atlas 4096"`) works either way.
- **`--hash`**: `TRELLIS_POST_HASH=1` for generation and post. Fills `front_hash` (verts:faces hash after `drop_components`;
  compare with the serial CLI's `drop_components` line) and `dump_ok` (generation `fill_holes` hash = post-replay `input` hash;
  `NO` means the dump did not round-trip).
- **`batch.csv`** extra columns: `gen_start gen_end post_start post_end finish_start finish_end` (UTC, ms), `dump_mode`,
  `front_hash`, `dump_ok`, `warnings` (`post:cpu_qem` = GPU QEM fell back to the 15-25 min CPU path, `*:oom`,
  `finish:oom_retried`). `--fail-on-cpu-qem` makes the CPU fallback a failed post (default: a warning).
- **`batch_report.md`**: wall, assets/hour, generation-GPU busy %, idle gaps, "second post worker may help" when post is slower
  than the next generation on two or more props, a per-prop table, and a one-line verdict. Read this instead of the logs.
- **`--preflight`** (default on; `--no-preflight`): `nvidia-smi` memory.used must be below `--gen-gpu-max-gb` (3) on the
  generation GPU and `--post-gpu-max-gb` (1.5) on the post GPU (a resident ComfyUI or `llama-server` on the 3060 Ti would
  push QEM to the CPU). nvidia-smi missing = skipped. Exit code 4 when refused.
- **`--gpulock PATH_TO_gpulock.py`**: imports Expiramental3d's `tools/gpulock.py`, reserves both GPUs for `--lock-hours` (3),
  acquires gpu1 and gpu0 as `--lock-owner` (expiramental3d) with the runner's own pid, releases and unreserves in `finally`.
  Busy or reserved by a higher priority: exit 3, nothing started. Started under `gpulock run` instead? Pass
  `--gpulock-assume-held` (the locks are checked, not taken, and not released). `gpulock acquire` as a stand-alone command
  cannot hold a lock (its pid exits at once), which is why the runner imports the module.
- **`--phase-lock PATH`** (default `<lock dir>\gpu<post-gpu>.phase` with `--gpulock`): exported as `ARCTIC_GPU_PHASE_LOCK` to
  post-replay (around GPU QEM) and the finish command (around the Cycles bake, `finish.py`), never to generation, so the
  3060 Ti runs one of the two at a time. Unset = no lock.
- **`--remesh-mem-gib`**: `TRELLIS_REMESH_MEM_GIB` for post-replay; auto = 4 with one post worker (24 remesh workers instead of
  8), unset with several, `0` = the binary's default (1 GiB).
- **Finish retry**: a finish whose log shows an out-of-memory is retried once after `--finish-retry-wait` (30 s);
  `--finish-retries 0` turns it off.
- **Run order = argument order** for files (directories are sorted): pass the longest generation first and the cheapest
  post + finish last.

## Longest-first order (`--order longest-first`, opt-in)

Starts the slowest jobs first so both cards finish together (item 13 estimate: 60-115 s per batch). Needs `--est-from <earlier
batch.csv>`; without it, or with default `--order queue`, nothing changes. A job missing from the csv counts as the median one.
Only the start order changes: `batch.csv` rows keep the given order and each prop's output is hash-equal (tests/test_lanes.py).

## Live priority queue (`--queue-dir`, opt-in, needs `--gen-lanes 2`)

Add jobs while a batch runs: `arctic_batch.py submit --queue-dir Q --image X.png --priority 100`, end with `close` (or `--idle-exit S`).
Priority decides who starts next, never interrupts a running generation. Design, status and the Space Gothic slot: `docs/arctic/live-queue.md`.

## Card slot for Space Gothic (`--slot-yield`, opt-in, needs `--gen-lanes 2` and `--gpulock`)

When another owner runs `gpulock want --gpu N` (or reserves the card), the runner finishes the job on card N, releases it, and takes it back
after their run. No preemption. Design and rules: `docs/arctic/live-queue.md` (Phase 2).

## Second generation lane (`--gen-lanes 2`, opt-in)

Default `--gen-lanes 1` changes nothing. With `2`, the 3060 Ti (`--lane2-gpu`, must equal `--post-gpu`) generates as well as
the 4070; `dispatcher.py` decides who takes which job and admits post/finish by declared VRAM. Design, acceptance tests and
the go/no-go rule: Expiramental3d `research/notes/gen-lane2-design.md`.

```
python arctic_batch.py run <images...> -o out --gen-lanes 2 --est-from <earlier batch.csv> --dump-mode raw --hash --gpulock ...
```

- Lane B runs with `CUDA_VISIBLE_DEVICES=<lane2-gpu>` and `--gpu 0` (the 4070 is hidden). `--lane2-env KEY=VAL` adds and
  `--lane2-unset KEY` removes env for lane B only. Streaming off on the 3060 Ti (no desktop) is `--lane2-unset TRELLIS_CONVNEXT_STREAM_ALL`
  (the engine tests presence, so `=0` does NOT turn it off), plus `--lane2-env TRELLIS_CONVNEXT_STREAM_GB=6` to raise the 2 GB auto-stream threshold.
- Lane B takes a job only when it lowers the projected makespan by `--min-gain-s` (a big job is never left for the slow lane at
  the tail). `--est-from batch.csv` gives job sizes from an earlier run; without it sizes are unknown and the queue is split.
  The lane's speed is learned from its first finished job (`--lane2-speed` is the starting belief).
- VRAM: generation (`--gen-need-gb` 4.5) and the post/finish slot (`max(--post-need-gb 3.0, --finish-need-gb 2.0)`; QEM and bake
  share one slot because the phase lock serialises them) must fit `--gpu-cap-gb` minus the baseline and `--vram-margin-gb`.
  `--job-vram NAME=GB` marks a big prop (8.9 GB cabin-class stays on the 4070); `--pin NAME=A|B` forces a lane.
  Post/finish wait for room; after `--gate-timeout` they start anyway and the row gets `post:gate_timeout`.
- A generation that runs out of memory on lane B is re-queued once, pinned to lane A (`gen:oom` warning kept in `batch.csv`).
- `batch.csv` gains `gen_lane, gen_gpu, phase_gpu, gate_wait_s`. `--phase-placement flexible` (post/finish may use the 4070 when
  the 3060 Ti is full) stays off until the QEM cross-card check passes.
- What-if without a GPU: `python dispatcher.py sim --csv batch.csv --speed 0.5` and `python dispatcher.py threshold --csv batch.csv`.
- Locks are unchanged: the runner holds both GPU locks for the whole batch (Space Gothic stays out). Hand-off to Space Gothic
  between jobs is a later step.
