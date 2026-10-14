# arctic-batch

Overlap GPU generation with CPU post-processing for batches of images. Part of ArcticMint; Python standard library only.

> **Status.** Developed and unit-tested with a fake engine (no GPU needed). Real-hardware checks are called out
> where they are still pending; the opt-in features below are not defaults until those pass.

One asset is roughly 150 s of GPU generation followed by about 65 s of CPU post on 24 threads, and each side idles while the
other works. `arctic_batch.py` keeps the generation card busy back to back and post-processes finished dumps in a pool.

## Contents

- [How it works](#how-it-works)
- [Running a batch](#running-a-batch)
- [Same-output rule and the check](#same-output-rule-and-the-check)
- [Options](#options)
- [Opt-in scheduling features](#opt-in-scheduling-features)
- [Tests](#tests)

## How it works

| Part | What it does |
|------|--------------|
| **GPU worker** | `trellis-cli -i IMG -o DUMPDIR/NAME.unused.glb -m MODELS --res 1024 --gpu 1 --seed S --dump-post DUMPDIR/NAME.bin` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. `--dump-post` runs weld, hole fill, DC remesh, clean and floater removal, writes the already-remeshed mesh plus the sparse PBR volume, and exits (no decimate, UV, bake or GLB). |
| **Post pool** | `post-replay DUMP OUT.glb --geometry-res 1024 --band 2 --faces 300000 --no-audit`, with `TRELLIS_POST_THREADS = total_threads / N`. `--post-workers` defaults to 1: a second worker mostly contends for the second GPU and the CPU, so raise it only if `batch_report.md` says it may help. |
| **QEM device** | `post-replay` never runs ggml, so its GPU QEM would use CUDA device 0. The runner sets `CUDA_VISIBLE_DEVICES=<--post-gpu>` (default `0`, the 3060 Ti in PCI order) to keep the 4070 SUPER free. `--qem cpu` forces the CPU path, which is slower; GPU QEM is not deterministic run to run, so CPU and GPU QEM GLBs can differ. |
| **Backpressure** | At most `--max-queue` (2) finished dumps wait for a post worker (dumps are 0.3 to 0.6 GB); generation blocks beyond that. Dumps are deleted after a successful post unless `--keep-dumps`. |
| **`--serial`** | Generate then post, one asset at a time, post getting all `--total-threads`. Used as the A/B baseline. |

`post-replay` re-remeshes the cleaned dump, as `pc-verify.ps1 -DumpMode Cleaned` does. It is the same input in serial and
overlap mode, so the comparison is fair.

## Running a batch

From the repo root, after building `trellis-cli` and `post-replay` (PowerShell shown; the Python entry point works anywhere):

```powershell
$b = "build"   # wherever trellis-cli(.exe) and post-replay(.exe) live
powershell -ExecutionPolicy Bypass -File tools\arctic-batch\arctic-batch.ps1 run images -o out\overlap `
  --cli $b\trellis-cli.exe --replay $b\post-replay.exe -m path\to\gguf --post-workers 1 --total-threads 24
powershell -ExecutionPolicy Bypass -File tools\arctic-batch\arctic-batch.ps1 run images -o out\serial --serial `
  --cli $b\trellis-cli.exe --replay $b\post-replay.exe -m path\to\gguf
```

```bash
python tools/arctic-batch/arctic_batch.py run images/ -o out/overlap --cli build/trellis-cli --replay build/post-replay -m /path/to/gguf
```

Output: `OUT/*.glb`, `OUT/logs/NAME.{gen,post}.log`, and `OUT/batch.csv` with per asset `gen_s, queue_wait_s, post_s,
asset_total_s, dump_mb, glb_sha256_masked, status`, plus a final `WALL_TOTAL` row (wall seconds, `assets_per_hour`).
Other flags: `--gen-gpu 1 --post-gpu 0 --res 1024 --seed 1 --faces 300000 --cli-extra "..." --replay-extra "..."`.

## Same-output rule and the check

Overlap changes only *when* post runs. Both modes use the same generation command (seed, GPU) and the same `post-replay`
arguments, and post thread count does not change output (hash-verified by `pc-verify.ps1`: 1 vs 24 threads vs legacy).
Residual risks are GPU QEM non-determinism and generation on the 4070 being perturbed by concurrent load. To check:

```bash
python tools/arctic-batch/arctic_batch.py compare out/serial out/overlap
```

`compare` hashes every GLB with the `"generated"` timestamp masked and prints PASS or FAIL per asset. If it fails only with
GPU QEM, repeat both runs with `--qem cpu` (and run `serial` twice to see whether serial-vs-serial already differs). Watch
`nvidia-smi` on the 4070 during the overlap run: peak should stay at generation's ~11.1 GB with no QEM spike.

## Options

### Reporting and safety

- **`batch.csv`** extra columns: `gen_start gen_end post_start post_end finish_start finish_end` (UTC, ms), `dump_mode`,
  `front_hash`, `dump_ok`, `warnings` (`post:cpu_qem` = GPU QEM fell back to the 15 to 25 min CPU path, `*:oom`,
  `finish:oom_retried`). `--fail-on-cpu-qem` makes the CPU fallback a failed post (default: a warning).
- **`batch_report.md`**: wall time, assets/hour, generation-GPU busy %, idle gaps, a "second post worker may help" note when
  post is slower than the next generation on two or more props, a per-prop table, and a one-line verdict. Read this before the logs.
- **`--hash`**: sets `TRELLIS_POST_HASH=1` for generation and post. Fills `front_hash` (verts:faces hash after
  `drop_components`) and `dump_ok` (generation `fill_holes` hash equals the `post-replay` `input` hash; `NO` means the dump
  did not round-trip).
- **`--preflight`** (default on; `--no-preflight` to skip): `nvidia-smi` memory.used must be below `--gen-gpu-max-gb` (3) on the
  generation GPU and `--post-gpu-max-gb` (1.5) on the post GPU, since another resident GPU process would push QEM to the CPU.
  Skipped if `nvidia-smi` is missing. Exit code 4 when refused.
- **Finish retry**: a finish step whose log shows an out-of-memory is retried once after `--finish-retry-wait` (30 s);
  `--finish-retries 0` turns it off.
- **`--remesh-mem-gib`**: sets `TRELLIS_REMESH_MEM_GIB` for `post-replay`. Auto is 4 with one post worker (24 remesh workers
  instead of 8), unset with several, `0` uses the binary's default (1 GiB).

### Dump mode

**`--dump-mode raw`** runs generation with `TRELLIS_DUMP_POST=<dump> TRELLIS_STOP_AFTER_POST_DUMP=1` (no `--dump-post`) and
exits right after the texture decode. Weld, hole fill, BVH, remesh, clean and drop then run in the post pool, so
`post-replay` follows the serial CLI's exact chain with no second remesh. The dump is the pre-weld mesh after `fill_holes`,
about 0.25 GB larger than a cleaned dump. Do not pass `--no-remesh`, `--fill-holes`, `--no-weld` or `--no-fill` in
`--replay-extra` (refused). The default stays `cleaned` until the PC check passes. Pass option values that start with `--` as
`--replay-extra=--atlas`; a value with a space (`"--atlas 4096"`) works either way.

### Run order

Run order is argument order for files (directories are sorted): pass the longest generation first and the cheapest post last.

## Opt-in scheduling features

All of these are off by default and change nothing unless enabled.

### Longest-first order (`--order longest-first`)

Starts the slowest jobs first so both cards finish together (estimated saving 60 to 115 s per batch). Needs `--est-from
<earlier batch.csv>`; without it, or with the default `--order queue`, nothing changes. A job missing from the csv counts as
the median one. Only the start order changes: `batch.csv` rows keep the given order and each prop's output is hash-equal
(`tests/test_lanes.py`).

### Second generation lane (`--gen-lanes 2`)

With `2`, the second card (`--lane2-gpu`, must equal `--post-gpu`) generates as well as the 4070; `dispatcher.py` decides who
takes which job and admits post work by declared VRAM.

```bash
python arctic_batch.py run <images...> -o out --gen-lanes 2 --est-from <earlier batch.csv> --dump-mode raw --hash
```

- Lane B runs with `CUDA_VISIBLE_DEVICES=<lane2-gpu>` and `--gpu 0`. `--lane2-env KEY=VAL` adds and `--lane2-unset KEY` removes
  environment for lane B only. To turn convnext streaming off on a card with no desktop attached use
  `--lane2-unset TRELLIS_CONVNEXT_STREAM_ALL` (the engine tests presence, so `=0` does not turn it off), plus
  `--lane2-env TRELLIS_CONVNEXT_STREAM_GB=6` to raise the 2 GB auto-stream threshold.
- Lane B takes a job only when it lowers the projected makespan by `--min-gain-s`, so a big job is never left for the slow lane
  at the tail. `--est-from batch.csv` supplies job sizes; without it sizes are unknown and the queue is split. The lane's speed
  is learned from its first finished job (`--lane2-speed` is the starting belief).
- VRAM: generation (`--gen-need-gb` 4.5) and the post/finish slot (`max(--post-need-gb 3.0, --finish-need-gb 2.0)`) must fit
  `--gpu-cap-gb` minus the baseline and `--vram-margin-gb`. `--job-vram NAME=GB` marks a big prop; `--pin NAME=A|B` forces a
  lane. Post waits for room; after `--gate-timeout` it starts anyway and the row gets `post:gate_timeout`.
- A generation that runs out of memory on lane B is re-queued once, pinned to lane A (`gen:oom` stays in `batch.csv`).
- `batch.csv` gains `gen_lane, gen_gpu, phase_gpu, gate_wait_s`. `--phase-placement flexible` (post may use the 4070 when the
  other card is full) stays off until the QEM cross-card check passes.
- What-if without a GPU: `python dispatcher.py sim --csv batch.csv --speed 0.5` and `python dispatcher.py threshold --csv batch.csv`.

### Live priority queue (`--queue-dir`, needs `--gen-lanes 2`)

Add jobs while a batch runs with `arctic_batch.py submit --queue-dir Q --image X.png --priority 100`, and end with `close` (or
`--idle-exit S`). Priority decides who starts next and never interrupts a running generation. Design and status:
[`docs/arctic/live-queue.md`](../../docs/arctic/live-queue.md).

### Optional GPU lock (`--gpulock`, `--phase-lock`, `--slot-yield`)

For machines where other tools also use the GPUs. `--gpulock PATH_TO_gpulock.py` imports a cooperative lock module that you
supply, reserves both GPUs for `--lock-hours` (3), and acquires them as `--lock-owner` using the runner's own pid; it releases
in `finally`. If the GPUs are busy or reserved by a higher priority, the runner exits with code 3 and starts nothing. If the
runner is already started under that tool's own `run` wrapper, pass `--gpulock-assume-held`. A stand-alone `acquire` cannot
hold a lock (its pid exits at once), which is why the runner imports the module.

- **`--phase-lock PATH`**: exported as `ARCTIC_GPU_PHASE_LOCK` to `post-replay` (around GPU QEM) and the finish command
  (around the Cycles bake, `finish.py`), never to generation, so the second card runs one of the two at a time. Unset means no lock.
- **`--slot-yield`** (needs `--gen-lanes 2` and `--gpulock`): when another lock owner asks for a card, the runner finishes its
  current job on that card, releases it, and takes it back afterwards. No preemption. Implemented
  (needs a lock module with `want` and `foreign_demand`); see
  [`docs/arctic/live-queue.md`](../../docs/arctic/live-queue.md) (Phase 2).

`finish.py` and the lock module are not in this repository.

## Tests

No GPU needed. `python -m unittest discover -s tools/arctic-batch/tests` uses a fake `trellis-cli` and `post-replay`
(`tests/fake_engine.py`) and checks: overlap faster than serial, identical masked GLBs, per-mode environment (threads,
`CUDA_VISIBLE_DEVICES`), failure isolation, and `compare`.

Draft-then-refine seed picking lives in [`draft_refine.py`](draft_refine.py); see the top-level README.
