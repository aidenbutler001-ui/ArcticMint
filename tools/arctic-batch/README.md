# arctic-batch: overlap GPU generation with CPU post (ArcticMint, experimental branch)

One asset today = GPU generation (~150 s) then CPU post (~65 s @24 threads); each side idles during the other.
`arctic_batch.py` (Python stdlib only) runs the 4070 SUPER back to back and post-processes finished dumps in a pool.

- **GPU worker**: `trellis-cli -i IMG -o DUMPDIR\NAME.unused.glb -m MODELS --res 1024 --gpu 1 --seed S --dump-post DUMPDIR\NAME.bin`
  with `CUDA_DEVICE_ORDER=PCI_BUS_ID`. `--dump-post` runs the CLI's weld -> fill_small_holes -> DC remesh -> clean -> drop
  floaters, writes the *already remeshed* mesh + sparse PBR volume and exits (no decimate/UV/bake/GLB).
- **Post pool** (N workers, default 2): `post-replay DUMP OUT.glb --geometry-res 1024 --band 2 --faces 300000 --no-audit`
  with `TRELLIS_POST_THREADS = total_threads / N` (default 24/2 = 12). Note post-replay re-remeshes the cleaned dump
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
  --cli $b\trellis-cli.exe --replay $b\post-replay.exe -m C:\AI\trellis\models --post-workers 2 --total-threads 24
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
