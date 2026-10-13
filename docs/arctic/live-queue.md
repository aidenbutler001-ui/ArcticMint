# Live priority queue for arctic-batch (step 3 of the workflow plan) - design and status

**Status:** phase 1 built and tested with the fake engine; **opt-in** (`--queue-dir`), default runs are unchanged. It is not the
default until a PC run on real props passes and the output hashes match the plain run. Phase 2 (the Space Gothic slot) is design only.

## Why
The batch list is fixed at start. The 4070 idles at the end of every batch until the next one is started by hand (item 13: ~200 s per
batch boundary), and an urgent prop cannot jump ahead of a long list. About half of the scheduler exists (two lanes, VRAM gate, pins,
`Job.priority` in `dispatcher.py`, which nothing could set). This adds the missing half without a server or a socket.

## Phase 1 (this PR): spool directory + priority
```
python arctic_batch.py run [images...] -o OUT --gen-lanes 2 --queue-dir Q [--idle-exit 600] [--order longest-first] ...
python arctic_batch.py submit --queue-dir Q --image X.png [--priority 100] [--est-s 600] [--vram-gb 4.5] [--pin A|B] [--name N]
python arctic_batch.py close  --queue-dir Q
```
- A job is a file `Q/<time>-<name>.job.json`, written as `.tmp` then renamed (never half-read). The lane workers already wake every 2 s;
  each wake claims new files by moving them to `Q/claimed/` (exactly one runner takes a job) and adds them to the pending list.
  Bad JSON, a missing image or a duplicate name (case-insensitive, like the batch) goes to `Q/rejected/` with a `.why.txt`.
- **Priority:** higher starts first (default 0, urgent 100). Equal priority = arrival order, or slowest-first with `--order longest-first`
  (the dispatcher now does this itself, so late jobs sort in too). Priority decides who **starts** next. A running generation is never
  preempted, so a job's output cannot depend on when it was submitted.
- **End:** `close` (finish what is queued, then exit) or `--idle-exit S` (also exit after S s with nothing to do). Without either, the
  runner waits for more jobs. Images given on the command line are the first jobs; the images argument is optional with `--queue-dir`.
- `batch.csv` and the report list the command-line jobs first, then the live ones in arrival order.
- Same-output rule: ordering and queueing only change *when* a job starts. Tested with the fake engine: the same six pictures run as a
  plain batch and as a live batch (three submitted mid-run, one urgent) give hash-equal GLBs (`cmd_compare`).
- GPU locks are unchanged (both cards are reserved and locked for the whole live session). The post/finish pipeline is untouched.

## Phase 2 (not built): the Space Gothic slot
Aim: between jobs, lane B (or A) is handed to Space Gothic instead of being held for the whole session (item 13 estimate: ~200 s per
boundary). Needs SG's agreement on how they ask (nothing in `gpulock.py` today), so no code yet. Proposal for the SG lead: a
`gpu<N>.wanted` file (JSON owner, what, until) in the lock directory, same rules as `.reserved`. When a lane finishes a job and a `.wanted`
for its card exists, and that card's post/finish work is idle (VramGate shows nothing running), the runner unreserves and releases that
card, waits until SG's lock is gone, then reserves and acquires again and carries on. No preemption, and SG waits at most one job.
Open question for the lead: lane B's card is also the post/finish card, so yielding it also pauses post/finish; is that acceptable?

## PC test plan (needed before it is a default)
One run on the PC after SG's window: two small props from the command line, a third submitted mid-run at priority 100, `close`; check
it starts before the second queued prop, the three GLBs hash-equal to the plain batch of the same pictures, and `Q/claimed/` holds 1 file.
