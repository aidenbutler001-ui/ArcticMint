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

## Phase 2 (design only, not built): the Space Gothic slot
Aim: at a job boundary a card is handed to Space Gothic (SG) instead of being held for the whole session (~200 s per boundary, item 13).
Nothing here replaces Aiden's rule: SG still needs his OK in their project before they run anything. The slot is only the *mechanism* by
which a run that is already approved gets the card without a human stopping our batch. Status: waiting for Aiden's answer in SG's project.

**Why not a plain `reserve` by SG.** `gpulock.reserve()` refuses a lower-or-equal priority owner while another owner's reservation is
active (`pri <= existing`), and arctic-batch always holds one (`--lock-hours`, default 3; priority 100 vs SG's 50). So SG's `reserve` would
be refused for the whole session, and even if it were stored it would only block lower priorities, which does not reach us. The request
needs its own file.

**Answer 1: what the call looks like.** A new verb in both tools, same directory and JSON style as the rest:
```
python tools\gpulock.py want   --gpu 0 --owner space-gothic --minutes 40 --what "ray-tracing measurement"   # writes gpu0.wanted
python tools\gpulock.py unwant --gpu 0 --owner space-gothic                                                   # deletes it
tools\gpulock.ps1 want -Gpu 0 -Owner space-gothic -Minutes 40 -What "..."       # same in PowerShell (SG's side)
```
`gpu<N>.wanted` = JSON `{owner, what, until_utc, priority}`. `until_utc` caps how long the runner will look ahead (default 40 min, max 120).
After `want`, SG polls `status`, `acquire --wait` as today; the lock itself is still taken with `gpulock run` for the child's lifetime, so a
crash frees it. `unwant` is optional: the file also expires at `until_utc` and is deleted by the runner after SG's lock has come and gone.

**What the runner does** (at a job boundary of the lane on that card, never mid-job; priority 100 does not override this):
1. See `gpu<N>.wanted` from another owner (or an active reservation of another owner on its card).
2. Stop starting jobs on that lane. Wait until nothing of ours runs on that card (VramGate empty: the lane's generation, and post/finish
   when it is the post card, i.e. lane B's 3060 Ti), at most one job + one post/finish (~4-6 min worst case, so SG waits that long at most).
3. `unreserve` and `release` that card's lock (we hold it as our own pid), then wait for SG's lock to appear and go.
4. When the lock is free and the `.wanted` file is gone or expired: `reserve` + `acquire` again, resume. The other lane keeps working
   meanwhile (a batch with one card runs at one-lane speed; the other card's post/finish is unchanged).
If SG's lock is still held after `until_utc`, we keep waiting: **no preemption in either direction**, as today.

**Answer 2: both projects want the same card.**
- `want` is first come, first served per card. A second `want` from another owner is refused (exit 3) and prints who holds the slot
  and until when; the same owner may renew. The runner never writes `.wanted` for itself (it already has the card).
- Reservations keep today's rules unchanged (priority 100 vs 50): if SG reserves a card while we hold no reservation (a batch run with
  `--lock-hours 0`), the runner treats that active foreign reservation exactly like a `.wanted` and yields at the next boundary.
- If SG `want`s a card while our lock is held by a *non*-slot-aware process (a single `gpulock run`, e.g. a PC sheet), nothing changes:
  they wait for that lock like any other.
- A stale `.wanted` (expired) is ignored and removed by whoever sees it first, like a stale lock.

**Small code change when built:** `want`/`unwant` in gpulock.py and gpulock.ps1 (+ tests in test_gpulock), `status` prints the
`wanted:` line, the runner learns `yield_card()` (about 60 lines in `run_lanes`, tested with the fake gpulock and fake engine: yield at the
boundary, resume after, output hash-equal).

## PC test plan (needed before it is a default)
One run on the PC after SG's window: two small props from the command line, a third submitted mid-run at priority 100, `close`; check
it starts before the second queued prop, the three GLBs hash-equal to the plain batch of the same pictures, and `Q/claimed/` holds 1 file.
