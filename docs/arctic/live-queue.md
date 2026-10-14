# Live priority queue for arctic-batch: design and status

**Status:** phase 1 built and tested with the fake engine; **opt-in** (`--queue-dir`), default runs are unchanged. It is not the
default until a PC run on real props passes and the output hashes match the plain run. Phase 2 (`--slot-yield`, yielding a card to another GPU user) is also implemented and opt-in.

## Why
The batch list is fixed at start. The 4070 idles at the end of every batch until the next one is started by hand (an estimated ~200 s per
batch boundary), and an urgent prop cannot jump ahead of a long list. About half of the scheduler exists (two lanes, VRAM gate, pins,
`Job.priority` in `dispatcher.py`, which nothing could set). This adds the missing half without a server or a socket.

## Phase 1 (built): spool directory + priority
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

## Phase 2 (built, `--slot-yield`): yielding a card to another GPU user
Aim: between jobs, a lane's card is handed to another program that needs it instead of being held for the whole session
(estimated ~200 s per boundary). Needs `--gen-lanes 2` and `--gpulock`, and the runner must hold the locks itself (not `--gpulock-assume-held`).
As built (`slot_loop` in `arctic_batch.py`, polling every 2 s): the other program announces demand with the lock module's `want`, or reserves the card.
The runner asks the module for `foreign_demand(card, owner)`.
1. **Pause.** On demand the card is paused: no new job (generation, post or finish) is placed on it. Running work is not interrupted.
   If the demand disappears before the card is released, the pause is lifted.
2. **Release.** When nothing of ours is running on the card, the runner unreserves and releases it (`yield_card`) and logs the hand-over.
   No preemption: the other program waits for the work already on the card to finish, which can be a generation, a post or a finish.
3. **Re-acquire.** The runner watches the lock file. Once the wanting owner has held the card and released it, a leftover `want` is
   cleared with `unwant`; with no demand left, the runner reserves and acquires the card again without waiting (`retake_card`) and
   lifts the pause. If the other owner got there first, it retries on the next poll.
The lock module must provide `foreign_demand` (the runner exits otherwise).

Original design, superseded: a `gpu<N>.wanted` file (JSON owner, what, until) in the lock directory, same rules as `.reserved`. The code
uses the lock module's want/foreign_demand calls instead and does not read such a file.
Open question: lane B's card is also the post/finish card, so yielding it also pauses post/finish; is that acceptable?

## Hardware test plan (needed before it is a default)
One run on the 4070 SUPER + 3060 Ti machine: two small props from the command line, a third submitted mid-run at priority 100, `close`; check
it starts before the second queued prop, the three GLBs hash-equal to the plain batch of the same pictures, and `Q/claimed/` holds 1 file.
