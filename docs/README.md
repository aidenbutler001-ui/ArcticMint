# Documentation index

| Where | What |
|-------|------|
| [`../README.md`](../README.md) | Overview, build, CLI options, what ArcticMint adds |
| [`getting-started.md`](getting-started.md) | Building and generating a first model; the inherited installer and desktop app |
| [`../ROADMAP.md`](../ROADMAP.md) | Milestones of the port and ArcticMint's own work |
| [`../NOTICE.md`](../NOTICE.md) | Credits and third-party licences |

## ArcticMint notes: [`arctic/`](arctic/)

| Doc | Topic |
|-----|-------|
| [`post-mt.md`](arctic/post-mt.md) | Multithreaded CPU post-processing, why its output is byte-identical, measurements |
| [`post-mt-golden.txt`](arctic/post-mt-golden.txt) | Golden stage hashes used by the equivalence test |
| [`ss-only.md`](arctic/ss-only.md) | `--ss-only`: screening seeds by structure before the expensive stages |
| [`live-queue.md`](arctic/live-queue.md) | Live priority queue for the batch runner (opt-in) |
| [`pr-history.md`](arctic/pr-history.md) | Descriptions of pull requests merged in the earlier fork |

Tool READMEs: [`arctic-batch`](../tools/arctic-batch/README.md), [`arctic-post`](../tools/arctic-post/README.md),
[`mv_preview`](../tools/mv_preview/README.md).

## Engine specification: [`spec/`](spec/)

Inherited reverse-engineering notes on the TRELLIS.2 architecture and the C++ port (pipeline, samplers, DiT, sparse ops,
decoders, mesh/GLB export, divergence matrix, perf profile). Start with [`00-synthesis.md`](spec/00-synthesis.md) and
[`01-pipeline.md`](spec/01-pipeline.md); the performance profile is [`29-perf-profile.md`](spec/29-perf-profile.md).

## Pixal3D: [`pixal3d/`](pixal3d/README.md)

The optional `--model pixal3d` backend and its known gaps.

## Inherited, untested here

[`../app/`](../app/README.md) (desktop app), [`../install/`](../install/portable-readme.txt) (installers, portable build) and
[`../assets/showcase/`](../assets/showcase/README.md) come from upstream.
