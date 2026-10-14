# Roadmap

ArcticMint is a C++/ggml port of TRELLIS.2-4B image-to-3D. The engine was built bottom-up, validating each neural
component against PyTorch reference tensors before composing the full pipeline.

Milestones **M0 to M6 are the inherited engine history** (from [pwilkin/trellis.cpp](https://github.com/pwilkin/trellis.cpp);
see the [README](README.md#what-arcticmint-adds) for what ArcticMint adds on top). Items below that mention other GPUs
come from upstream's testing, not ours. ArcticMint's own work is under [Beyond the port](#beyond-the-port).

Legend: ☐ todo · ◐ in progress · ☑ done

## Status

Working end-to-end: UV-textured PBR GLB via the **1024 cascade by default**. `trellis-cli -i <image.png> -o <out.glb>` runs it;
`--res 512` selects the lighter res-512 path. The cascade is enabled by **FlashAttention with padded K/V** (`src/dit.cpp::sdpa`): it fits
a 16 GB card (the manual softmax wanted a single ~18 GB score-matrix alloc at the
sparse-structure stage) and the HR flow no longer NaNs (ggml's tiled CUDA FA poisoned the unpadded
last key-tile at the HR token count, ≈53k). BiRefNet bg-removal is done (M2); only polish remains (M6).

## M0 — Foundations
- ☑ Reverse-engineer architecture → `docs/spec/` (23 docs + IMPL_NOTES)
- ☑ Download models (flows 512+1024, decoders, DINOv3, BiRefNet weights)
- ☑ Vendor GGML (`thirdparty/ggml`, CUDA, sm 86+120) + top-level CMake
- ☑ Python venv (numpy safetensors gguf pillow torch timm) for conversion + reference checks
- ☑ safetensors→GGUF converter (`tools/convert.py`) with per-model key remap + 5D-conv reshapes
- ☑ GGUF loader + smoke test

## M1 — Shared GGML primitives
- ☑ DiT blocks: timestep embed, AdaLN (share_mod) modulation, GELU-tanh MLP
- ☑ Attention: QKV, QK-RMSNorm, 3D interleaved-pair RoPE (dense + voxel-coord); SDPA via
      **FlashAttention** (`ggml_flash_attn_ext`, BF16 K/V zero-padded to a 256-tile multiple) —
      O(N) memory and NaN-free at the cascade's HR token count; `TRELLIS_NOFA=1` forces manual softmax
- ☑ Cross-attention to image-condition tokens (affine norm2)
- ☑ FlowEuler guidance-interval sampler (rescale_t warp, CFG interval, guidance_rescale)
- ☑ SparseTensor (coords[N,4], feats[N,C]) + serialize order
- ☑ Sparse ops: linear, norm, act, ConvNeXt, C2S subdivision (spatial↔channel)
- ☑ Sparse 3×3×3 submanifold conv (coord-hashmap neighbor gather)

## M2 — Image front-end
- ☑ DINOv3 ViT-L/16 in GGML + preprocessing → conditioning tokens (rel 1.8e-2 vs timm)
- ☑ BiRefNet background removal in GGML → alpha matte (Swin-L backbone + ASPPDeformable decoder +
      custom deformable-conv CUDA kernel; full pipeline matches PyTorch — logits rel 8e-3, fully on GPU,
      ~13s). Selected with `--bg-removal birefnet`; the default is automatic (pre-matted images keep their alpha, otherwise BiRefNet). (Swin window-pad uses a host zero-column since ggml_pad's
      CUDA grid dim = N+1 = 65537 exceeds the 65535 limit at res-256/stage-0.)
- ☑ Text→image entry point via stable-diffusion.cpp (Z-Image), upstream-side tooling; not part of ArcticMint's own use

## M3 — Stage ① Sparse Structure
- ☑ SparseStructureFlowModel (dense 16³ DiT) forward + full sampling (rel 4.3e-3, exact voxels)
- ☑ SS conv3d decoder → 64³ occupancy → active voxel extraction (max-pool to ss_res)

## M4 — Stage ② Shape
- ☑ SLatFlowModel (sparse DiT) img2shape forward + sampling → 32-ch shape latent
- ☑ FlexiDualGrid decoder (sparse ConvNeXt + C2S) → dual grid → mesh (CPU geometry)
- ☑ Export untextured mesh to GLB  ← MVP milestone hit

## M5 — Stage ③ Texture + finish
- ☑ SLatFlowModel imgshape2tex forward + sampling → texture latent (concat shape slat)
- ☑ Sparse U-Net tex decoder → 6-ch PBR per voxel (replays shape decoder's subdiv masks)
- ☑ Bake PBR onto mesh — reference-parity postprocess (spec 27/28): weld → fill →
      **narrow-band UDF dual-contouring remesh** → quadric simplify (meshopt) → bottom-up
      normal-cone cluster merge (cumesh port) → stock xatlas per cluster, auto-resolution pack,
      UVs normalized to the full atlas → trilinear voxel-PBR texel shading with BVH surface snap →
      Telea inpaint → GLB with smooth normals + lossy-WebP textures (`EXT_texture_webp`).
      `--box-uv` keeps the O(F) 6-way box projection as the fast path
- ☑ 512→1024 cascade (LR flow_512 → upsample → quantize res64 → HR flow_1024 → decode res1024);
      needed the FlashAttention + padded-K/V fix (M1) to fit VRAM and keep the HR flow finite

## M6 — Productionize
- ☑ `trellis-cli` driver (`-i`, `-o`, `-m`, `--gpu`, `--seed`; per-stage model load/free)
- ☑ Multi-GPU placement (`--gpu N`), f16 default (`TRELLIS_F32` for f32)
- ☑ Docs + parity test suite (`trellis-test-*` binaries + `tools/ref_*.py`; postprocess
      validated against the reference CUDA `to_glb` run on identical dumped inputs — visual
      parity + `tools/glb_metrics.py` at-or-above-parity texel density, spec 28 addenda)
- ☑ Polish: quadric decimation, mesh fill_holes, degenerate-face-safe unwrap
- ☑ Perf profile + backend comparison (upstream's measurements, spec 29): Vulkan fastest on Strix Halo; CUDA build
      ~2.9x on discrete Blackwell; ROCm needs `GGML_CUDA_DISABLE_GRAPHS=1` and still trails
- ☐ HR-token-count attention scaling (~45k tokens): ggml FA trails flash-attn-varlen ~1.4x
      on same-GPU heavy inputs — upstream ggml or varlen-style path

## Beyond the port
ArcticMint-specific work. Notes in [`docs/arctic/`](docs/arctic/).

- ☑ Multithreaded CPU post-processing, byte-identical to the single-thread path ([`post-mt.md`](docs/arctic/post-mt.md))
- ☑ Standalone post-processing build and synthetic dump generator ([`tools/arctic-post`](tools/arctic-post/README.md))
- ☑ Parallel batch runner, with `compare` for hash-equal output ([`tools/arctic-batch`](tools/arctic-batch/README.md))
- ☑ `--ss-only` structure screening ([`ss-only.md`](docs/arctic/ss-only.md))
- ◐ Opt-in second generation lane, live queue and draft-then-refine: built and unit-tested, awaiting runs on real hardware ([`live-queue.md`](docs/arctic/live-queue.md))
- ◐ `tools/arctic-post/pc-verify.ps1`: written, never run on Windows/NVIDIA

## Validation strategy
The full reference environment needs CUDA-only extensions (o-voxel, flash-attn, spconv, nvdiffrast) that
don't all install everywhere, so we validate **per-component**: dump reference outputs for each module
(DiT block, conv3d SS decoder, DINOv3, sparse conv, C2S) with safetensors+torch on GPU, and
assert the C++/GGML output matches. Geometry is validated by shape stats + visual inspection.
Stochastic e2e won't bit-match; same-noise per-stage parity is the bar.
