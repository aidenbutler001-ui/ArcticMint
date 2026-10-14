# Notice: credits and third-party licences

ArcticMint is a derivative work and builds on the projects below. This file lists them for convenience; it is not legal advice.
Licences below were checked against the upstream licence files or model cards on 2026-10-14 (the image-model rows through the Hugging Face and GitHub pages, without reading the LICENSE files of the Comfy-Org repacks); the source is linked in each row.
[`LICENSE`](LICENSE) and upstream copyright headers are unchanged.

| Component | Author | Licence | Notes |
|-----------|--------|---------|-------|
| ArcticMint / [trellis.cpp](https://github.com/pwilkin/trellis.cpp) | Piotr Wilkin (Copyright (c) 2026) | MIT | Full text in [`LICENSE`](LICENSE). The engine, model ports and most of the CLI come from this work. |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [`TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) weights | Microsoft | MIT | Code: [`LICENSE`](https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE) is MIT, Copyright (c) Microsoft Corporation. Weights: the [model card](https://huggingface.co/microsoft/TRELLIS.2-4B) says "This model is released under the MIT License" (metadata `mit`). The TRELLIS.2 README notes that nvdiffrast and nvdiffrec have their own licences; ArcticMint does not use them. |
| [ggml](https://github.com/ggml-org/ggml) | The ggml authors | MIT | [`LICENSE`](https://github.com/ggml-org/ggml/blob/master/LICENSE): MIT, Copyright (c) 2023-2026 The ggml authors. Used as the git submodule `thirdparty/ggml`, which points at a fork. |
| [DINOv3](https://github.com/facebookresearch/dinov3) ViT-L/16 | Meta | DINOv3 License (not MIT) | [`LICENSE.md`](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) (last updated Aug 19, 2025). See below. |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | ZhengPeng7 and co-authors | MIT | Code: [`LICENSE`](https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE) is MIT, Copyright (c) 2024 ZhengPeng. The [model card](https://huggingface.co/ZhengPeng7/BiRefNet) metadata also says `mit`. Used for background removal. |
| [xatlas](https://github.com/jpcy/xatlas) | Jonathan Young | MIT | Vendored in `thirdparty/xatlas`. |
| [stb](https://github.com/nothings/stb) (`stb_image`, `stb_image_write`, `stb_image_resize`) | Sean Barrett and contributors | Public domain | Vendored in `thirdparty/stb`. |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | Yuji Hirose | MIT | Vendored in `thirdparty/cpp-httplib`; used by `trellis-server`. |
| fqms (Fast-Quadric-Mesh-Simplification) | Spacerat and contributors | MIT | Vendored in `thirdparty/fqms`. |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | Arseny Kapoulkine | MIT | Vendored in `thirdparty/meshoptimizer`. |
| [`<model-viewer>`](https://github.com/google/model-viewer) (bundles three.js) | Google and contributors, three.js authors | Mixed: Apache-2.0, BSD-3-Clause, MIT | The upstream [`LICENSE`](https://github.com/google/model-viewer/blob/master/LICENSE) is Apache-2.0. The vendored minified bundle opens with `SPDX-License-Identifier: BSD-3-Clause` (Copyright Google LLC) and also carries many Apache-2.0 and some MIT notices (three.js is [MIT](https://github.com/mrdoob/three.js/blob/dev/LICENSE), Copyright three.js authors). Vendored in `app/public/vendor/` and `tools/mv_preview/` (identical files). |
| [Pixal3D](https://github.com/TencentARC/Pixal3D) | TencentARC | MIT | Code: [`LICENSE`](https://github.com/TencentARC/Pixal3D/blob/master/LICENSE) is MIT, Copyright (c) 2026 Tencent; the README adds that third-party components keep their own terms. Weights: the [`TencentARC/Pixal3D`](https://huggingface.co/TencentARC/Pixal3D) card says `mit`. Optional; run by `--model pixal3d`. |
| [FLUX.2 \[klein\] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8) (fp8) | Black Forest Labs | Apache-2.0 | Optional text-to-image model for [`tools/prompt-to-model/`](tools/prompt-to-model/README.md); not bundled, you download it. The model card metadata says `apache-2.0` and links the Apache-2.0 text. It asks that modified versions do not use BFL or FLUX trademarks in a confusing way, lists out-of-scope uses (unlawful content, non-consensual imagery and similar) and encourages deployers to filter inputs and outputs. The 9B and dev models are under a non-commercial licence and are not used here. |
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI | Apache-2.0 | Optional alternative text-to-image model for the same tool; not bundled. Model card metadata: `apache-2.0`. The int8 ComfyUI repack named in the workflow comes from [`Comfy-Org/z_image_turbo`](https://huggingface.co/Comfy-Org/z_image_turbo), also `apache-2.0` on its card, as are its text encoder and `ae.safetensors` VAE. |
| Qwen3-4B text encoder and FLUX.2 VAE (ComfyUI repacks) | Qwen team; Black Forest Labs | Apache-2.0 | The files `qwen_3_4b.safetensors` and `flux2-vae.safetensors` in [`Comfy-Org/flux2-klein`](https://huggingface.co/Comfy-Org/flux2-klein), whose card says `apache-2.0` (one licence for the repository; it does not list one per file). |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Comfy-Org and contributors | GPL-3.0 | Not bundled or linked. `prompt-to-model` only talks to a ComfyUI you run yourself over its HTTP API. The repository page says nothing about the licence of generated outputs. |
| GGUF conversions of the above | derivatives of the source models | *other / see source model* | The [`ilintar/trellis2-gguf`](https://huggingface.co/ilintar/trellis2-gguf) card has licence `other` and the text "see the source model"; it names no specific licence. Each component stays under its original licence. |

## DINOv3

The image encoder is DINOv3 ViT-L/16 from Meta, distributed under the
[DINOv3 License](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). As checked in that text: if you distribute the
DINO Materials or derivative works (this includes converted weights), you may only do so under that agreement and must include a
copy of it; research publications that use them must acknowledge the use; use must comply with trade controls (including
military, nuclear and weapons end-use restrictions) and laws. The text we read has no "Built with DINOv3" requirement; we credit
DINOv3 voluntarily. The Hugging Face card for [`facebook/dinov3-vitl16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
is gated and points to the same licence. This repository does not reproduce the text; read it before redistributing converted weights.

## Converted model files

The GGUF files (for example [`ilintar/trellis2-gguf`](https://huggingface.co/ilintar/trellis2-gguf), or your own output of
`tools/convert.py`) are derivatives of the models above. They carry the licence label *other / see source model*, so treat
every component as bound by its original licence, including the DINOv3 agreement (include a copy of it with any DINOv3-derived weights).

## Upstream code

Files inherited from trellis.cpp keep their original headers. ArcticMint additions (`src/post_parallel.*`, `src/post_mesh.cpp`,
`tools/arctic-post/`, `tools/arctic-batch/`, `tools/prompt-to-model/`) are released under the same MIT licence as the rest of the repository.
