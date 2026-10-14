# Notice: credits and third-party licences

ArcticMint is a derivative work and builds on the projects below. This file lists them for convenience; it is not legal advice.
Where a licence has not been checked against its actual text in this repository, the entry says **verify**.
[`LICENSE`](LICENSE) and upstream copyright headers are unchanged.

| Component | Author | Licence | Notes |
|-----------|--------|---------|-------|
| ArcticMint / [trellis.cpp](https://github.com/pwilkin/trellis.cpp) | Piotr Wilkin (Copyright (c) 2026) | MIT | Full text in [`LICENSE`](LICENSE). The engine, model ports and most of the CLI come from this work. |
| [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) and [`TRELLIS.2-4B`](https://huggingface.co/microsoft/TRELLIS.2-4B) weights | Microsoft | MIT (as stated in the README) | Verify against the upstream repository and model card before redistributing weights. |
| [ggml](https://github.com/ggml-org/ggml) | ggml authors | MIT 
| [DINOv3](https://github.com/facebookresearch/dinov3) ViT-L/16 | Meta | DINOv3 licence agreement (not MIT) | See below. |
| [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | ZhengPeng7 and co-authors | Verify | Used for background removal. Licence not checked here. |
| [xatlas](https://github.com/jpcy/xatlas) | Jonathan Young | MIT | Vendored in `thirdparty/xatlas`. |
| [stb](https://github.com/nothings/stb) (`stb_image`, `stb_image_write`, `stb_image_resize`) | Sean Barrett and contributors | Public domain | Vendored in `thirdparty/stb`. |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | Yuji Hirose | MIT | Vendored in `thirdparty/cpp-httplib`; used by `trellis-server`. |
| fqms (Fast-Quadric-Mesh-Simplification) | Spacerat and contributors | MIT | Vendored in `thirdparty/fqms`. |
| [meshoptimizer](https://github.com/zeux/meshoptimizer) | Arseny Kapoulkine | MIT | Vendored in `thirdparty/meshoptimizer`. |
| [`<model-viewer>`](https://github.com/google/model-viewer) (bundles three.js) | Google and contributors | BSD-3-Clause (header of the bundle), three.js is MIT | Vendored as a minified bundle in `app/public/vendor/` and `tools/mv_preview/`. |
| [Pixal3D](https://github.com/TencentARC/Pixal3D) | TencentARC | Verify | Optional; run by `--model pixal3d`. |
| GGUF conversions of the above | derivatives of the source models | *other / see source model* | Each component stays under its original licence. |

## Built with DINOv3

The image encoder is DINOv3 ViT-L/16 from Meta. DINOv3 is distributed under its own licence agreement, which requires the
attribution **"Built with DINOv3"** and that you pass its agreement on to anyone you give these weights to. Read and follow
the DINOv3 licence before using or redistributing the converted weights; this repository does not reproduce its text.

## Converted model files

The GGUF files (for example [`ilintar/trellis2-gguf`](https://huggingface.co/ilintar/trellis2-gguf), or your own output of
`tools/convert.py`) are derivatives of the models above. They carry the licence label *other / see source model*, so treat
every component as bound by its original licence, including the DINOv3 agreement.

## Upstream code

Files inherited from trellis.cpp keep their original headers. ArcticMint additions (`src/post_parallel.*`, `src/post_mesh.cpp`,
`tools/arctic-post/`, `tools/arctic-batch/`) are released under the same MIT licence as the rest of the repository.
