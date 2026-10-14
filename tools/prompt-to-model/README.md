# prompt-to-model

A sentence in, a textured GLB out. Part of ArcticMint; Python standard library only.

> **Status.** Unit-tested against a fake ComfyUI and a fake engine (no GPU needed). The two bundled ComfyUI workflows are the
> ones the authors ran on their own PC, but they have not been re-run from this folder on real hardware yet.

The engine (`trellis-cli`) reads a picture, not text. `prompt_to_model.py` adds the missing first step:

1. **Picture.** It fills the prompt, seed and size into a ComfyUI workflow, sends it to a ComfyUI you already run, saves the
   PNG, and asks ComfyUI to unload its models so the card is free again.
2. **Model.** It runs `trellis-cli -i PICTURE -o NAME.glb -m MODELS --seed S` and writes `NAME.json` (prompt, final prompt,
   seed, picture path, timings, engine exit code) beside the picture and the GLB.

## Run it

```bash
python tools/prompt-to-model/prompt_to_model.py "a weathered wooden treasure chest with iron bands" \
  --models /path/to/gguf --cli build/trellis-cli --comfy-url http://127.0.0.1:8188 -o out/chest
```

| flag | effect |
|------|--------|
| `-o DIR` | output folder (required); files are named from `--name`, else from the prompt |
| `-m DIR` / `--cli CMD` | engine GGUF folder and the `trellis-cli` command (a path, or e.g. `python fake.py`) |
| `--comfy-url URL` | the running ComfyUI (default `http://127.0.0.1:8188`) |
| `--image-model klein\|zimage` | FLUX.2-klein-4B (default) or Z-Image-Turbo |
| `--style 3d-ready\|none` | `3d-ready` appends single-object framing words (centered, whole silhouette, plain light grey background, soft light, no shadow); `none` sends your prompt as typed |
| `--seed N` / `--size PX` | picture seed (also the 3D seed) and picture side, default 1 and 1024 |
| `--picture-only`, `--count N` | stop after the picture; `--count` makes seeds `SEED..SEED+N-1` to choose from |
| `--image FILE` | skip the picture step and run the engine on the picture you picked |
| `--gpu N` | passed to `trellis-cli`; also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `N` is the `nvidia-smi` index |
| `--engine-args "..."` | extra `trellis-cli` flags, for example `"--res 1024 --steps 24 --tex-reroll 2"` |
| `--workflows DIR` | replacement workflow files (ComfyUI "Save (API format)" JSON with the same file names) |

Choosing among several pictures is a human step:

```bash
python tools/prompt-to-model/prompt_to_model.py "a brass diving helmet" -o out/helmet --picture-only --count 6
# look at out/helmet/*.png, then:
python tools/prompt-to-model/prompt_to_model.py --image out/helmet/brass_diving_helmet_s3.png --seed 3 -o out/helmet -m /path/to/gguf --cli build/trellis-cli
```

Exit code 0 means a picture (with `--picture-only`) or a GLB was written; 1 means a step failed and the message says which.

## Writing prompts that work

- Describe **one object**, with positive wording and countable parts stated exactly ("exactly one wheel with eight spokes").
  Negations ("no handles") do little on these few-step models.
- Leave sizes in metres out of the prompt; a text-to-image model cannot honour them.
- Check the picture before spending engine time: the whole object inside the frame, one object, plain background, no cast shadow.

## What ComfyUI needs

This tool does not install or start ComfyUI and downloads nothing. You need a ComfyUI recent enough to have the FLUX.2 and Z-Image
nodes (the workflows were exported from ComfyUI 0.39) and these files in its model folders. Before submitting, the tool asks ComfyUI
which loader files it lists and stops with the names of any that are missing.

| `--image-model` | `models/diffusion_models/` | `models/text_encoders/` | `models/vae/` |
|-----------------|----------------------------|-------------------------|---------------|
| `klein` | `flux-2-klein-4b-fp8.safetensors` (about 3.8 GB, from `black-forest-labs/FLUX.2-klein-4b-fp8`) | `qwen_3_4b.safetensors` (about 7.5 GB, `Comfy-Org/flux2-klein`) | `flux2-vae.safetensors` (`Comfy-Org/flux2-klein`) |
| `zimage` | `z_image_turbo_int8_convrot.safetensors` (about 5.8 GB, `Comfy-Org/z_image_turbo`) | `qwen_3_4b_fp8_mixed.safetensors` (about 5.3 GB, same repo) | `ae.safetensors` (same repo) |

Licences are in [`NOTICE.md`](../../NOTICE.md). A different checkpoint or file name means editing the loader names in the workflow JSON.

**One card or two.** The image model and the engine run one after the other, never together, and the tool frees ComfyUI's VRAM before
the engine starts. With two GPUs you can still keep ComfyUI on a smaller card (`python main.py --cuda-device 0 --port 8188`) and give
the engine the other with `--gpu 1`, so the engine never has to share its card with the image model.

## Tests

```bash
python3 -m unittest discover -s tools/prompt-to-model/tests -v
```

`tests/fake_comfy.py` imitates ComfyUI's HTTP API (it can also run on its own: `python tests/fake_comfy.py --port 8188`) and
`tests/fake_engine.py` imitates `trellis-cli`. CI runs them in `.github/workflows/python-tools.yml`.

## Files

| File | What it is |
|------|------------|
| `prompt_to_model.py` | the command |
| `comfy_client.py` | ComfyUI HTTP client and the workflow filler; the node ids it sets are listed in `IMAGE_MODELS` |
| `workflows/*.json` | ComfyUI API-format workflows for the two image models |
