# prompt-to-model

A sentence in, a textured GLB out. Part of ArcticMint; Python standard library only.

> **Status.** Unit-tested against a fake ComfyUI and a fake engine (no GPU needed). The authors have run the FLUX.2-klein-4B path
> end to end once on their PC (ComfyUI on an RTX 3060 Ti, 2026-10-14): about 10 s for the 1024 picture,
> 328 s for the whole run (one run, so treat the numbers as an example, not a benchmark). The Z-Image-Turbo path and other machines have not been run.

The engine (`trellis-cli`) reads a picture, not text. `prompt_to_model.py` adds the missing first step:

1. **Picture.** It fills the prompt, seed and size into a ComfyUI workflow, sends it to a ComfyUI you already run, saves the
   PNG, and asks ComfyUI to unload its models so the card can be freed for the engine.
2. **Model.** It runs `trellis-cli -i PICTURE -o NAME_sS.glb -m MODELS --seed S` and writes `NAME_sS.json` (prompt, final prompt,
   seed, picture path, timings, engine exit code) beside the picture and the GLB.

Every output carries the seed (`NAME_s1.png`, `NAME_s1.glb`, `NAME_s1.json`), so a second seed never replaces an earlier run. If any
file the run would write already exists, it stops before doing any work; `--force` overwrites.

## Run it

```bash
python tools/prompt-to-model/prompt_to_model.py "a weathered wooden treasure chest with iron bands" \
  --models /path/to/gguf --cli build/trellis-cli --comfy-url http://127.0.0.1:8188 -o out/chest
```

PowerShell has no `\` line continuation; use a backtick or one line, and your build's path (`build\Release\trellis-cli.exe` with the
Visual Studio generator, `build\trellis-cli.exe` with Ninja). `--cli` may be a path with spaces, unquoted:

```powershell
python tools\prompt-to-model\prompt_to_model.py "a weathered wooden treasure chest with iron bands" `
  --models D:\models\gguf --cli build\Release\trellis-cli.exe -o out\chest
```

| flag | effect |
|------|--------|
| `--prompt-file FILE` | read the prompt from a UTF-8 text file instead of the command line (see [Quoting](#quoting-prompts-on-windows-powershell)) |
| `-o DIR` | output folder (required); files are `NAME_s<seed>.*`, NAME from `--name`, else from the prompt (letters, digits and underscores only) |
| `--force` | overwrite the files of an earlier run with the same name and seed (default: refuse) |
| `-m DIR` / `--cli CMD` | engine GGUF folder and the `trellis-cli` command (a path, or a command line such as `python fake.py`) |
| `--comfy-url URL` | the running ComfyUI (default `http://127.0.0.1:8188`) |
| `--image-model klein\|zimage` | FLUX.2-klein-4B (default) or Z-Image-Turbo |
| `--style 3d-ready\|none` | `3d-ready` appends single-object framing words (centered, whole silhouette, plain light grey background, soft light, no shadow); `none` sends your prompt as typed |
| `--seed N` / `--size PX` | picture seed (also the 3D seed) and picture side (a multiple of 16). Defaults: seed 1, or with `--image` the `_s<seed>` in the picture's file name; size 1024 |
| `--picture-only`, `--count N` | stop after the picture; `--count` makes seeds `SEED..SEED+N-1` to choose from |
| `--image FILE` | skip the picture step and run the engine on the picture you picked; if a `--picture-only` run made it, its prompt and settings carry over into the new `.json` |
| `--gpu N` | passed to `trellis-cli` with `--require-gpu` (a wrong index fails at once instead of running on the CPU); also sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so `N` is the `nvidia-smi` index |
| `--picture-timeout S` / `--engine-timeout S` | give up on ComfyUI (the job is interrupted and removed from its queue) or kill the engine after this many seconds, default 900 and 3600 |
| `--engine-args "..."` | extra `trellis-cli` flags, for example `"--res 1024 --steps 24 --tex-reroll 2"` |
| `--workflows DIR` | replacement workflow files (ComfyUI "Save (API format)" JSON with the same file names) |

Choosing among several pictures is a human step:

```bash
python tools/prompt-to-model/prompt_to_model.py "a brass diving helmet" -o out/helmet --picture-only --count 6
# look at out/helmet/*.png, then:
python tools/prompt-to-model/prompt_to_model.py --image out/helmet/brass_diving_helmet_s3.png --seed 3 -o out/helmet -m /path/to/gguf --cli build/trellis-cli
# writes out/helmet/brass_diving_helmet_s3.glb; the .json keeps the prompt (no --force needed for the same run)
```

A `--picture-only` run writes one `.json` per picture, so any of them can be picked later.

Exit code 0 means a picture (with `--picture-only`) or a GLB was written; 1 means a step failed and the message says which.

### Quoting prompts on Windows PowerShell

On the authors' PC, Windows PowerShell 5.1 passed apostrophes (single quotes) inside a prompt through intact but dropped double quotes
inside it before Python saw them (`"brass" helmet` arrived as `brass helmet`). No wrapping or escaping style was found to bring them back,
so for a prompt that contains double quotes use a file:

```powershell
python tools\prompt-to-model\prompt_to_model.py --prompt-file prompt.txt -o out\helmet --models D:\models\gguf
```

`--prompt-file` reads UTF-8 (with or without a byte-order mark) or UTF-16 with a byte-order mark, which is what `>` and `Out-File` write in
Windows PowerShell 5.1, and joins the lines into one. `Set-Content` writes the ANSI code page in 5.1; the tool refuses such a file with
a message to save it as UTF-8. The unit tests cover these encodings; the PowerShell behaviour itself has not been re-run here.

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

The three `klein` file names were checked on a real PC: they are the names of the files in ComfyUI's folders on the run above. The `zimage` names come from the workflow and have not been run here. Licences are in [`NOTICE.md`](../../NOTICE.md). A different checkpoint or file name means editing the loader names in the workflow JSON.

**One card or two.** The image model and the engine run one after the other, never together. After the picture the tool asks ComfyUI
to unload its models (`/free`). On the authors' PC this worked: the 8 GB RTX 3060 Ti fell back to about 211 MiB used once the tool
exited, and klein at 1024 took about 10 s per picture; both figures come from that one run. With two GPUs you can keep ComfyUI on one card and give the engine the
other with `--gpu`. Pick ComfyUI's card by the same index the engine uses: `--cuda-device` alone sets `CUDA_VISIBLE_DEVICES`, and CUDA's
default order is fastest first, which is not the `nvidia-smi` order. In PowerShell:

```powershell
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'; python main.py --cuda-device 0 --port 8188   # ComfyUI on nvidia-smi index 0
```

and then `--gpu 1` for the engine. In that run, with `PCI_BUS_ID` set, ComfyUI landed on the RTX 3060 Ti (index 0) as intended. The tool does not start
ComfyUI; start it yourself and pass `--comfy-url` if it is not on port 8188.

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
