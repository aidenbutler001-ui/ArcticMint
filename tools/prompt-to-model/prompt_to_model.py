#!/usr/bin/env python3
"""prompt_to_model.py: a sentence in, a textured GLB out. Python standard library only.

Step 1 asks a ComfyUI you already run (FLUX.2-klein-4B or Z-Image-Turbo, workflows in workflows/) for a single-object picture of your
prompt. Step 2 hands that picture to trellis-cli, which writes the GLB. The picture, the GLB and a small JSON record land in --out.

  python tools/prompt-to-model/prompt_to_model.py "a weathered wooden treasure chest with iron bands" \\
      --models /path/to/gguf --cli build/trellis-cli --comfy-url http://127.0.0.1:8188 -o out/chest

  --picture-only        stop after the picture(s); --count N makes N seeds to choose from, then rerun with --image PICTURE
  --image PICTURE       skip step 1 and run step 2 on a picture you picked
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import comfy_client  # noqa: E402

# Appended to the prompt so the 3D engine gets one clean, fully visible subject on a plain background.
FRAMING = ("single object, front toward the viewer at three-quarter view, centered, full silhouette visible, "
           "plain light grey background, soft even studio lighting, no shadow")
STYLES = {"3d-ready": FRAMING, "none": ""}


def split_cmd(s):
    """shlex split; on Windows shlex keeps the quote characters, so surrounding quotes are stripped."""
    posix = os.name != "nt"
    toks = shlex.split(s, posix=posix)
    if not posix:
        toks = [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t for t in toks]
    return toks


def slug(text, limit=40):
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:limit].strip("_")
    return s or "model"


def final_prompt(prompt, style):
    suffix = STYLES[style]
    return prompt.strip().rstrip(",") + (", " + suffix if suffix else "")


def make_picture(client, model, prompt, seed, path, size, workflow_dir, timeout):
    wf = comfy_client.build_workflow(model, prompt, seed, os.path.splitext(os.path.basename(path))[0], size, size, workflow_dir)
    missing = client.missing_models(wf)
    if missing:
        raise comfy_client.ComfyError("ComfyUI does not list these model files: %s. Put them in ComfyUI's models folders (see README)."
                                      % ", ".join("%s=%s" % (f, v) for _, _, f, v in missing))
    imgs = client.wait(client.submit(wf), timeout=timeout)
    with open(path, "wb") as f:
        f.write(client.fetch(imgs[0]))


def run_engine(cmd, picture, glb, models, seed, gpu, extra, timeout):
    args = cmd + ["-i", picture, "-o", glb, "-m", models, "--seed", str(seed)] + (["--gpu", str(gpu)] if gpu is not None else []) + extra
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"       # --gpu then means the nvidia-smi index
    print("engine: " + " ".join(shlex.quote(a) for a in args), flush=True)
    try:
        return subprocess.run(args, env=env, timeout=timeout or None).returncode
    except subprocess.TimeoutExpired:
        return -1
    except OSError as e:
        print("error: cannot start the engine (%s): %s" % (cmd[0], e), file=sys.stderr)
        return -2


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", nargs="?", help="what to make, one object, positive wording (\"exactly one wheel with eight spokes\")")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--name", help="file name stem (default: from the prompt)")
    ap.add_argument("-m", "--models", help="folder with the engine's GGUF files (needed unless --picture-only)")
    ap.add_argument("--cli", default="trellis-cli", help="trellis-cli command (path, or e.g. 'python fake.py')")
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188", help="a running ComfyUI")
    ap.add_argument("--image-model", choices=sorted(comfy_client.IMAGE_MODELS), default="klein", help="klein = FLUX.2-klein-4B, zimage = Z-Image-Turbo")
    ap.add_argument("--workflows", help="folder with replacement workflow JSON files (API format; default: the bundled ones)")
    ap.add_argument("--style", choices=sorted(STYLES), default="3d-ready", help="3d-ready appends single-object framing words; none = your prompt as typed")
    ap.add_argument("--seed", type=int, default=1, help="picture seed; also the 3D seed")
    ap.add_argument("--count", type=int, default=1, help="pictures to make, seeds SEED..SEED+N-1 (more than 1 needs --picture-only)")
    ap.add_argument("--size", type=int, default=1024, help="picture side in pixels")
    ap.add_argument("--picture-only", action="store_true", help="stop after the picture(s)")
    ap.add_argument("--image", help="skip the picture step and use this file")
    ap.add_argument("--gpu", type=int, help="3D engine GPU (nvidia-smi index); sets CUDA_DEVICE_ORDER=PCI_BUS_ID")
    ap.add_argument("--engine-args", default="", help="extra trellis-cli arguments, e.g. '--res 1024 --steps 24'")
    ap.add_argument("--picture-timeout", type=float, default=900, help="seconds to wait for ComfyUI")
    ap.add_argument("--engine-timeout", type=float, default=3600, help="seconds before the engine is killed (0 = never)")
    a = ap.parse_args(argv)

    if a.image and a.picture_only:
        ap.error("--image and --picture-only do nothing together")
    if a.image and a.count != 1:
        ap.error("--count goes with the picture step; --image skips it")
    if not a.image and not a.prompt:
        ap.error("give a prompt, or --image")
    if a.count < 1:
        ap.error("--count is at least 1")
    if a.count > 1 and not a.picture_only:
        ap.error("--count %d makes %d pictures; add --picture-only, look at them, then rerun with --image and --seed" % (a.count, a.count))
    if not a.picture_only and not a.models:
        ap.error("--models is needed for the 3D step")
    if a.size < 256:
        ap.error("--size is at least 256")

    os.makedirs(a.out, exist_ok=True)
    name = a.name or slug(a.prompt or os.path.splitext(os.path.basename(a.image))[0])
    record = {"prompt": a.prompt, "style": a.style, "seed": a.seed, "image_model": a.image_model if not a.image else None, "pictures": []}
    t0 = time.time()

    if a.image:
        if not os.path.isfile(a.image):
            print("error: no such picture: %s" % a.image, file=sys.stderr)
            return 1
        pictures = [a.image]
    else:
        text = final_prompt(a.prompt, a.style)
        record["final_prompt"] = text
        client = comfy_client.Comfy(a.comfy_url)
        if not client.alive():
            print("error: ComfyUI is not reachable at %s. Start it (python main.py --port ...) or pass --comfy-url." % a.comfy_url, file=sys.stderr)
            return 1
        pictures = []
        try:
            for i in range(a.count):
                path = os.path.join(a.out, "%s_s%d.png" % (name, a.seed + i))
                print("picture %d/%d: %s seed %d" % (i + 1, a.count, comfy_client.IMAGE_MODELS[a.image_model]["label"], a.seed + i), flush=True)
                make_picture(client, a.image_model, text, a.seed + i, path, a.size, a.workflows, a.picture_timeout)
                pictures.append(path)
        except comfy_client.ComfyError as e:
            print("error: %s" % e, file=sys.stderr)
            return 1
        finally:
            client.free()               # the engine gets the card back
    record["pictures"] = pictures
    record["picture_s"] = round(time.time() - t0, 1)

    rc = 0
    if not a.picture_only:
        glb = os.path.join(a.out, name + ".glb")
        t1 = time.time()
        rc = run_engine(split_cmd(a.cli), pictures[0], glb, a.models, a.seed, a.gpu, split_cmd(a.engine_args) if a.engine_args else [],
                        a.engine_timeout)
        record.update(glb=glb if rc == 0 else None, engine_exit=rc, engine_s=round(time.time() - t1, 1))
        if rc == 0 and not os.path.isfile(glb):
            print("error: the engine exited 0 but wrote no %s" % glb, file=sys.stderr)
            rc = record["engine_exit"] = -3
        if rc != 0:
            print("error: the 3D step failed (exit %s)" % rc, file=sys.stderr)
    with open(os.path.join(a.out, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
        f.write("\n")
    if rc == 0:
        print("done: " + (record.get("glb") or ", ".join(pictures)))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
