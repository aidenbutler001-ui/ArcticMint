# Showcase

Everything here was made on one Windows PC (RTX 4070 SUPER for the 3D engine, RTX 3060 Ti for the pictures and the renders) from a one-sentence
prompt. Nothing was modelled or painted by hand, and nothing was retouched.

How each one was made:

1. **Picture.** FLUX.2-klein-4B (Apache-2.0, fp8) through ComfyUI turned the prompt into a single-object reference picture. Four seeds per subject were
   tried and one was kept by eye. The prompt always ended with the same framing words:
   *single object, front toward the viewer at three-quarter view, centered, full silhouette visible, plain light grey background, soft even studio lighting, no shadow*.
2. **Model.** `trellis-cli` (this repository) turned the kept picture into a UV-textured PBR GLB: `--res 1024 --steps 24 --tex-res 1024 --atlas 4096 --faces 150000 --tex-reroll 2 --seed 1`.
   `--tex-reroll 2` redoes only the colour step if its brightness is far from the picture's; the mushrooms needed one redo and the other three kept their first colour pass. Each took 11 to 14 minutes (see the table).
3. **Render.** Blender 4.5 (Cycles, 256 samples, AgX) on a plain grey backdrop with three area lights. The model is only centred, scaled and set on the floor; nothing is edited.
4. **Repository copy.** The GLBs in [`assets/showcase/`](../../assets/showcase/) are the engine output run through a collapse decimation to 80,000 faces with the texture atlas shrunk from 4096 to 2048 px
   (WebP), to keep the repository light. The renders come from the full-size output.

| | Prompt (before the framing words) | Picture seed | Engine time | Files |
|---|---|---|---|---|
| Diving helmet | antique brass deep-sea diving helmet with riveted seams, round glass portholes and a bolted collar, aged patina, hard-surface prop, game asset | 11 | 13 min | [render](diving-helmet.jpg), [picture to model](diving-helmet-picture-to-model.jpg), [GLB](../../assets/showcase/diving-helmet/diving-helmet.glb) |
| Stone lantern | mossy carved stone garden lantern on a square base, weathered granite with green moss and lichen, chunky simple forms, game asset | 33 | 11 min | [render](stone-lantern.jpg), [picture to model](stone-lantern-picture-to-model.jpg), [GLB](../../assets/showcase/stone-lantern/stone-lantern.glb) |
| Mushrooms | cluster of three fat forest mushrooms with warm brown caps and cream stems, tiny moss at the base, stylized organic prop, game asset | 11 | 14 min | [render](mushrooms.jpg), [picture to model](mushrooms-picture-to-model.jpg), [GLB](../../assets/showcase/mushrooms/mushrooms.glb) |
| Teapot | glazed ceramic teapot with a hand-painted blue floral pattern, curved spout and handle, small lid knob, game asset | 22 | 12 min | [render](teapot.jpg), [picture to model](teapot-picture-to-model.jpg), [GLB](../../assets/showcase/teapot/teapot.glb) |

## What to expect

- Props and organic forms work well. Hard-surface detail such as rivets and portholes survives at 1024.
- Fine printed patterns come out softer than in the picture (look at the teapot's flowers), and the back of an object is invented by the model, since it saw one view.
- Times are for this PC with the verbose log on and the 4070 SUPER also driving a monitor. Most of the time is the 1024 cascade and the post-processing of a mesh of about 24 million
  faces down to 150,000.
