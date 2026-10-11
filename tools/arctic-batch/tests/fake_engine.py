"""Fake trellis-cli / post-replay. Usage: fake_engine.py cli|replay <args...>"""
import hashlib, os, sys, time, json
mode, args = sys.argv[1], sys.argv[2:]
GEN, POST = float(os.environ.get("FAKE_GEN_S", "0.4")), float(os.environ.get("FAKE_POST_S", "0.4"))
if mode == "cli":
    img = args[args.index("-i") + 1]; dump = args[args.index("--dump-post") + 1]
    if "bad" in os.path.basename(img): sys.exit(3)
    time.sleep(GEN)
    open(dump, "wb").write(b"DUMP" + hashlib.sha256(open(img, "rb").read()).digest())
else:
    dump, glb = args[0], args[1]
    time.sleep(POST)
    body = open(dump, "rb").read()
    env = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "TRELLIS_POST_THREADS", "CUDA_DEVICE_ORDER")}
    open(glb + ".env.json", "w").write(json.dumps(env))
    open(glb, "wb").write(b'glTF{"generated":"t%d","x":1}' % time.time_ns() + hashlib.sha256(body).digest())
