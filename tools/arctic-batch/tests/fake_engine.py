"""Fake trellis-cli / post-replay / finish. Usage: fake_engine.py cli|replay|finish <args...>

cli     : writes the dump named by --dump-post, or by $TRELLIS_DUMP_POST (raw mode); prints its argv and the env it
          saw as one ARGS line, and a fill_holes hash line when TRELLIS_POST_HASH is set.
replay  : writes a GLB and <glb>.env.json (env seen); prints hash lines when TRELLIS_POST_HASH is set;
          FAKE_CPU_QEM=1 prints the CPU-fallback line, FAKE_POST_FAIL=1 exits 5.
finish  : <glb> <name> <out>; writes <out>/env.json; FAKE_FINISH_OOM_ONCE=<marker> fails the first call with an OOM line.
"""
import hashlib, os, sys, time, json
mode, args = sys.argv[1], sys.argv[2:]
GEN, POST = float(os.environ.get("FAKE_GEN_S", "0.4")), float(os.environ.get("FAKE_POST_S", "0.4"))
ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "TRELLIS_POST_THREADS", "CUDA_DEVICE_ORDER", "TRELLIS_POST_HASH",
            "TRELLIS_DUMP_POST", "TRELLIS_STOP_AFTER_POST_DUMP", "TRELLIS_REMESH_MEM_GIB", "ARCTIC_GPU_PHASE_LOCK")


def hash_line(stage, body):
    h = hashlib.sha256(body).hexdigest()
    return "  [post-hash] %-20s V=%-10d F=%-10d verts=%s faces=%s area=1.5" % (stage, 10, 20, h[:16], h[16:32])


if mode == "cli":
    img = args[args.index("-i") + 1]
    dump = args[args.index("--dump-post") + 1] if "--dump-post" in args else os.environ["TRELLIS_DUMP_POST"]
    print("ARGS " + json.dumps({"argv": args, "env": {k: os.environ.get(k) for k in ENV_KEYS}}))
    if "bad" in os.path.basename(img): sys.exit(3)
    time.sleep(GEN)
    body = b"DUMP" + hashlib.sha256(open(img, "rb").read()).digest()
    open(dump, "wb").write(body)
    if os.environ.get("TRELLIS_POST_HASH"):
        print(hash_line("fill_holes", body))
elif mode == "replay":
    dump, glb = args[0], args[1]
    time.sleep(POST)
    body = open(dump, "rb").read()
    env = {k: os.environ.get(k) for k in ENV_KEYS}
    open(glb + ".env.json", "w").write(json.dumps(env))
    print("ARGS " + json.dumps(args))
    if os.environ.get("TRELLIS_POST_HASH"):
        print(hash_line("input", body if not os.environ.get("FAKE_BAD_DUMP_HASH") else body + b"x"))
        print(hash_line("drop_components", body + b"drop"))
    if os.environ.get("FAKE_CPU_QEM"):
        print("[decimate] GPU decimation unavailable; falling back to the CPU path (output is unaffected)")
    if os.environ.get("FAKE_POST_FAIL"): sys.exit(5)
    open(glb, "wb").write(b'glTF{"generated":"t%d","x":1}' % time.time_ns() + hashlib.sha256(body).digest())
else:
    glb, name, out = args[:3]
    marker = os.environ.get("FAKE_FINISH_OOM_ONCE")
    if marker and not os.path.exists(marker):
        open(marker, "w").write("1")
        print("RuntimeError: CUDA error: out of memory")
        sys.exit(1)
    time.sleep(float(os.environ.get("FAKE_FINISH_S", "0.2")))
    json.dump({k: os.environ.get(k) for k in ENV_KEYS}, open(os.path.join(out, "env.json"), "w"))
