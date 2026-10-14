#!/usr/bin/env python3
"""Stand-in for ComfyUI's HTTP API (/prompt, /history, /view, /object_info, /system_stats, /free, /interrupt), so prompt_to_model can be
tested without a GPU. Draws a simple shaded object on grey, coloured from the prompt, as the "generated" image.

  python tests/fake_comfy.py --port 8188
Env: FAKE_SPEED (default 1; 0.01 for tests), FAKE_COMFY_MISSING=1 hides the klein model file,
     FAKE_COMFY_FAIL=1 makes /prompt reject, FAKE_COMFY_EXEC_FAIL=1 makes the run fail while sampling.
"""
import argparse, hashlib, json, math, os, struct, sys, threading, time, urllib.parse, uuid, zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {
    "UNETLoader": {"unet_name": ["flux-2-klein-4b-fp8.safetensors", "z_image_turbo_int8_convrot.safetensors"]},
    "CLIPLoader": {"clip_name": ["qwen_3_4b.safetensors", "qwen_3_4b_fp8_mixed.safetensors"]},
    "VAELoader": {"vae_name": ["flux2-vae.safetensors", "ae.safetensors"]},
}


def png(w, h, rows):
    raw = b"".join(b"\x00" + r for r in rows)

    def chunk(t, d):
        c = struct.pack(">I", len(d)) + t + d
        return c + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def draw(prompt, seed, size=512):
    """A rounded, shaded blob (colour and proportions from the prompt + seed) on a light grey background."""
    d = hashlib.md5(("%s|%s" % (prompt, seed)).encode()).digest()
    base = [60 + d[i] % 150 for i in range(3)]
    aspect = 0.7 + (d[3] % 60) / 100.0
    rows = []
    c = size / 2.0
    for y in range(size):
        row = bytearray()
        for x in range(size):
            dx, dy = (x - c) / (size * 0.30), (y - c * 1.05) / (size * 0.30 * aspect)
            r = (abs(dx) ** 3 + abs(dy) ** 3) ** (1 / 3.0)
            if r < 1:
                nz = math.sqrt(max(0.0, 1 - r * r))
                light = max(0.0, 0.5 * (-dx * 0.6 - dy * 0.6) + 0.7 * nz)
                row += bytes(min(255, int(ch * (0.35 + 0.9 * light))) for ch in base)
            else:
                g = 228 - int(14 * y / size)
                row += bytes((g, g, g))
        rows.append(bytes(row))
    return png(size, size, rows)


class FakeComfy:
    def __init__(self, port=0, speed=None):
        self.speed = float(os.environ.get("FAKE_SPEED", "1")) if speed is None else speed
        self.jobs = {}
        self.lock = threading.Lock()
        self.freed = 0
        self.interrupted = 0
        self.deleted = []
        me = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                if not isinstance(body, (bytes, bytearray)):
                    body = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path, _, qs = self.path.partition("?")
                q = dict(urllib.parse.parse_qsl(qs))
                if path == "/system_stats":
                    return self._send(200, {"system": {"comfyui_version": "0.34.0-fake"},
                                            "devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3060 Ti (fake)", "type": "cuda", "index": 0,
                                                         "vram_total": 8589934592, "vram_free": 8000000000}]})
                if path.startswith("/object_info/"):
                    cls = urllib.parse.unquote(path.split("/", 2)[2])
                    fields = me.models().get(cls)
                    if fields is None:
                        return self._send(404, {})
                    return self._send(200, {cls: {"input": {"required": {k: [v] for k, v in fields.items()}}}})
                if path.startswith("/history/"):
                    return self._send(200, me.history(path.split("/", 2)[2]))
                if path == "/view":
                    j = me.find_by_file(q.get("filename", ""))
                    if not j:
                        return self._send(404, {})
                    return self._send(200, draw(j["prompt"], j["seed"]), "image/png")
                self._send(404, {})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/prompt":
                    code, out = me.submit(body)
                    return self._send(code, out)
                if self.path == "/free":
                    me.freed += 1
                    return self._send(200, {})
                if self.path == "/queue":
                    me.deleted += body.get("delete", [])
                    return self._send(200, {})
                if self.path == "/interrupt":
                    me.interrupted += 1
                    with me.lock:
                        for j in me.jobs.values():
                            j["interrupted"] = True
                    return self._send(200, {})
                self._send(404, {})

        self.srv = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.srv.daemon_threads = True
        self.port = self.srv.server_address[1]
        self.url = "http://127.0.0.1:%d" % self.port

    def models(self):
        m = {k: dict(v) for k, v in MODELS.items()}
        if os.environ.get("FAKE_COMFY_MISSING"):
            m["UNETLoader"]["unet_name"] = [x for x in m["UNETLoader"]["unet_name"] if "klein" not in x]
        return m

    def submit(self, body):
        wf = body.get("prompt") or {}
        if os.environ.get("FAKE_COMFY_FAIL"):
            return 400, {"error": {"type": "invalid_prompt", "message": "fake rejection"}, "node_errors": {}}
        node_errors = {}
        for nid, n in wf.items():
            have = self.models().get(n.get("class_type"))
            for field, val in (n.get("inputs") or {}).items():
                if have and field in have and val not in have[field]:
                    node_errors[nid] = {"class_type": n["class_type"], "errors": [{"message": "Value not in list",
                                        "details": "%s: '%s' not in %s" % (field, val, have[field])}]}
        if node_errors:
            return 400, {"error": {"type": "prompt_outputs_failed_validation", "message": "Prompt outputs failed validation"},
                         "node_errors": node_errors}
        text = next((n["inputs"].get("text", "") for n in wf.values() if n.get("class_type") == "CLIPTextEncode"), "")
        seed = next((n["inputs"].get("noise_seed", n["inputs"].get("seed")) for n in wf.values()
                     if "noise_seed" in n.get("inputs", {}) or n.get("class_type") == "KSampler"), 0)
        prefix = next((n["inputs"].get("filename_prefix", "x") for n in wf.values() if n.get("class_type") == "SaveImage"), "x")
        pid = uuid.uuid4().hex
        fname = "%s_%05d_.png" % (os.path.basename(prefix), len(self.jobs) + 1)
        with self.lock:
            self.jobs[pid] = {"prompt": text, "seed": seed, "ready": time.time() + 5.0 * self.speed, "interrupted": False,
                              "sub": os.path.dirname(prefix), "file": fname}
        return 200, {"prompt_id": pid, "number": len(self.jobs), "node_errors": {}}

    def history(self, pid):
        with self.lock:
            j = self.jobs.get(pid)
        if not j:
            return {}
        if j["interrupted"]:
            return {pid: {"status": {"status_str": "error", "completed": False,
                                     "messages": [["execution_error", {"exception_message": "interrupted"}]]}, "outputs": {}}}
        if time.time() < j["ready"]:
            return {}
        if os.environ.get("FAKE_COMFY_EXEC_FAIL"):
            return {pid: {"status": {"status_str": "error", "completed": False,
                                     "messages": [["execution_error", {"exception_message": "CUDA out of memory (fake)"}]]}, "outputs": {}}}
        return {pid: {"status": {"status_str": "success", "completed": True, "messages": []},
                      "outputs": {"9": {"images": [{"filename": j["file"], "subfolder": j["sub"], "type": "output"}]}}}}

    def find_by_file(self, fname):
        with self.lock:
            return next((j for j in self.jobs.values() if j["file"] == fname), None)

    def start(self):
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        return self

    def stop(self):
        self.srv.shutdown()
        self.srv.server_close()


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8188)
    a = ap.parse_args(argv)
    f = FakeComfy(a.port)
    print("fake ComfyUI on %s" % f.url, flush=True)
    try:
        f.srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(sys.argv[1:])
