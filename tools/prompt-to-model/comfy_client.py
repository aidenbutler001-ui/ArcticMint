"""Minimal ComfyUI API client for the text-to-image step. Python standard library only.

Flow: POST /prompt -> poll /history/<id> -> GET /view -> POST /free (unload the image model so the card is free for the 3D engine).
Talks to a ComfyUI you already run (default http://127.0.0.1:8188); nothing here goes to the internet.
"""
import copy
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKFLOWS = os.path.join(HERE, "workflows")

# model key -> workflow file and the (node id, input) to set. Node ids were checked against the bundled JSON files.
IMAGE_MODELS = {
    "klein": {"label": "FLUX.2-klein-4B", "file": "flux2_klein4b_api.json", "prompt": ("74", "text"), "seed": ("73", "noise_seed"),
              "size": [("62", "width", "height"), ("66", "width", "height")]},
    "zimage": {"label": "Z-Image-Turbo", "file": "zimage_turbo_api.json", "prompt": ("27", "text"), "seed": ("3", "seed"),
               "size": [("13", "width", "height")]},
}
OUTPUT_PREFIX = "prompt_to_model"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "[::1]")


class ComfyError(Exception):
    """Something the user can act on (ComfyUI not running, a model file missing, the prompt was rejected...)."""


def load_workflow(model_key, workflow_dir=None):
    spec = IMAGE_MODELS[model_key]
    path = os.path.join(workflow_dir or DEFAULT_WORKFLOWS, spec["file"])
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise ComfyError("cannot read the %s workflow %s: %s" % (spec["label"], path, e))


def build_workflow(model_key, prompt, seed, tag, width=1024, height=1024, workflow_dir=None):
    """The workflow with this job's prompt, seed, size and output name filled in. Raises ComfyError if a node is missing."""
    spec = IMAGE_MODELS[model_key]
    wf = copy.deepcopy(load_workflow(model_key, workflow_dir))

    def node(nid):
        if nid not in wf:
            raise ComfyError("workflow %s has no node %s (re-exported? edit the node ids in comfy_client.IMAGE_MODELS)" % (spec["file"], nid))
        return wf[nid]["inputs"]

    pn, pf = spec["prompt"]
    node(pn)[pf] = prompt
    sn, sf = spec["seed"]
    node(sn)[sf] = int(seed)
    for nid, wname, hname in spec["size"]:
        if nid in wf:
            wf[nid]["inputs"][wname], wf[nid]["inputs"][hname] = int(width), int(height)
    saved = [n for n, v in wf.items() if v.get("class_type") == "SaveImage"]
    if not saved:
        raise ComfyError("workflow %s has no SaveImage node" % spec["file"])
    for n in saved:
        wf[n]["inputs"]["filename_prefix"] = "%s/%s" % (OUTPUT_PREFIX, tag)
    return wf


class Comfy:
    def __init__(self, url, timeout=20):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex
        host = (urllib.parse.urlsplit(self.url).hostname or "").lower()
        # a system proxy (HTTP_PROXY) must not catch calls to the ComfyUI on this machine
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if host in LOCAL_HOSTS else urllib.request.build_opener()

    def _open(self, path, data=None, timeout=None):
        req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with self.opener.open(req, timeout=timeout or self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(self._explain(e.code, e.read()))
        except (urllib.error.URLError, OSError, ValueError, http.client.HTTPException) as e:
            raise ComfyError("ComfyUI is not reachable at %s (%s)" % (self.url, getattr(e, "reason", e)))

    @staticmethod
    def _explain(code, body):
        try:
            j = json.loads(body)
        except ValueError:
            return "ComfyUI answered HTTP %d" % code
        err = j.get("error") if isinstance(j, dict) else None
        msg = (err.get("message") if isinstance(err, dict) else err) or "HTTP %d" % code
        details = []
        for nid, ne in (j.get("node_errors") or {}).items() if isinstance(j, dict) else []:
            for e in ne.get("errors", []):
                details.append("node %s (%s): %s %s" % (nid, ne.get("class_type"), e.get("message", ""), e.get("details", "")))
        return "ComfyUI rejected the request: %s%s" % (msg, "; " + "; ".join(details[:3]) if details else "")

    def get(self, path, timeout=None):
        raw = self._open(path, timeout=timeout)
        try:
            return json.loads(raw or b"null")
        except ValueError:
            raise ComfyError("ComfyUI answered %s with something that is not JSON" % path)

    def post(self, path, obj, timeout=None):
        raw = self._open(path, json.dumps(obj).encode(), timeout=timeout)
        try:
            return json.loads(raw) if raw.strip() else {}
        except ValueError:
            raise ComfyError("ComfyUI answered %s with something that is not JSON" % path)

    def alive(self):
        try:
            self._open("/system_stats", timeout=3)
            return True
        except ComfyError:
            return False

    def combo_values(self, class_type, field):
        """The model files ComfyUI offers for a loader's combo input (None if it cannot say)."""
        try:
            info = self.get("/object_info/" + urllib.parse.quote(class_type), timeout=10)
            spec = (info.get(class_type) or {}).get("input", {})
            spec = (spec.get("required") or {}).get(field) or (spec.get("optional") or {}).get(field)
            return list(spec[0]) if spec and isinstance(spec[0], list) else None
        except (ComfyError, AttributeError, TypeError, IndexError):
            return None

    def missing_models(self, wf):
        """Loader inputs whose file ComfyUI does not list: [(node id, class, field, value)]. Empty = all there (or unknowable)."""
        out = []
        for nid, n in wf.items():
            ct = n.get("class_type", "")
            if not ct.endswith("Loader"):
                continue
            for field, val in n.get("inputs", {}).items():
                if isinstance(val, str) and field.endswith("_name"):
                    have = self.combo_values(ct, field)
                    if have is not None and val not in have:
                        out.append((nid, ct, field, val))
        return out

    def submit(self, wf):
        j = self.post("/prompt", {"prompt": wf, "client_id": self.client_id})
        if not isinstance(j, dict) or "prompt_id" not in j:
            raise ComfyError("ComfyUI did not return a prompt id")
        return j["prompt_id"]

    def wait(self, prompt_id, timeout=900, poll=0.5):
        """Blocks until the prompt finishes; returns its list of output image dicts."""
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout:
                self.cancel(prompt_id)
                raise ComfyError("image generation took longer than %d s; interrupted" % timeout)
            ent = (self.get("/history/" + prompt_id) or {}).get(prompt_id)
            if ent:
                st = ent.get("status") or {}
                if st.get("status_str") == "error":
                    msgs = [m for m in st.get("messages", []) if m and m[0] == "execution_error"]
                    why = (msgs[0][1].get("exception_message") if msgs else "") or "execution error"
                    raise ComfyError("ComfyUI failed while generating: %s" % why.strip()[:300])
                imgs = [i for o in (ent.get("outputs") or {}).values() for i in o.get("images", [])]
                if imgs or st.get("completed"):
                    if not imgs:
                        raise ComfyError("ComfyUI finished but produced no image")
                    return imgs
            time.sleep(poll)

    def fetch(self, img):
        q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")})
        return self._open("/view?" + q, timeout=60)

    def free(self):
        """Ask ComfyUI to unload its models and free VRAM. ComfyUI does this on its next loop, not before the call returns."""
        try:
            self._open("/free", json.dumps({"unload_models": True, "free_memory": True}).encode(), timeout=15)
        except ComfyError:
            pass

    def interrupt(self, prompt_id=None):
        try:
            self._open("/interrupt", json.dumps({"prompt_id": prompt_id} if prompt_id else {}).encode(), timeout=5)
        except ComfyError:
            pass

    def cancel(self, prompt_id):
        """Stop this job whether it is running (interrupt) or still waiting in the queue (delete)."""
        self.interrupt(prompt_id)
        try:
            self._open("/queue", json.dumps({"delete": [prompt_id]}).encode(), timeout=5)
        except ComfyError:
            pass
