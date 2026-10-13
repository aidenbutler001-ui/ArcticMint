#!/usr/bin/env python3
"""Live job queue for arctic-batch (OPT-IN: --queue-dir; stdlib only). Jobs are files in a spool directory, so anything (a
person, a script, another Claude session) can add a job while a batch runs, and nothing needs a socket or a service.

  python arctic_batch.py submit --queue-dir Q --image path.png [--priority 100] [--est-s 600] [--vram-gb 4.5] [--pin A|B] [--name N]
  python arctic_batch.py close  --queue-dir Q          (no more jobs: the runner finishes what is queued and exits)

Layout:  Q/<time>-<name>.job.json  new job (written as .tmp then renamed, so the runner never reads half a file)
         Q/claimed/                a job the runner took (moved there before it is queued: exactly one runner takes it)
         Q/rejected/               a bad job + <file>.why.txt (bad JSON, missing image, duplicate name)
         Q/CLOSE                   the close marker
Priority: higher starts first (default 0; 100 = urgent). Equal priority keeps arrival order (or longest first, --order
longest-first). A running generation is never interrupted: priority decides who starts next, not who stops.
"""
import json, os, re, time

NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def submit(qdir, image, name=None, priority=0, est_s=None, vram_gb=None, pin=None):
    """Add a job; returns the job file path. The image is stored as an absolute path."""
    image = os.path.abspath(image)
    name = name or os.path.splitext(os.path.basename(image))[0]
    if not NAME_RE.match(name):
        raise ValueError("bad job name %r (letters, digits, _ . - only)" % name)
    if pin not in (None, "A", "B"):
        raise ValueError("pin must be A or B")
    os.makedirs(qdir, exist_ok=True)
    job = {"name": name, "image": image, "priority": int(priority)}
    for k, v in (("est_s", est_s), ("vram_gb", vram_gb), ("pin", pin)):
        if v is not None:
            job[k] = v
    path = os.path.join(qdir, "%020d-%s.job.json" % (time.time_ns(), name))
    _write_atomic(path, json.dumps(job))
    return path


def close(qdir):
    os.makedirs(qdir, exist_ok=True)
    _write_atomic(os.path.join(qdir, "CLOSE"), "closed\n")


class LiveQueue:
    def __init__(self, qdir, log=print):
        self.qdir, self.log = qdir, log
        self.claimed_dir, self.rejected_dir = os.path.join(qdir, "claimed"), os.path.join(qdir, "rejected")
        for d in (qdir, self.claimed_dir, self.rejected_dir):
            os.makedirs(d, exist_ok=True)

    def is_closed(self):
        return os.path.exists(os.path.join(self.qdir, "CLOSE"))

    def _reject(self, fn, why):
        self.log("live queue: rejected %s: %s" % (fn, why))
        try:
            os.replace(os.path.join(self.qdir, fn), os.path.join(self.rejected_dir, fn))
            with open(os.path.join(self.rejected_dir, fn + ".why.txt"), "w", encoding="utf-8") as f:
                f.write(why + "\n")
        except OSError:
            pass

    def poll(self, known):
        """Claim every new job file, oldest first. `known` = names already in this batch (lower-case compare: NTFS).
        Returns a list of dicts (name, image, priority, est_s, vram_gb, pin); bad files go to rejected/."""
        out, seen = [], {k.lower() for k in known}
        try:
            files = sorted(f for f in os.listdir(self.qdir) if f.endswith(".job.json"))
        except OSError:
            return out
        for fn in files:
            src = os.path.join(self.qdir, fn)
            try:
                with open(src, encoding="utf-8") as f:
                    job = json.load(f)
                name, image = str(job["name"]), str(job["image"])
            except (OSError, ValueError, KeyError, TypeError) as e:
                self._reject(fn, "unreadable job file: %r" % (e,))
                continue
            if not NAME_RE.match(name):
                self._reject(fn, "bad job name %r" % name)
            elif name.lower() in seen:
                self._reject(fn, "duplicate job name %r (names are case-insensitive: they name the dump and the GLB)" % name)
            elif not os.path.isfile(image):
                self._reject(fn, "image not found: %s" % image)
            else:
                try:
                    os.replace(src, os.path.join(self.claimed_dir, fn))   # the claim: moved first, queued second
                except OSError:
                    continue
                seen.add(name.lower())
                out.append({"name": name, "image": image, "priority": int(job.get("priority", 0)),
                            "est_s": job.get("est_s"), "vram_gb": job.get("vram_gb"), "pin": job.get("pin")})
        return out
