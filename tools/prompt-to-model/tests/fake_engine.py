#!/usr/bin/env python3
"""Stand-in for trellis-cli: checks the picture exists and writes a stub GLB. Env FAKE_ENGINE_FAIL=1 exits 3, FAKE_ENGINE_SLEEP=N waits N s."""
import os
import sys
import time

a = sys.argv[1:]
arg = lambda f: a[a.index(f) + 1]
time.sleep(float(os.environ.get("FAKE_ENGINE_SLEEP", "0")))
if os.environ.get("FAKE_ENGINE_FAIL"):
    sys.exit(3)
if not os.path.isfile(arg("-i")):
    sys.exit(2)
with open(arg("-o"), "wb") as f:
    f.write(b"glTF" + (" ".join(a)).encode())
