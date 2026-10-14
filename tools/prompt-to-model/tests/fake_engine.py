#!/usr/bin/env python3
"""Stand-in for trellis-cli: checks the picture exists and writes a stub GLB. Env FAKE_ENGINE_FAIL=1 exits 3."""
import os
import sys

a = sys.argv[1:]
arg = lambda f: a[a.index(f) + 1]
if os.environ.get("FAKE_ENGINE_FAIL"):
    sys.exit(3)
if not os.path.isfile(arg("-i")):
    sys.exit(2)
with open(arg("-o"), "wb") as f:
    f.write(b"glTF" + (" ".join(a)).encode())
