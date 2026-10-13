"""Fake judge: prints a score from the draft GLB's file name digits (a_2_d -> 2); name containing 'veto' exits 4."""
import os, re, sys
n = os.path.basename(sys.argv[1])
if "veto" in n: sys.exit(4)
print("noise line"); print(re.findall(r"\d+", n)[0] if re.findall(r"\d+", n) else "0")
