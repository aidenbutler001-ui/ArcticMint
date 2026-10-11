#!/usr/bin/env python3
"""Exit 0 if two GLBs match exactly, ignoring asset.extras.generated in the JSON chunk."""
import json, struct, sys

def load(path):
    d = open(path, 'rb').read()
    magic, ver, total = struct.unpack_from('<4sII', d, 0)
    if magic != b'glTF' or total != len(d):
        sys.exit(f'{path}: bad GLB header')
    chunks, off = [], 12
    while off < len(d):
        n, t = struct.unpack_from('<I4s', d, off)
        chunks.append((t, d[off + 8:off + 8 + n]))
        off += 8 + n
    out = []
    for t, b in chunks:
        if t == b'JSON':
            j = json.loads(b.rstrip(b' \x00'))
            j.get('asset', {}).get('extras', {}).pop('generated', None)
            out.append((t, json.dumps(j, sort_keys=True)))
        else:
            out.append((t, b))
    return out

a, b = load(sys.argv[1]), load(sys.argv[2])
if a != b:
    sys.exit('GLBs differ beyond the generated timestamp')
print('GLBs identical (ignoring generated timestamp)')
