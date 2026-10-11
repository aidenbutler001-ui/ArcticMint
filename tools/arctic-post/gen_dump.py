#!/usr/bin/env python3
"""Synthetic TRELLIS_DUMP_POST generator: surface-nets mesh of a noisy SDF on an N^3 grid.
Layout: i32 V,F,Mv,res; f32 verts[V*3]; i32 faces[F*3]; i32 coords[Mv*3]; f32 pbr6[Mv*6]
usage: gen_dump.py out.bin --N 1024 --scale 1.0 --shell 0.0 --amp 1.0 --rivet 1 --pbr-res 512
"""
import argparse, time, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--N", type=int, default=1024)
ap.add_argument("--scale", type=float, default=1.0)   # object size multiplier
ap.add_argument("--shell", type=float, default=0.0)   # >0: hollow, wall thickness in world units (adds inner surface)
ap.add_argument("--amp", type=float, default=1.0)     # noise amplitude multiplier
ap.add_argument("--rivet", type=int, default=1)       # rivet / greeble repetition on/off
ap.add_argument("--pbr-res", type=int, default=512)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--dup", type=float, default=0.05)    # fraction of vertices snapped onto a neighbour (weld work)
ap.add_argument("--holes", type=int, default=3000)    # quads removed -> small holes
ap.add_argument("--flipfrac", type=float, default=0.10)
a = ap.parse_args()
N = a.N; B = 16
rng = np.random.default_rng(a.seed)
t0 = time.time()

# ---- value noise ----
OCT = [(48.0, 1.6), (24.0, 1.0), (12.0, 0.7), (6.0, 0.45), (3.0, 0.3), (1.7, 0.2)]  # (wavelength vox, amp vox)
nz = []
for wl, am in OCT:
    g = int(N / wl) + 3
    nz.append((wl, am * a.amp, rng.standard_normal((g, g, g)).astype(np.float32)))
def noise(px, py, pz):  # coords in voxel units
    out = np.zeros(px.shape, np.float32)
    for wl, am, tab in nz:
        x = px / wl; y = py / wl; z = pz / wl
        xi = np.floor(x).astype(np.int32); yi = np.floor(y).astype(np.int32); zi = np.floor(z).astype(np.int32)
        fx = x - xi; fy = y - yi; fz = z - zi
        fx = fx*fx*(3-2*fx); fy = fy*fy*(3-2*fy); fz = fz*fz*(3-2*fz)
        c = 0
        for dx in (0, 1):
            wx = fx if dx else 1 - fx
            for dy in (0, 1):
                wy = fy if dy else 1 - fy
                for dz in (0, 1):
                    wz = fz if dz else 1 - fz
                    c = c + tab[xi+dx, yi+dy, zi+dz] * (wx*wy*wz)
        out += am * c
    return out  # in voxels

# ---- base SDF (world units, cube [-.5,.5]); ~1-Lipschitz ----
S = a.scale
def box(px, py, pz, c, h, r):
    qx = np.abs(px-c[0]) - h[0] + r; qy = np.abs(py-c[1]) - h[1] + r; qz = np.abs(pz-c[2]) - h[2] + r
    o = np.sqrt(np.maximum(qx,0)**2 + np.maximum(qy,0)**2 + np.maximum(qz,0)**2)
    return o + np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0) - r
def cyl_x(px, py, pz, c, rad, hl):
    d1 = np.sqrt((py-c[1])**2 + (pz-c[2])**2) - rad
    d2 = np.abs(px-c[0]) - hl
    return np.sqrt(np.maximum(d1,0)**2 + np.maximum(d2,0)**2) + np.minimum(np.maximum(d1,d2),0)
def base(px, py, pz):
    px = px/S; py = py/S; pz = pz/S
    body = box(px, py, pz, (0,-0.08,0), (0.30,0.13,0.20), 0.01)
    lid = np.maximum(cyl_x(px, py, pz, (0,0.05,0), 0.20, 0.30), -(py - 0.05) )  # half cylinder above y=.05
    f = np.minimum(body, lid)
    for xb in (-0.2, 0.0, 0.2):   # metal bands slightly proud
        band = np.maximum(np.abs(px-xb) - 0.03, np.minimum(box(px,py,pz,(0,-0.08,0),(0.32,0.15,0.22),0.01), cyl_x(px,py,pz,(0,0.05,0),0.22,0.32)))
        f = np.minimum(f, band)
    lock = box(px, py, pz, (0,-0.02,0.215), (0.04,0.05,0.02), 0.005); f = np.minimum(f, lock)
    gem = np.sqrt((px-0.0)**2 + (py-0.2)**2 + (pz-0.0)**2) - 0.03; f = np.minimum(f, gem)
    feet = np.minimum.reduce([box(px,py,pz,(sx,-0.23,sz),(0.03,0.03,0.03),0.01) for sx in (-0.26,0.26) for sz in (-0.16,0.16)])
    f = np.minimum(f, feet)
    if a.rivet:
        # domain-repeated rivets on the front face of the body (spacing .03)
        sp = 0.03
        qx = (px + 0.5*sp) % sp - 0.5*sp; qy = (py + 0.5*sp) % sp - 0.5*sp
        riv = np.sqrt(qx**2 + qy**2 + (pz-0.205)**2) - 0.009
        riv = np.maximum(riv, np.maximum(np.abs(px)-0.28, np.abs(py+0.08)-0.11))
        f = np.minimum(f, riv)
    if a.shell > 0:   # hollow: keep a wall of given thickness
        f = np.maximum(f, -(f + a.shell/S))
    return f*S

def field(px, py, pz):  # voxel-unit lattice coords -> sdf in voxels
    wx = px/N - 0.5; wy = py/N - 0.5; wz = pz/N - 0.5
    return base(wx, wy, wz).astype(np.float32)*N + noise(px, py, pz)

# ---- block culling ----
nb = N//B
cx = (np.arange(nb)*B + B/2)
gx, gy, gz = np.meshgrid(cx, cx, cx, indexing='ij')
fb = base(gx/N-0.5, gy/N-0.5, gz/N-0.5).astype(np.float32)*N
amp_max = sum(am for _, am, _ in nz) * 1.2 + 1.0
keep = np.abs(fb) < (B*0.87 + amp_max + 2.0)   # half diagonal ~ 13.9 voxels
# Lipschitz of 'S' scaled base is 1; noise adds at most amp_max
bl = np.argwhere(keep)
print(f"blocks near surface: {len(bl)}/{nb**3}  ({time.time()-t0:.1f}s)", flush=True)

# ---- evaluate per block, collect crossings ----
L = B+1
ii, jj, kk = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
rec_cell = []; rec_pt = []; quad_cells = []; quad_dir = []
BATCH = 128
for s in range(0, len(bl), BATCH):
    bb = bl[s:s+BATCH]*B                      # (nb,3) origin lattice
    px = (bb[:,0,None,None,None] + ii).astype(np.float32); py = (bb[:,1,None,None,None] + jj).astype(np.float32); pz = (bb[:,2,None,None,None] + kk).astype(np.float32)
    f = field(px, py, pz)                     # (nb,L,L,L)
    neg = f < 0
    for ax in range(3):
        sl0 = [slice(None)]*4; sl1 = [slice(None)]*4
        sl0[ax+1] = slice(0, B); sl1[ax+1] = slice(1, B+1)
        # restrict other axes to owned range [0,B)
        for o in range(3):
            if o != ax: sl0[o+1] = slice(0, B); sl1[o+1] = slice(0, B)
        f0 = f[tuple(sl0)]; f1 = f[tuple(sl1)]
        n0 = neg[tuple(sl0)]; n1 = neg[tuple(sl1)]
        cr = n0 != n1
        bi, a0, a1, a2 = np.nonzero(cr)
        if len(bi) == 0: continue
        loc = np.stack([a0, a1, a2], 1)
        v0 = f0[cr]; v1 = f1[cr]
        t = v0/(v0 - v1)
        t = np.clip(t, 1e-3, 1-1e-3).astype(np.float32)
        start = loc + bb[bi]                          # global lattice start (i,j,k)
        pt = start.astype(np.float32); pt[:, ax] += t
        out_dir = (v1 > v0)                           # f increases along +axis -> outward normal +axis
        u = (ax+1) % 3; v = (ax+2) % 3
        offs = [(-1,-1),(0,-1),(0,0),(-1,0)]
        cells = []
        for (du, dv) in offs:
            c = start.copy(); c[:, u] += du; c[:, v] += dv
            cells.append(c)
        cells = np.stack(cells, 1)                    # (n,4,3)
        quad_cells.append(cells.astype(np.int32)); quad_dir.append(out_dir)
        rec_pt.append(pt)
    if (s//BATCH) % 20 == 0: print(f"  batch {s//BATCH}/{(len(bl)+BATCH-1)//BATCH}  quads so far {sum(len(q) for q in quad_cells)} ({time.time()-t0:.0f}s)", flush=True)
qc = np.concatenate(quad_cells); qd = np.concatenate(quad_dir); qp = np.concatenate(rec_pt)
del quad_cells, quad_dir, rec_pt
nq = len(qc); print(f"quads {nq} ({time.time()-t0:.0f}s)", flush=True)
# valid cells within [0,N)
ok = ((qc >= 0) & (qc < N)).all(axis=(1,2)); qc = qc[ok]; qd = qd[ok]; qp = qp[ok]; nq = len(qc)
lin = (qc[:,:,0].astype(np.int64)*N + qc[:,:,1])*N + qc[:,:,2]      # (nq,4)
uniq, inv = np.unique(lin.ravel(), return_inverse=True)
inv = inv.reshape(nq, 4).astype(np.int32)
V = len(uniq)
sums = np.zeros((V,3), np.float64); cnt = np.zeros(V, np.int32)
for k in range(4):
    np.add.at(sums, inv[:,k], qp.astype(np.float64)) if False else None
for c3 in range(3):
    for k in range(4):
        sums[:, c3] += np.bincount(inv[:,k], weights=qp[:,c3], minlength=V)
for k in range(4): cnt += np.bincount(inv[:,k], minlength=V).astype(np.int32)
vpos = (sums/cnt[:,None]).astype(np.float32)          # lattice coords (cell vertex ~ mean of crossings)
del sums, qp
print(f"cell vertices V={V} ({time.time()-t0:.0f}s)", flush=True)
# ---- pbr coords: active cells at pbr-res (unique of cell>>shift) ----
cell_xyz = np.stack([uniq//(N*N), (uniq//N) % N, uniq % N], 1).astype(np.int32)
# ---- dup snapping: move a fraction of verts onto a neighbouring vert (+tiny noise <1/16 voxel) ----
if a.dup > 0:
    nd = int(V*a.dup); idx = rng.choice(V, nd, replace=False)
    # neighbour = vertex of a quad-adjacent cell
    q = rng.integers(0, nq, nd)
    # choose any vertex belonging to a random quad that contains idx? cheap proxy: pick a random quad containing it
    first_quad = np.full(V, -1, np.int64); first_quad[inv[::-1,0]] = np.arange(nq)[::-1]
    tgt_q = first_quad[idx]; okq = tgt_q >= 0
    idx = idx[okq]; tgt_q = tgt_q[okq]
    nbv = inv[tgt_q, rng.integers(0, 4, len(idx))]
    vpos[idx] = vpos[nbv] + rng.uniform(-0.06, 0.06, (len(idx),3)).astype(np.float32)
# ---- triangles ----
q0, q1, q2, q3 = inv[:,0], inv[:,1], inv[:,2], inv[:,3]
tris = np.empty((nq, 2, 3), np.int32)
# outward => order as given; else reversed
tris[:,0] = np.stack([q0,q1,q2],1); tris[:,1] = np.stack([q0,q2,q3],1)
rev = ~qd
tris[rev, 0] = np.stack([q0,q2,q1],1)[rev]; tris[rev, 1] = np.stack([q0,q3,q2],1)[rev]
# winding patches flipped (blocks of 32^3 cells)
if a.flipfrac > 0:
    qb = (qc[:,0,:] // 32)
    h = ((qb[:,0].astype(np.int64)*73856093) ^ (qb[:,1].astype(np.int64)*19349663) ^ (qb[:,2].astype(np.int64)*83492791) ^ a.seed) % 1000
    fl = h < a.flipfrac*1000
    tris[fl] = tris[fl][:, :, ::-1]
# holes: drop random quads
if a.holes > 0:
    drop = rng.choice(nq, min(a.holes, nq//10), replace=False)
    m = np.ones(nq, bool); m[drop] = False
    tris = tris[m]
faces = tris.reshape(-1, 3)
# floaters: small spheres' worth of tiny disconnected triangles fans (40 blobs of 12 tris)
fl_v = []; fl_f = []
base_id = V
for b in range(40):
    c = rng.uniform(0.15, 0.85, 3)*N; r = rng.uniform(1.5, 4)
    d = rng.standard_normal((8,3)); d /= np.linalg.norm(d,axis=1)[:,None]
    fl_v.append((c + r*d).astype(np.float32))
    ring = np.arange(8)
    fl_f.append(np.stack([np.full(8, base_id), base_id+1+ring%7, base_id+1+(ring+1)%7],1))  # fan around vertex 0
    base_id += 8
vpos = np.concatenate([vpos] + fl_v); faces = np.concatenate([faces] + [x.astype(np.int32) for x in fl_f])
verts_w = (vpos/N - 0.5).astype(np.float32)
V = len(verts_w); F = len(faces)
# ---- PBR sparse volume at pbr_res ----
pr = a.pbr_res; sh = N//pr
pc = np.unique(((cell_xyz//sh).astype(np.int64)*np.array([pr*pr, pr, 1])).sum(1))
coords = np.stack([pc//(pr*pr), (pc//pr)%pr, pc%pr], 1).astype(np.int32)
Mv = len(coords)
w = (coords.astype(np.float32)/pr - 0.5)
pn = noise(coords[:,0]*float(N/pr), coords[:,1]*float(N/pr), coords[:,2]*float(N/pr))
pbr = np.empty((Mv, 6), np.float32)
pbr[:,0] = np.clip(0.55 + 0.35*np.sin(w[:,0]*40) + 0.1*pn, 0, 1)
pbr[:,1] = np.clip(0.35 + 0.3*np.sin(w[:,1]*25+1) + 0.1*pn, 0, 1)
pbr[:,2] = np.clip(0.25 + 0.3*np.cos(w[:,2]*30) + 0.1*pn, 0, 1)
pbr[:,3] = np.clip(0.5 + 0.5*np.sin(w[:,0]*12), 0, 1)     # metallic
pbr[:,4] = np.clip(0.6 + 0.3*pn, 0, 1)                    # roughness
pbr[:,5] = 1.0
with open(a.out, "wb") as fo:
    np.array([V, F, Mv, pr], np.int32).tofile(fo)
    verts_w.tofile(fo); faces.astype(np.int32).tofile(fo); coords.tofile(fo); pbr.tofile(fo)
print(f"wrote {a.out}: V={V} F={F} Mv={Mv} res={pr}  ({time.time()-t0:.0f}s)")
