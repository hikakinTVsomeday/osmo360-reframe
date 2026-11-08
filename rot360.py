# Estimate the camera's full 3-axis rotation between consecutive frames, straight from the .OSV
# (both fisheye streams decoded small on the GPU). Image-based; the DJI gyro stream is not read.
# usage: rot360.py <calib.json> <osv> <out.json> [size=1024] [max_frames]
# output dR[j]: 3x3 rotation with v_j = dR[j] @ v_{j-1} for a fixed far-away direction (camera coords:
#   x right, y down, z = lens0 axis), plus inlier counts per frame.
import sys, json, subprocess, math, threading, queue, time
import numpy as np, cv2

cal = json.load(open(sys.argv[1], encoding='utf-8')); osv = sys.argv[2]; out = sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
maxf = int(sys.argv[5]) if len(sys.argv) > 5 else None
s = N / 3840

def rotmat_np(y, p, r):  # radians; identical to geom360.rotmat
    cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz

LENS = []
for i in (0, 1):
    LENS.append((cal[f'f{i}'] * s, (cal[f'cx{i}'] + 0.5) * s - 0.5, (cal[f'cy{i}'] + 0.5) * s - 0.5, cal[f'k{i}']))
R1 = rotmat_np(*(math.radians(cal[k]) for k in ('ry1', 'rp1', 'rr1')))
FLIP = np.array([-1.0, 1.0, -1.0])

def pix2dir(i, pts):
    f, cx, cy, k = LENS[i]
    dx = pts[:, 0] - cx; dy = pts[:, 1] - cy; r = np.hypot(dx, dy) + 1e-9
    th = r / f
    for _ in range(4):
        th = th - (th + k * th ** 3 - r / f) / (1 + 3 * k * th ** 2)
    st = np.sin(th)
    d = np.stack([st * dx / r, st * dy / r, np.cos(th)], 1)
    if i == 1:  # geom360.world_to_lens1: d1 = R1 @ flip(d)  ->  d = flip(R1^T d1)
        d = (d @ R1) * FLIP
    return d, th

TH_MAX = math.radians(86)
masks = []
for i in (0, 1):
    f, cx, cy, k = LENS[i]
    yy, xx = np.mgrid[0:N, 0:N]
    rmax = f * (TH_MAX + k * TH_MAX ** 3)
    masks.append(((np.hypot(xx - cx, yy - cy) < rmax) * 255).astype(np.uint8))

def kabsch(A, B):
    """best R with B ~ R A (rows are vectors). A,B (...,n,3)"""
    H = np.einsum('...ni,...nj->...ij', A, B)
    U, S, Vt = np.linalg.svd(H)
    V = np.swapaxes(Vt, -1, -2); Ut = np.swapaxes(U, -1, -2)
    d = np.linalg.det(V @ Ut)
    D = np.ones(S.shape); D[..., 2] = d
    return (V * D[..., None, :]) @ Ut

rng = np.random.default_rng(0)
THR = math.radians(0.3)
def ransac_rot(A, B, it=256):
    n = len(A)
    if n < 12: return None
    idx = rng.integers(0, n, (it, 3))
    Rs = kabsch(A[idx], B[idx])
    err = np.linalg.norm(np.einsum('kij,nj->kni', Rs, A) - B[None], axis=2)
    m = err[int((err < THR).sum(1).argmax())] < THR
    e = None
    for _ in range(3):
        if m.sum() < 8: return None
        R = kabsch(A[m], B[m]); e = np.linalg.norm(A @ R.T - B, axis=1); m = e < THR
    return R, int(m.sum()), n, float(np.degrees(np.median(e[m]))) if m.any() else 9.0

FS = N * N * 3 // 2
def dec(stream):
    vf = f'scale_cuda=w={N}:h={N}:format=nv12,hwdownload,format=nv12'
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda', '-i', osv,
                             '-map', f'0:v:{stream}', '-vf', vf, '-f', 'rawvideo', '-'], stdout=subprocess.PIPE, bufsize=FS * 4)
def reader(proc, q):
    while True:
        b = proc.stdout.read(FS)
        if len(b) < FS: q.put(None); return
        q.put(b)
d0, d1 = dec(0), dec(1)
q0, q1 = queue.Queue(maxsize=8), queue.Queue(maxsize=8)
threading.Thread(target=reader, args=(d0, q0), daemon=True).start()
threading.Thread(target=reader, args=(d1, q1), daemon=True).start()

LK = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
prev = None; dR = [np.eye(3).ravel().tolist()]; inl = [0]; npts = [0]; res = [0.0]
j = 0; t0 = time.time()
while True:
    b0 = q0.get(); b1 = q1.get()
    if b0 is None or b1 is None: break
    g = [np.frombuffer(b, np.uint8, N * N).reshape(N, N).copy() for b in (b0, b1)]
    if prev is not None:
        A = []; Bv = []
        for i in (0, 1):
            p0 = cv2.goodFeaturesToTrack(prev[i], 450, 0.004, N / 42, mask=masks[i], blockSize=7)
            if p0 is None: continue
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev[i], g[i], p0, None, **LK)
            pb, st2, _ = cv2.calcOpticalFlowPyrLK(g[i], prev[i], p1, None, **LK)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.4)
            if ok.sum() < 3: continue
            a, _ = pix2dir(i, p0.reshape(-1, 2)[ok]); b, thb = pix2dir(i, p1.reshape(-1, 2)[ok])
            keep = thb < math.radians(90)
            A.append(a[keep]); Bv.append(b[keep])
        r = ransac_rot(np.concatenate(A), np.concatenate(Bv)) if A else None
        if r is None:
            dR.append(None); inl.append(0); npts.append(0); res.append(9.0)
        else:
            dR.append(r[0].ravel().tolist()); inl.append(r[1]); npts.append(r[2]); res.append(r[3])
    prev = g; j += 1
    if j % 300 == 0: print(f'{j} frames {time.time() - t0:.0f}s  last inliers {inl[-1]}/{npts[-1]}', flush=True)
    if maxf and j >= maxf: break
for d in (d0, d1): d.kill()
json.dump({'fps': 30000 / 1001, 'N': N, 'dR': dR, 'inl': inl, 'npts': npts, 'res_deg': res}, open(out, 'w'))
iv = np.array(inl[1:])
print('done', out, j, 'frames', f'{time.time() - t0:.0f}s', 'inliers median', int(np.median(iv)), 'min', int(iv.min()),
      'failed', sum(1 for x in dR if x is None))
