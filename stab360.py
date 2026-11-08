# Full 3-axis stabilization for the reframe: the camera's true orientation per source frame + a smooth view path.
# Two ways to get the orientation:
#   gyro : the camera's own fused attitude read from the .OSV (osvmeta.py), mapped to camera axes with gyrocmp.py
#   image: my image-based frame-to-frame rotations (rot360.py), integrated, gravity from upright2.py samples
# usage:
#   stab360.py gyro  <gyro.json> <gyrocal.json> <plans.json> <clip> <out.json> [heading_sigma_s=0.8]
#   stab360.py image <rot.json>  <up.json>      <plans.json> <clip> <out.json> [heading_sigma_s=0.8]
# output: fps, C[j] = 3x3 cam_from_world (row-major 9 floats), view path Y (world yaw deg), P (pitch deg), F (hfov deg)
# world: x right, y down (gravity), z forward-horizontal; camera: x right, y down, z = lens0 axis (geom360).
import sys, json, math, numpy as np
sys.path.insert(0, __file__.rsplit('\\', 1)[0].rsplit('/', 1)[0])
from gyrocmp import Gyro, q2m
from path360 import gauss, smoothstep, wrap

FPS = 30000 / 1001
P_DJI = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])  # DJI world (z down) -> my world (y down)

def orthonormalize(R):
    U, _, Vt = np.linalg.svd(R); d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1, 1, d]) @ Vt

def min_rot(a, b):  # smallest rotation taking unit a to unit b
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)

def gyro_C(gyro_path, cal_path, n=None):
    g = Gyro(json.load(open(gyro_path))); cal = json.load(open(cal_path))
    n = len(g.ft) if n is None else min(n, len(g.ft))
    X = np.array(cal['X_cam_from_body'])
    Rm = np.array([q2m(q) for q in g.at(g.ft[:n] + cal['dt_us'])])  # world_from_body
    if cal['convention'] != 'world_from_body': Rm = np.swapaxes(Rm, 1, 2)
    C = np.einsum('ij,nkj,lk->nil', X, Rm, P_DJI)  # X Rm^T P^T
    return C

def up_samples(up):
    S = up['samples']; out = []
    for s in S:
        pa, rb = math.radians(s['pa']), math.radians(s['rb'])
        cp, sp, cr, sr = math.cos(pa), math.sin(pa), math.cos(rb), math.sin(rb)
        Rx = np.array([[1, 0, 0], [0, cp, sp], [0, -sp, cp]]); Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
        out.append((s['t'], Rx @ Rz @ np.array([0.0, -1, 0]), max(s['score'] - 0.08, 0.01)))
    return out

def image_C(rot_path, up_path, n=None, sigma_s=3.0):
    rot = json.load(open(rot_path)); up = json.load(open(up_path))
    dR = rot['dR']; inl = rot['inl']; n = len(dR) if n is None else min(n, len(dR))
    C = np.zeros((n, 3, 3)); C[0] = np.eye(3); last = np.eye(3)
    for j in range(1, n):
        if dR[j] is not None and inl[j] >= 30: last = np.array(dR[j]).reshape(3, 3)
        C[j] = orthonormalize(last @ C[j - 1])
    # gravity: each up sample says where "up" is in the chain's (drifting) world; smooth robustly over time
    S = up_samples(up); fps = rot['fps']
    ts = np.array([s[0] for s in S]); w = np.array([s[2] for s in S])
    u0 = np.array([C[min(int(round(t * fps)), n - 1)].T @ u for t, u, _ in S])
    keep = np.ones(len(S), bool)
    for _ in range(2):
        tf = ts[keep]; uk = u0[keep]; wk = w[keep]
        for i in range(len(S)):
            K = np.exp(-0.5 * ((tf - ts[i]) / sigma_s) ** 2) * wk; m = (K[:, None] * uk).sum(0); m /= np.linalg.norm(m)
            keep[i] = math.degrees(math.acos(np.clip(np.dot(m, u0[i]), -1, 1))) < 6
    tf = np.arange(n) / fps
    K = np.exp(-0.5 * ((tf[:, None] - ts[keep][None, :]) / sigma_s) ** 2) * w[keep][None, :]
    U = K @ u0[keep]; U /= np.linalg.norm(U, axis=1, keepdims=True)
    for j in range(n):
        C[j] = C[j] @ min_rot(U[j], np.array([0.0, -1, 0])).T
    print(f'image: gravity samples kept {keep.sum()}/{len(S)}, drift of up over the clip {math.degrees(math.acos(np.clip(np.dot(U[0], U[-1]), -1, 1))):.2f} deg')
    return C

def heading(C):  # world yaw of the camera's lens0 axis, degrees, unwrapped (+ = right)
    f = np.einsum('nji,j->ni', C, np.array([0.0, 0, 1]))  # C^T e_z
    return np.degrees(np.unwrap(np.arctan2(f[:, 0], f[:, 2])))

def build_world_path(plan, h, fps, sigma_s=0.8):
    n = len(h); Hs = gauss(h, sigma_s * fps)
    # anchor of a world-locked shot: the yaw was chosen on a still of the camera's view at `ref`, so it is relative to
    # the camera's ACTUAL heading then (barely smoothed), not to the slow-following heading Hs. Using Hs here put the
    # view 20-45 deg off when the camera was being swung around at the ref time (clip 0047, 2026-11-08).
    Ha = gauss(h, 1.5)
    kfs = [dict(k) for k in plan]
    for k in kfs:
        if k['lock'] == 'world':
            k['_w'] = k['yaw'] + Ha[min(int(round(k.get('ref', k['t']) * fps)), n - 1)]
    def fr(k, i):
        return (k['yaw'] + Hs[i] if k['lock'] == 'cam' else k['_w']), k['pitch'], k['fov']
    Y = np.zeros(n); P = np.zeros(n); Fv = np.zeros(n); prev_dy = {}
    for i in range(n):
        t = i / fps; k = 0
        while k + 1 < len(kfs) and kfs[k + 1]['t'] <= t: k += 1
        y, p, f = fr(kfs[k], i)
        if k + 1 < len(kfs):
            nk = kfs[k + 1]; e = nk.get('ease', 1.5); t1 = nk['t']; t0 = max(t1 - e, kfs[k]['t'])
            if t > t0 and t1 > t0:
                s = smoothstep((t - t0) / (t1 - t0)); y2, p2, f2 = fr(nk, i)
                raw = y2 - y
                dy = prev_dy[k] + wrap(raw - prev_dy[k]) if k in prev_dy else wrap(raw)
                prev_dy[k] = dy
                y = y + s * dy; p = p + s * (p2 - p); f = f + s * (f2 - f)
        Y[i] = y; P[i] = p; Fv[i] = f
    Yu = np.degrees(gauss(np.unwrap(np.radians(Y)), 2.0))
    return Yu, gauss(P, 2.0), gauss(Fv, 2.0)

if __name__ == '__main__':
    mode, a1, a2, plans, clip, outp = sys.argv[1:7]
    sig = float(sys.argv[7]) if len(sys.argv) > 7 else 0.8
    C = gyro_C(a1, a2) if mode == 'gyro' else image_C(a1, a2)
    h = heading(C)
    plan = json.load(open(plans, encoding='utf-8'))[clip]
    Y, P, F = build_world_path(plan, h, FPS, sig)
    # how much the camera shakes vs. how much the view moves (deg per frame, median / p95)
    def stat(x): d = np.abs(np.diff(x)); return f'{np.median(d):.3f}/{np.percentile(d, 95):.3f}'
    print(f'{mode}: frames {len(C)}; camera heading change per frame {stat(h)}; view yaw change per frame {stat(Y)}')
    json.dump({'fps': FPS, 'mode': mode, 'C': C.reshape(len(C), 9).tolist(), 'Y': Y.tolist(), 'P': P.tolist(),
               'F': F.tolist(), 'h': h.tolist()}, open(outp, 'w'))
    print('wrote', outp)
