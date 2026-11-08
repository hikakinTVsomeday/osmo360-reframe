# Compare the camera's own fused attitude (osvmeta.py) with the image-based rotations (rot360.py):
# find the axis mapping body->camera, the quaternion direction convention and the time offset.
# usage: gyrocmp.py <gyro.json> <rot.json> [out_calib.json]
import sys, json, math, numpy as np

def q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])

def logm(R):  # rotation vector (rad), works batched (...,3,3)
    tr = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1); a = np.arccos(tr)
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.where(a < 1e-8, 0.5, a / (2 * np.sin(np.maximum(a, 1e-8))))
    return v * s[..., None]

def kabsch(A, B, w=None):  # B ~ R A
    w = np.ones(len(A)) if w is None else w
    H = (A * w[:, None]).T @ B
    U, S, Vt = np.linalg.svd(H); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T

class Gyro:
    def __init__(self, g):
        T = []; Q = []
        fr = g['frames']
        # Attitude timestamps wrap at 2**32 microseconds while frame timestamps
        # remain 64-bit. Align each attitude packet with its own frame clock.
        wrap_period = 2 ** 32
        attitude_times = [
            r['att_ts'] + round((r['frame_ts'] - r['att_ts']) / wrap_period) * wrap_period
            if r.get('att_ts') is not None else None
            for r in fr
        ]
        for j, r in enumerate(fr):
            qs = r.get('q') or []
            if not qs: continue
            t0 = attitude_times[j]
            t1 = attitude_times[j + 1] if j + 1 < len(fr) and attitude_times[j + 1] is not None else t0 + 1006 * len(qs)
            for i, q in enumerate(qs):
                T.append(t0 + (t1 - t0) * i / len(qs)); Q.append(q)
        Q = np.array(Q, float); Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        for i in range(1, len(Q)):
            if np.dot(Q[i], Q[i - 1]) < 0: Q[i] = -Q[i]
        self.T = np.array(T, float); self.Q = Q
        self.ft = np.array([r['frame_ts'] for r in fr], float)
    def at(self, t):  # nlerp, t in microseconds (array)
        i = np.clip(np.searchsorted(self.T, t) - 1, 0, len(self.T) - 2)
        f = np.clip((t - self.T[i]) / (self.T[i + 1] - self.T[i]), 0, 1)[:, None]
        q = self.Q[i] * (1 - f) + self.Q[i + 1] * f
        return q / np.linalg.norm(q, axis=1, keepdims=True)

def rel_body(Rm, conv):
    # Rm[j]: rotation matrix of the attitude at frame j. returns body-frame motion dB[j] (v_j = dB v_{j-1})
    if conv == 'world_from_body':  # v_world = R v_body -> v_body = R^T w
        return np.einsum('nji,njk->nik', Rm[1:], Rm[:-1])  # R_j^T R_{j-1}
    return np.einsum('nij,nkj->nik', Rm[1:], Rm[:-1])      # R_j R_{j-1}^T

if __name__ == '__main__':
    g = Gyro(json.load(open(sys.argv[1]))); rot = json.load(open(sys.argv[2]))
    n = min(len(g.ft), len(rot['dR']))
    dR = np.array([np.eye(3).ravel() if x is None else x for x in rot['dR'][:n]]).reshape(-1, 3, 3)
    inl = np.array(rot['inl'][:n], float)
    wi = logm(dR[1:]); good = inl[1:] >= 150
    best = None
    for conv in ('world_from_body', 'body_from_world'):
        for dt in np.arange(-60000, 60001, 1000):
            Rm = np.array([q2m(q) for q in g.at(g.ft[:n] + dt)])
            wb = logm(rel_body(Rm, conv))
            X = kabsch(wb[good], wi[good])
            res = np.linalg.norm(wb[good] @ X.T - wi[good], axis=1)
            score = np.sqrt(np.mean(res ** 2))
            if best is None or score < best[0]: best = (score, conv, dt, X)
    score, conv, dt, X = best
    # refine dt at 100 us
    for d2 in np.arange(dt - 1000, dt + 1001, 100):
        Rm = np.array([q2m(q) for q in g.at(g.ft[:n] + d2)])
        wb = logm(rel_body(Rm, conv)); X2 = kabsch(wb[good], wi[good])
        s2 = np.sqrt(np.mean(np.linalg.norm(wb[good] @ X2.T - wi[good], axis=1) ** 2))
        if s2 < score: score, dt, X = s2, d2, X2
    Rm = np.array([q2m(q) for q in g.at(g.ft[:n] + dt)])
    wb = logm(rel_body(Rm, conv)) @ X.T
    res = np.degrees(np.linalg.norm(wb - wi, axis=1))
    mag = np.degrees(np.linalg.norm(wi, axis=1))
    print(f'convention {conv}, time offset {dt / 1000:.1f} ms')
    print('body->camera X =\n', np.round(X, 3))
    print(f'per-frame rotation: image median {np.median(mag[good]):.3f} deg, p95 {np.percentile(mag[good], 95):.3f}')
    print(f'difference gyro vs image: median {np.median(res[good]):.4f} deg, p95 {np.percentile(res[good], 95):.4f}, rms {np.degrees(score):.4f}')
    # world up in camera coords from the gyro, per world axis candidate
    for name, wv in (('+Z', [0, 0, 1]), ('-Z', [0, 0, -1]), ('+Y', [0, 1, 0]), ('-Y', [0, -1, 0]), ('+X', [1, 0, 0]), ('-X', [-1, 0, 0])):
        wv = np.array(wv, float)
        up_b = np.einsum('nji,j->ni', Rm, wv) if conv == 'world_from_body' else np.einsum('nij,j->ni', Rm, wv)
        up_c = up_b @ X.T
        print(f'  world {name} as seen by the camera (frame 0, mid, end):', np.round(up_c[[0, n // 2, n - 1]], 3).tolist())
    if len(sys.argv) > 3:
        json.dump({'convention': conv, 'dt_us': float(dt), 'X_cam_from_body': X.tolist(),
                   'diff_median_deg': float(np.median(res[good]))}, open(sys.argv[3], 'w'), indent=1)
