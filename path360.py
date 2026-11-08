import numpy as np, json
def wrap(a): return (a + 180.0) % 360.0 - 180.0
def smoothstep(u): u = min(max(u, 0.0), 1.0); return u*u*(3-2*u)
def gauss(x, s):
    if s <= 0: return x
    r = int(3*s)+1; k = np.exp(-0.5*(np.arange(-r, r+1)/s)**2); k /= k.sum()
    xp = np.pad(x, r, mode='edge'); return np.convolve(xp, k, mode='valid')
def build_path(plan, motion, n):
    fps = motion['fps']; R = np.array(motion['R'][:n], float)
    if len(R) < n: R = np.pad(R, (0, n-len(R)), mode='edge')
    R = gauss(R, 1.5)
    kfs = [dict(k) for k in plan]
    for k in kfs:
        if k['lock'] == 'world':
            i = min(int(round(k.get('ref', k['t'])*fps)), n-1); k['_w'] = k['yaw'] + R[i]
    def fr(k, i):
        y = k['yaw'] if k['lock'] == 'cam' else k['_w'] - R[i]
        return y, k['pitch'], k['fov']
    Y = np.zeros(n); P = np.zeros(n); Fv = np.zeros(n)
    prev_dy = {}
    for i in range(n):
        t = i / fps
        k = 0
        while k+1 < len(kfs) and kfs[k+1]['t'] <= t: k += 1
        y, p, f = fr(kfs[k], i)
        if k+1 < len(kfs):
            nk = kfs[k+1]; e = nk.get('ease', 1.5); t1 = nk['t']; t0 = max(t1 - e, kfs[k]['t'])
            if t > t0 and t1 > t0:
                s = smoothstep((t - t0)/(t1 - t0))
                y2, p2, f2 = fr(nk, i)
                raw = y2 - y
                if k in prev_dy: dy = prev_dy[k] + wrap(raw - prev_dy[k])
                else: dy = wrap(raw)
                prev_dy[k] = dy
                y = y + s*dy; p = p + s*(p2-p); f = f + s*(f2-f)
        Y[i] = y; P[i] = p; Fv[i] = f
    Yu = np.unwrap(np.radians(Y)); Yu = np.degrees(gauss(Yu, 2.0))
    return wrap(Yu), gauss(P, 2.0), gauss(Fv, 2.0)

def level_series(up, n, fps, sigma=1.2, rej=10.0):
    S = up['samples']
    t = np.array([s['t'] for s in S]); pa = np.array([s['pa'] for s in S]); rb = np.array([s['rb'] for s in S])
    w = np.array([max(s['score'] - 0.08, 0.01) for s in S])
    keep = np.ones(len(S), bool)
    def wmed(v, ww):
        o = np.argsort(v); c = np.cumsum(ww[o]); return v[o][np.searchsorted(c, c[-1]/2)]
    for i in range(len(S)):
        m = (np.abs(t - t[i]) <= 2.6)
        if m.sum() >= 3:
            if abs(pa[i] - wmed(pa[m], w[m])) > rej or abs(rb[i] - wmed(rb[m], w[m])) > rej: keep[i] = False
    t, pa, rb, w = t[keep], pa[keep], rb[keep], w[keep]
    tf = np.arange(n) / fps
    K = np.exp(-0.5*((tf[:, None] - t[None, :])/sigma)**2) * w[None, :]
    K /= K.sum(1, keepdims=True) + 1e-9
    return K @ pa, K @ rb
