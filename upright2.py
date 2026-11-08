# estimate camera up vector from straight vertical segments. M = rotmat(0,pa,rb); d_cam = M d_level; up_cam = M (0,-1,0)
# usage: upright2.py <scriptdir> <calib> <lrf> <out.json> <step_s>
import torch, json, sys, subprocess, numpy as np, math, cv2
sys.path.insert(0, sys.argv[1])
from geom360 import *
cal = json.load(open(sys.argv[2], encoding='utf-8')); lrf = sys.argv[3]; out = sys.argv[4]; step = float(sys.argv[5])
s = 1024/3840; p = {}
for k, v in cal.items():
    v = float(v)
    if k in ('f0','f1'): v *= s
    elif k in ('cx0','cy0','cx1','cy1'): v = (v+0.5)*s-0.5
    p[k] = torch.tensor(v, device='cuda')
N = 512; HF = 90.0; fx = (N/2)/math.tan(math.radians(HF)/2)
views = [(y, 0.0) for y in range(0, 360, 45)] + [(y, 30.0) for y in range(0, 360, 90)] + [(y, -30.0) for y in range(45, 360, 90)]
t2 = lambda a: torch.tensor(math.radians(a), device='cuda')
VR = []; VD = []
for (yw, pt) in views:
    R = rotmat(t2(yw), t2(-pt), t2(0.0)); VR.append(R.cpu().numpy()); VD.append(persp_dirs(N, N, yw, pt, 0.0, HF))
lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
def pix2dir(x, y, R):
    v = np.stack([(x + 0.0 - N/2)/fx, (y - N/2)/fx, np.ones_like(x)], -1)
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    return v @ R.T
# candidate grid
PA, RB = np.meshgrid(np.arange(-40, 40.1, 1.0), np.arange(-40, 40.1, 1.0), indexing='ij')
def up_of(pa, rb):
    a = np.radians(pa); b = np.radians(rb)
    out = []
    for ai, bi in zip(a.ravel(), b.ravel()):
        M = rotmat(torch.tensor(0.0), torch.tensor(float(ai)), torch.tensor(float(bi))).numpy()
        out.append(M @ np.array([0, -1.0, 0]))
    return np.array(out)
UP = up_of(PA, RB)  # [K,3]
def estimate(x):
    normals = []; wts = []
    for R, d in zip(VR, VD):
        img = sample(d, x[..., :1024], x[..., 1024:], p, size=1024, blend_deg=6)
        g = (img.mean(0).clamp(0, 1).cpu().numpy()*255).astype(np.uint8)
        segs = lsd.detect(g)[0]
        if segs is None: continue
        segs = segs[:, 0, :]
        L = np.hypot(segs[:, 2]-segs[:, 0], segs[:, 3]-segs[:, 1])
        ang = np.degrees(np.arctan2(np.abs(segs[:, 2]-segs[:, 0]), np.abs(segs[:, 3]-segs[:, 1])))  # 0 = vertical
        keep = (L > 25) & (ang < 40)
        segs = segs[keep]; L = L[keep]
        if len(segs) == 0: continue
        a = pix2dir(segs[:, 0], segs[:, 1], R); b = pix2dir(segs[:, 2], segs[:, 3], R)
        n = np.cross(a, b); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        normals.append(n); wts.append(L)
    if not normals: return None
    n = np.concatenate(normals); w = np.concatenate(wts)
    dots = np.abs(n @ UP.T)  # [S,K]
    e = np.degrees(np.arcsin(np.clip(dots, 0, 1)))
    sc = (w[:, None] * np.exp(-(e/1.5)**2)).sum(0)
    k = int(sc.argmax())
    tot = w.sum()
    base_k = int(np.argmin(np.abs(PA.ravel()) + np.abs(RB.ravel())))
    return float(PA.ravel()[k]), float(RB.ravel()[k]), float(sc[k]/tot), float(sc[base_k]/tot), len(n)
proc = subprocess.Popen(['ffmpeg','-v','error','-i',lrf,'-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE)
fs = 2048*1024*3; i = 0; nexts = 0.0; fps = 30000/1001; recs = []
while True:
    buf = proc.stdout.read(fs)
    if len(buf) < fs: break
    t = i/fps
    if t >= nexts:
        a = np.frombuffer(buf, np.uint8).reshape(1024, 2048, 3)
        x = torch.from_numpy(a.copy()).cuda().permute(2,0,1).float()[None]/255
        with torch.no_grad(): r = estimate(x)
        if r:
            recs.append({'t': round(t, 3), 'pa': r[0], 'rb': r[1], 'score': r[2], 'base': r[3], 'n': r[4]})
            print(f'{t:6.1f} pa {r[0]:6.1f} rb {r[1]:6.1f} score {r[2]:.3f} base {r[3]:.3f} n {r[4]}', flush=True)
        nexts += step
    i += 1
proc.stdout.close(); proc.wait()
json.dump({'fps': fps, 'samples': recs}, open(out, 'w'))
