import torch, math, json, sys
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
from geom360 import *
dev = 'cuda'
caldir = sys.argv[2]
S = 1920  # half res for calibration
def load(fn):
    a = np.asarray(Image.open(fn).convert('RGB').resize((S, S), Image.BILINEAR), dtype=np.float32) / 255
    return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev)
frames = [(load(f'{caldir}/f{i}_0.png'), load(f'{caldir}/f{i}_1.png')) for i in range(5)]
# directions in seam band
N = 20000
g = torch.Generator(device='cpu').manual_seed(0)
th = torch.deg2rad(torch.empty(N).uniform_(83, 97, generator=g))
ph = torch.empty(N).uniform_(-math.pi, math.pi, generator=g)
d = torch.stack([torch.sin(th) * torch.cos(ph), torch.sin(th) * torch.sin(ph), torch.cos(th)], -1)
d = d[d[:, 1] < 0.6]  # drop bottom (stick / hand)
d = d.to(dev)[None]  # (1,N,3)
names = ['f0', 'cx0', 'cy0', 'f1', 'cx1', 'cy1', 'ry1', 'rp1', 'rr1', 'gain1', 'k0', 'k1']
init = dict(DEFAULT)
params = {k: torch.tensor(float(init[k]), device=dev, requires_grad=True) for k in names}
scale = {'f0': 1 / 100, 'cx0': 1 / 100, 'cy0': 1 / 100, 'f1': 1 / 100, 'cx1': 1 / 100, 'cy1': 1 / 100, 'ry1': 1, 'rp1': 1, 'rr1': 1, 'gain1': 10, 'k0': 100, 'k1': 100}
# optimize in normalized space
z = {k: torch.zeros((), device=dev, requires_grad=True) for k in names}
def P():
    return {k: torch.tensor(float(init[k]), device=dev) + z[k] / scale[k] for k in names}
opt = torch.optim.Adam(list(z.values()), lr=0.02)
def lossf(p):
    tot = 0
    for t0, t1 in frames:
        pp = dict(p)
        # half-res scaling of pixel params
        for k in ['f0', 'cx0', 'cy0', 'f1', 'cx1', 'cy1']:
            pp[k] = p[k] / 2
        c0, c1, _, _ = sample(d, t0, t1, pp, size=S, separate=True)
        # compare gradients-insensitive: use luminance L1
        tot = tot + (c0 - c1).abs().mean()
    return tot / len(frames)
with torch.no_grad():
    print('init loss', float(lossf(P())))
for it in range(600):
    opt.zero_grad()
    l = lossf(P())
    l.backward()
    opt.step()
    if it % 100 == 0 or it == 599:
        print(it, float(l), {k: round(float(v), 3) for k, v in P().items()})
res = {k: float(v) for k, v in P().items()}
json.dump(res, open(f'{caldir}/calib.json', 'w'), indent=1)
print('final', res)
