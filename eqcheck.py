# usage: eqcheck.py <scriptdir> <calib> <lrf> <out.jpg> t:pa:rb ...
import torch, json, sys, subprocess, numpy as np, math
from PIL import Image, ImageDraw
sys.path.insert(0, sys.argv[1])
from geom360 import *
cal = json.load(open(sys.argv[2], encoding='utf-8')); lrf = sys.argv[3]; out = sys.argv[4]
specs = [tuple(map(float, a.split(':'))) for a in sys.argv[5:]]
s = 1024/3840; p = {}
for k, v in cal.items():
    v = float(v)
    if k in ('f0','f1'): v *= s
    elif k in ('cx0','cy0','cx1','cy1'): v = (v+0.5)*s-0.5
    p[k] = torch.tensor(v, device='cuda')
W, H = 1440, 720; d = equirect_dirs(W, H)
t2 = lambda x: torch.tensor(math.radians(x), device='cuda')
rows = []
for t, pa, rb in specs:
    a = subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',lrf,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'], capture_output=True).stdout
    x = torch.from_numpy(np.frombuffer(a, np.uint8).reshape(1024,2048,3).copy()).cuda().permute(2,0,1).float()[None]/255
    ims = []
    for (pp, rr) in [(0,0), (pa, rb)]:
        M = rotmat(t2(0), t2(pp), t2(rr)); dd = d @ M.T
        img = sample(dd, x[..., :1024], x[..., 1024:], p, size=1024, blend_deg=6)
        im = Image.fromarray((img.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        dr = ImageDraw.Draw(im)
        for xx in range(0, W, W//16): dr.line([(xx,0),(xx,H)], fill=(255,255,0), width=1)
        dr.line([(0,H//2),(W,H//2)], fill=(0,255,255), width=1)
        dr.text((5,5), f't{t} pa{pp} rb{rr}', fill=(255,0,0))
        ims.append(im)
    row = Image.new('RGB', (W*2, H)); row.paste(ims[0], (0,0)); row.paste(ims[1], (W,0)); rows.append(row)
sh = Image.new('RGB', (W*2, H*len(rows)))
for i, r in enumerate(rows): sh.paste(r, (0, i*H))
sh.resize((W, H*len(rows)//2)).save(out, quality=85)
