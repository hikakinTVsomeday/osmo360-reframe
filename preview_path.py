# render reframed preview sheet from LRF. usage: preview_path.py <scriptdir> <calib> <lrf> <motion> <plans> <clip> <outdir> <step>
import torch, json, sys, subprocess, numpy as np, math, os
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, sys.argv[1])
from geom360 import *; from path360 import build_path, level_series
cal = json.load(open(sys.argv[2])); lrf = sys.argv[3]; motion = json.load(open(sys.argv[4]))
plan = json.load(open(sys.argv[5], encoding='utf-8'))[sys.argv[6]]; clip = sys.argv[6]; outdir = sys.argv[7]; step = float(sys.argv[8])
s = 1024/3840; p = {}
for k, v in cal.items():
    v = float(v)
    if k in ('f0','f1'): v *= s
    elif k in ('cx0','cy0','cx1','cy1'): v = (v+0.5)*s-0.5
    p[k] = torch.tensor(v, device='cuda')
n = len(motion['R']); fps = motion['fps']
Y, P, Fv = build_path(plan, motion, n)
LP = LR = None
if len(sys.argv) > 9:
    LP, LR = level_series(json.load(open(sys.argv[9])), n, fps)
tt = lambda a: torch.tensor(math.radians(float(a)), device='cuda')
W, H = 480, 270
try:
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)  # Windows; edit this path on other systems
except OSError:
    font = ImageFont.load_default()  # tiny bitmap font, but works everywhere
proc = subprocess.Popen(['ffmpeg','-v','error','-i',lrf,'-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE)
fs = 2048*1024*3; tiles = []; nexts = 0.0; i = 0
while True:
    buf = proc.stdout.read(fs)
    if len(buf) < fs: break
    t = i/fps
    if t >= nexts:
        a = np.frombuffer(buf, np.uint8).reshape(1024, 2048, 3)
        x = torch.from_numpy(a.copy()).cuda().permute(2,0,1).float()[None]/255
        d = persp_dirs(W, H, float(Y[i]), float(P[i]), 0.0, float(Fv[i]))
        if LP is not None: d = d @ rotmat(tt(0), tt(LP[i]), tt(LR[i])).T
        with torch.no_grad(): img = sample(d, x[..., :1024], x[..., 1024:], p, size=1024, blend_deg=6)
        im = Image.fromarray((img.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
        dr = ImageDraw.Draw(im); dr.rectangle([0,0,330,18], fill=(0,0,0))
        dr.text((3,1), f'{clip} {t:.1f}s y{Y[i]:.0f} p{P[i]:.0f} f{Fv[i]:.0f}' + (f' L{LP[i]:.0f}/{LR[i]:.0f}' if LP is not None else ''), fill=(255,255,0), font=font)
        tiles.append(im); nexts += step
    i += 1
proc.stdout.close(); proc.wait()
cols = 4; rows = 6; os.makedirs(outdir, exist_ok=True)
for si in range(0, len(tiles), cols*rows):
    ch = tiles[si:si+cols*rows]
    sh = Image.new('RGB', (W*cols, H*math.ceil(len(ch)/cols)))
    for k, im in enumerate(ch): sh.paste(im, ((k%cols)*W, (k//cols)*H))
    sh.save(f'{outdir}/{clip}_{si//(cols*rows):02d}.jpg', quality=82)
print(clip, len(tiles))
