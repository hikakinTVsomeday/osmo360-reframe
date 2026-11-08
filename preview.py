# usage: preview.py <scriptdir> <calib.json> <lrf> <clipname> <outdir> <interval>
import torch, json, sys, subprocess, numpy as np, os, math
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, sys.argv[1])
from geom360 import *
cal = json.load(open(sys.argv[2])); lrf = sys.argv[3]; name = sys.argv[4]; outdir = sys.argv[5]; iv = float(sys.argv[6])
s = 1024 / 3840
p = {k: torch.tensor(float(v) * (s if k in ('f0','cx0','cy0','f1','cx1','cy1') else 1), device='cuda') for k, v in cal.items()}
# need cx in LRF pixel coords: pixel-center convention u -> (u+0.5)*s - 0.5
for k in ('cx0','cy0','cx1','cy1'):
    p[k] = (torch.tensor(float(cal[k]), device='cuda') + 0.5) * s - 0.5
dur = float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',lrf]).decode())
W, H = 960, 480
d = equirect_dirs(W, H)
try:
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)  # Windows; edit this path on other systems
except OSError:
    font = ImageFont.load_default()  # tiny bitmap font, but works everywhere
tiles = []
t = 0.5
while t < dur:
    raw = subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',lrf,'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-'], capture_output=True).stdout
    a = np.frombuffer(raw, np.uint8).reshape(1024, 2048, 3)
    x = torch.from_numpy(a.copy()).cuda().permute(2,0,1).float()[None] / 255
    t0, t1 = x[..., :1024], x[..., 1024:]
    with torch.no_grad():
        img = sample(d, t0, t1, p, size=1024, blend_deg=6)
    im = Image.fromarray((img.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for yaw in range(-180, 181, 45):
        xx = int((yaw + 180) / 360 * (W - 1))
        dr.line([(xx, 0), (xx, H)], fill=(255,255,0), width=1)
        dr.text((xx + 3, 3), str(yaw), fill=(255,255,0), font=font)
    for lat in (-30, 0, 30):
        yy = int((90 - lat) / 180 * H)
        dr.line([(0, yy), (W, yy)], fill=(0,255,255), width=1)
    dr.rectangle([0, H - 26, 200, H], fill=(0,0,0))
    dr.text((5, H - 24), f'{name} t={t:.1f}s', fill=(255,255,255), font=font)
    tiles.append(im)
    t += iv
cols, rows = 2, 5
os.makedirs(outdir, exist_ok=True)
for si in range(0, len(tiles), cols*rows):
    chunk = tiles[si:si+cols*rows]
    sheet = Image.new('RGB', (W*cols, H*math.ceil(len(chunk)/cols)), 'black')
    for k, im in enumerate(chunk):
        sheet.paste(im, ((k % cols) * W, (k // cols) * H))
    sheet.save(f'{outdir}/{name}_{si//(cols*rows):02d}.jpg', quality=80)
print(name, len(tiles), 'tiles')
