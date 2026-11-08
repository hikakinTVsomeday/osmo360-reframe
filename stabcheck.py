# 2026-11-09 Claude: quick check of a planned reframe BEFORE the slow 4K render.
# Takes frames from the low-res proxy (.LRF, 2048x1024 dual fisheye, 25 fps) and applies exactly what render360s.py
# will apply at that moment: the camera orientation C[j] from the stab json + the view path Y/P/F.
# usage: stabcheck.py <scriptdir> <calib> <lrf> <stab.json> <ranges "a-b,c-d"> <step_s> <out.jpg> [label]
# output: a contact sheet, 5 tiles per row (480x270), each labelled "<label> t=<sec>  yaw/pitch/fov".
import sys, json, math, subprocess
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, sys.argv[1])
from geom360 import persp_dirs, sample

cal = json.load(open(sys.argv[2], encoding='utf-8')); lrf = sys.argv[3]; st = json.load(open(sys.argv[4]))
ranges = [tuple(map(float, r.split('-'))) for r in sys.argv[5].split(',') if r]
step = float(sys.argv[6]); out = sys.argv[7]; label = sys.argv[8] if len(sys.argv) > 8 else ''
fps = float(st['fps']); C = np.array(st['C'], float).reshape(-1, 3, 3)
Y, P, F = np.array(st['Y']), np.array(st['P']), np.array(st['F'])
s = 1024 / 3840; p = {}
for k, v in cal.items():
    v = float(v)
    if k in ('f0', 'f1'): v *= s
    elif k in ('cx0', 'cy0', 'cx1', 'cy1'): v = (v + 0.5) * s - 0.5
    p[k] = torch.tensor(v, device='cuda')
W, H = 480, 270
try:
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)  # Windows; edit this path on other systems
except OSError:
    font = ImageFont.load_default()  # tiny bitmap font, but works everywhere
times = []
for a, b in ranges:
    t = a + 0.15
    while t < b - 0.1:
        times.append(t); t += step
    if not times or times[-1] < b - 0.6: times.append(b - 0.25)
tiles = []
for t in times:
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', lrf, '-frames:v', '1', '-f', 'rawvideo',
                          '-pix_fmt', 'rgb24', '-'], capture_output=True).stdout
    if len(raw) < 2048 * 1024 * 3: continue
    x = torch.from_numpy(np.frombuffer(raw, np.uint8).reshape(1024, 2048, 3).copy()).cuda().permute(2, 0, 1).float()[None] / 255
    j = min(int(round(t * fps)), len(C) - 1)
    Ct = torch.tensor(C[j].T, device='cuda', dtype=torch.float32)
    d = persp_dirs(W, H, float(Y[j]), float(P[j]), 0.0, float(F[j])) @ Ct
    with torch.no_grad():
        img = sample(d, x[..., :1024], x[..., 1024:], p, size=1024, blend_deg=6)
    im = Image.fromarray((img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    dr.rectangle([0, H - 20, W, H], fill=(0, 0, 0))
    dr.text((4, H - 18), f'{label} t={t:.1f}  y{Y[j]:.0f} p{P[j]:.0f} f{F[j]:.0f}', fill=(255, 255, 0), font=font)
    tiles.append(im)
cols = 5; rows = max(1, math.ceil(len(tiles) / cols))
sheet = Image.new('RGB', (W * cols, H * rows), 'black')
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % cols) * W, (i // cols) * H))
sheet.save(out, quality=82)
print(out, len(tiles), 'tiles')
