# estimate camera heading R(t) (deg) from LRF via phase correlation of equirect upper band
# usage: motion.py <scriptdir> <calib.json> <lrf> <out.json>
import torch, json, sys, subprocess, numpy as np, math
sys.path.insert(0, sys.argv[1])
from geom360 import *
cal = json.load(open(sys.argv[2])); lrf = sys.argv[3]; out = sys.argv[4]
s = 1024 / 3840
p = {}
for k, v in cal.items():
    v = float(v)
    if k in ('f0', 'f1'): v = v * s
    elif k in ('cx0', 'cy0', 'cx1', 'cy1'): v = (v + 0.5) * s - 0.5
    p[k] = torch.tensor(v, device='cuda')
W, H = 720, 360
d = equirect_dirs(W, H)
r0, r1 = int((90 - 55) / 180 * H), int((90 - 12) / 180 * H)
band = d[r0:r1]
fps = eval(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','csv=p=0',lrf]).decode().strip())
proc = subprocess.Popen(['ffmpeg','-v','error','-i',lrf,'-f','rawvideo','-pix_fmt','rgb24','-'], stdout=subprocess.PIPE)
fs = 2048 * 1024 * 3
win = torch.hann_window(r1 - r0, device='cuda')[:, None]
prev = None; shifts = [0.0]; n = 0
def prep(x):
    with torch.no_grad():
        img = sample(band, x[..., :1024], x[..., 1024:], p, size=1024, blend_deg=6)
    g = img.mean(0)
    # gradient magnitude-ish highpass
    g = g - torch.nn.functional.avg_pool2d(g[None,None], 9, 1, 4, count_include_pad=False)[0,0]
    return g * win
while True:
    buf = proc.stdout.read(fs)
    if len(buf) < fs: break
    a = np.frombuffer(buf, np.uint8).reshape(1024, 2048, 3)
    x = torch.from_numpy(a.copy()).cuda().permute(2,0,1).float()[None] / 255
    g = prep(x)
    if prev is not None:
        F1 = torch.fft.fft2(prev); F2 = torch.fft.fft2(g)
        Rr = F2 * F1.conj(); Rr = Rr / (Rr.abs() + 1e-6)
        c = torch.fft.ifft2(Rr).real
        cx = c.sum(0)  # collapse y
        i = int(torch.argmax(cx))
        y0, y1, y2 = float(cx[(i-1) % W]), float(cx[i]), float(cx[(i+1) % W])
        den = (y0 - 2*y1 + y2); sub = 0.5*(y0 - y2)/den if abs(den) > 1e-9 else 0
        sh = i + sub
        if sh > W/2: sh -= W
        shifts.append(sh * 360 / W)  # feature shift in cam yaw (deg)
    prev = g; n += 1
proc.wait()
sh = np.array(shifts)
R = np.cumsum(-sh)  # camera heading
json.dump({'fps': fps, 'R': R.tolist()}, open(out, 'w'))
print(out, n, 'frames', 'fps', round(fps,3), 'R range', round(R.min(),1), round(R.max(),1))
for t in np.arange(0.5, n/fps, 2.0):
    print(f'  t={t:.1f} R={R[int(t*fps)]:.1f}', end='')
print()
