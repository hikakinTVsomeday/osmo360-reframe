# full-res reframe render. usage: render360.py <scriptdir> <calib> <osv> <motion> <plans> <clip> <up> <out.mp4> [t_end] [max_out_frames]
import torch, json, sys, subprocess, numpy as np, math, time, threading, queue
sys.path.insert(0, sys.argv[1])
from geom360 import *; from path360 import build_path, level_series
cal = json.load(open(sys.argv[2], encoding='utf-8')); osv = sys.argv[3]; motion = json.load(open(sys.argv[4]))
plan = json.load(open(sys.argv[5], encoding='utf-8'))[sys.argv[6]]; up = json.load(open(sys.argv[7])); out = sys.argv[8]
t_end = float(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] != '-' else None
max_out = int(sys.argv[10]) if len(sys.argv) > 10 else None
torch.backends.cudnn.benchmark = True
FPS_SRC = 30000/1001; FPS_OUT = 50
n = len(motion['R'])
Y, P, Fv = build_path(plan, motion, n); LP, LR = level_series(up, n, motion['fps'])
# 2026-11-08 fix: the LRF (used for motion) can have a few more frames than the OSV; never ask past the OSV's last frame
nb = min(int(subprocess.check_output(['ffprobe','-v','error','-select_streams',f'v:{i}','-show_entries','stream=nb_frames','-of','csv=p=0',osv]).decode().strip()) for i in (0,1))
n_src = min(n, nb)
dur = n_src / FPS_SRC if t_end is None else min(t_end, n_src / FPS_SRC)
n_out = int(math.floor(dur * FPS_OUT))
if max_out: n_out = min(n_out, max_out)
def pdict(scale):
    q = {}
    for k, v in cal.items():
        v = float(v)
        if k in ('f0','f1'): v *= scale
        elif k in ('cx0','cy0','cx1','cy1'): v = (v+0.5)*scale-0.5
        q[k] = torch.tensor(v, device='cuda')
    q['gain1'] = torch.tensor(1.0, device='cuda')
    return q
pY = pdict(1.0); pC = pdict(0.5); gain = float(cal['gain1'])
S = 3840; FS = S*S*3//2
def dec(stream):
    return subprocess.Popen(['ffmpeg','-v','error','-hwaccel','cuda','-hwaccel_output_format','cuda','-i',osv,'-map',f'0:v:{stream}',
        '-vf','scale_cuda=format=nv12,hwdownload,format=nv12','-f','rawvideo','-'], stdout=subprocess.PIPE, bufsize=FS*2)
def reader(proc, q):
    while True:
        b = proc.stdout.read(FS)
        if len(b) < FS: q.put(None); return
        q.put(b)
d0, d1 = dec(0), dec(1)
q0, q1 = queue.Queue(maxsize=3), queue.Queue(maxsize=3)
threading.Thread(target=reader, args=(d0, q0), daemon=True).start()
threading.Thread(target=reader, args=(d1, q1), daemon=True).start()
W, H = 3840, 2160
enc_cmd = ['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','yuv420p','-s',f'{W}x{H}','-r',str(FPS_OUT),'-i','-',
    '-i', osv, '-map','0:v','-map','1:a:0','-t',f'{n_out/FPS_OUT:.4f}',
    '-vf','format=p010le','-c:v','hevc_nvenc','-preset','p5','-rc','vbr','-b:v','50M','-maxrate','70M','-bufsize','100M',
    '-profile:v','main10','-tag:v','hvc1','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
    '-video_track_timescale','50000','-c:a','aac','-b:a','192k','-ar','48000','-ac','2', out]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=W*H*3)
def to_tex(b):
    a = torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda()
    yt = a[:S*S].view(1, 1, S, S).float()
    uv = a[S*S:].view(S//2, S//2, 2).permute(2, 0, 1)[None].float()
    return yt, uv
tt = lambda a: torch.tensor(math.radians(float(a)), device='cuda')
def lerp_ang(a, b, f): return a + f*((b - a + 180) % 360 - 180)
cur = -1; tex = None; eof = False; t0 = time.time(); bblend = 8.0*math.pi/180
for k in range(n_out):
    t = k / FPS_OUT; js = t * FPS_SRC; j = min(int(round(js)), n-1)
    while cur < j:
        b0 = q0.get(); b1 = q1.get()
        if b0 is None or b1 is None: eof = True; break
        cur += 1
        if cur == j: tex = (to_tex(b0), to_tex(b1))
    if tex is None or eof: break
    i0 = min(int(math.floor(js)), n-1); i1 = min(i0+1, n-1); f = js - math.floor(js)
    yaw = lerp_ang(Y[i0], Y[i1], f); pit = P[i0]*(1-f)+P[i1]*f; fov = Fv[i0]*(1-f)+Fv[i1]*f
    lp = LP[i0]*(1-f)+LP[i1]*f; lr = LR[i0]*(1-f)+LR[i1]*f
    M = rotmat(tt(0), tt(lp), tt(lr)).T
    with torch.no_grad():
        (y0, uv0), (y1, uv1) = tex
        dY = persp_dirs(W, H, yaw, pit, 0.0, fov) @ M
        c0, c1, th0, th1 = sample(dY, y0, y1, pY, size=S, mode='bicubic', separate=True)
        c1 = 16 + (c1 - 16) * gain
        w = ((th1 - th0) / bblend + 0.5).clamp(0, 1)
        Yo = (c0 * w + c1 * (1 - w)).clamp(0, 255)
        dC = persp_dirs(W//2, H//2, yaw, pit, 0.0, fov) @ M
        e0, e1, ph0, ph1 = sample(dC, uv0, uv1, pC, size=S//2, mode='bilinear', separate=True)
        e1 = 128 + (e1 - 128) * gain
        wc = ((ph1 - ph0) / bblend + 0.5).clamp(0, 1)
        Co = (e0 * wc + e1 * (1 - wc)).clamp(0, 255)
        frame = torch.cat([Yo.round().to(torch.uint8).flatten(), Co.round().to(torch.uint8).flatten()]).cpu().numpy()
    enc.stdin.write(frame.tobytes())
    if k % 250 == 0: print(f'{k}/{n_out} {time.time()-t0:.0f}s', flush=True)
enc.stdin.close(); enc.wait()
for d in (d0, d1): d.kill()
print('done', out, n_out, f'{time.time()-t0:.0f}s', flush=True)
