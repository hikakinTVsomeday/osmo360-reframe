# Stabilized full-res reframe render (2026-11-08). Same output format as render360.py, but:
#  - the camera's full orientation per source frame (stab360.py) is removed exactly, so shake in all 3 axes is gone;
#  - the view path is evaluated at the OUTPUT time (not at the repeated source frame), so 30->50 fps does not
#    turn smooth pans into a "move, move, stop" stutter.
# usage: render360s.py <scriptdir> <calib> <osv> <stab.json> <out.mp4> [t_end|-] [max_out_frames|-] [blend|-] [t_start]
# t_start (2026-11-09): start the output at this source time (decoders seek there, so later parts of long clips are quick).
import torch, json, sys, subprocess, numpy as np, math, time, threading, queue
sys.path.insert(0, sys.argv[1])
from geom360 import persp_dirs, sample
cal = json.load(open(sys.argv[2], encoding='utf-8')); osv = sys.argv[3]; st = json.load(open(sys.argv[4])); out = sys.argv[5]
t_end = float(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != '-' else None
max_out = int(sys.argv[7]) if len(sys.argv) > 7 and sys.argv[7] != '-' else None
t_start = float(sys.argv[9]) if len(sys.argv) > 9 and sys.argv[9] != '-' else 0.0
torch.backends.cudnn.benchmark = True
FPS_SRC = float(st.get('fps', 30000/1001)); FPS_OUT = 50  # 2026-11-09: source rate from stab json (50p clips exist)
C = np.array(st['C'], float).reshape(-1, 3, 3); Y = np.array(st['Y']); P = np.array(st['P']); Fv = np.array(st['F'])
n = len(C)
nb = min(int(subprocess.check_output(['ffprobe','-v','error','-select_streams',f'v:{i}','-show_entries','stream=nb_frames','-of','csv=p=0',osv]).decode().strip()) for i in (0,1))
n_src = min(n, nb)
dur = n_src / FPS_SRC if t_end is None else min(t_end, n_src / FPS_SRC)
n_out = int(math.floor(dur * FPS_OUT))
k0 = int(math.ceil(t_start * FPS_OUT - 1e-6))
if max_out: n_out = min(n_out, k0 + max_out)
j_first = int(math.floor(k0 / FPS_OUT * FPS_SRC))  # first source frame needed
SEEK = ['-ss', f'{(j_first - 0.5) / FPS_SRC:.5f}'] if j_first > 0 else []
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
    return subprocess.Popen(['ffmpeg','-v','error','-hwaccel','cuda','-hwaccel_output_format','cuda',*SEEK,'-i',osv,'-map',f'0:v:{stream}',
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
    *(['-ss', f'{k0/FPS_OUT:.4f}'] if k0 else []), '-i', osv, '-map','0:v','-map','1:a:0','-t',f'{(n_out-k0)/FPS_OUT:.4f}',
    '-vf','format=p010le','-c:v','hevc_nvenc','-preset','p5','-rc','vbr','-b:v','50M','-maxrate','70M','-bufsize','100M',
    '-profile:v','main10','-tag:v','hvc1','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-color_range','tv',
    '-video_track_timescale','50000','-c:a','aac','-b:a','192k','-ar','48000','-ac','2', out]
enc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, bufsize=W*H*3)
def to_tex(b):
    a = torch.frombuffer(bytearray(b), dtype=torch.uint8).cuda()
    yt = a[:S*S].view(1, 1, S, S).float()
    uv = a[S*S:].view(S//2, S//2, 2).permute(2, 0, 1)[None].float()
    return yt, uv
BLEND = len(sys.argv) > 8 and sys.argv[8] == 'blend'
# blend: show source frames j0 and j0+1 mixed by the output time, each with its OWN orientation removed first,
# so what is mixed differs only by walking movement (a few px) -> even motion instead of "move, move, stop".
cur = j_first - 1; texs = {}; eof = False; t0 = time.time(); bblend = 8.0*math.pi/180
def render_one(tex, Cm, yaw, pit, fov):
    Ct = torch.tensor(Cm.T, device='cuda', dtype=torch.float32)
    (y0, uv0), (y1, uv1) = tex
    dY = persp_dirs(W, H, yaw, pit, 0.0, fov) @ Ct
    c0, c1, th0, th1 = sample(dY, y0, y1, pY, size=S, mode='bicubic', separate=True)
    c1 = 16 + (c1 - 16) * gain
    w = ((th1 - th0) / bblend + 0.5).clamp(0, 1)
    Yo = c0 * w + c1 * (1 - w)
    dC = persp_dirs(W//2, H//2, yaw, pit, 0.0, fov) @ Ct
    e0, e1, ph0, ph1 = sample(dC, uv0, uv1, pC, size=S//2, mode='bilinear', separate=True)
    e1 = 128 + (e1 - 128) * gain
    wc = ((ph1 - ph0) / bblend + 0.5).clamp(0, 1)
    return Yo, e0 * wc + e1 * (1 - wc)
for k in range(k0, n_out):
    t = k / FPS_OUT; js = t * FPS_SRC
    i0 = min(int(math.floor(js)), n-1); i1 = min(i0+1, n-1); f = js - math.floor(js)
    if BLEND:
        if f < 0.03: need = [i0]
        elif f > 0.97: need = [i1]
        else: need = [i0, i1]
    else:
        need = [min(int(round(js)), n-1)]
    while cur < max(need):
        b0 = q0.get(); b1 = q1.get()
        if b0 is None or b1 is None: eof = True; break
        cur += 1
        if cur >= min(need): texs[cur] = (to_tex(b0), to_tex(b1))
    if eof or any(j not in texs for j in need): break
    for old in [x for x in texs if x < min(need)]: del texs[old]
    # view path at the output time js; camera orientation of each source frame actually shown
    yaw = float(Y[i0]*(1-f)+Y[i1]*f); pit = float(P[i0]*(1-f)+P[i1]*f); fov = float(Fv[i0]*(1-f)+Fv[i1]*f)
    with torch.no_grad():
        if len(need) == 1:
            Yo, Co = render_one(texs[need[0]], C[need[0]], yaw, pit, fov)
        else:
            Ya, Ca = render_one(texs[i0], C[i0], yaw, pit, fov)
            Yb, Cb = render_one(texs[i1], C[i1], yaw, pit, fov)
            Yo = Ya * (1 - f) + Yb * f; Co = Ca * (1 - f) + Cb * f
        Yo = Yo.clamp(0, 255); Co = Co.clamp(0, 255)
        frame = torch.cat([Yo.round().to(torch.uint8).flatten(), Co.round().to(torch.uint8).flatten()]).cpu().numpy()
    enc.stdin.write(frame.tobytes())
    if k % 250 == 0: print(f'{k}/{n_out} {time.time()-t0:.0f}s', flush=True)
enc.stdin.close(); enc.wait()
for d in (d0, d1): d.kill()
print('done', out, n_out, f'{time.time()-t0:.0f}s', flush=True)
