# Robust shake measure (median optical flow of the whole frame), for windows without intended pans.
# usage: jitter2.py <label> <start> <secs> <video> [<video> ...]
import sys, subprocess, numpy as np, cv2
label, ss, secs = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
W, H = 640, 360; SC = 3840 / W
def gauss(x, s):
    r = int(3 * s) + 1; k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); k /= k.sum()
    return np.convolve(np.pad(x, r, mode='edge'), k, mode='valid')
for vid in sys.argv[4:]:
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-ss', str(ss), '-t', str(secs), '-i', vid, '-vf', f'scale={W}:{H},format=gray',
                          '-f', 'rawvideo', '-'], stdout=subprocess.PIPE)
    prev = None; rows = []
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H: break
        g = np.frombuffer(b, np.uint8).reshape(H, W)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, 600, 0.005, 8)
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1; d = (nxt - pts).reshape(-1, 2)[ok] * SC
            rows.append(np.median(d, 0))
        prev = g
    a = np.array(rows)
    out = []
    for c in (0, 1):
        tr = np.cumsum(a[:, c]); hf = tr - gauss(tr, 12)
        sp = np.abs(a[:, c]); uneven = np.std(sp - gauss(sp, 5))
        out.append(f'{np.std(hf):6.1f} (uneven {uneven:5.1f})')
    name = vid.replace('\\', '/').split('/')[-1]
    print(f'{label:12s} {name:24s} shake px  left-right {out[0]}   up-down {out[1]}')
