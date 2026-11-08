import torch, math, json
import torch.nn.functional as F

def rotmat(yaw, pitch, roll):
    # angles in radians (tensors). yaw about y (right-handed: +yaw turns view to the right), pitch about x (+ looks up), roll about z
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    one, zero = torch.ones_like(cy), torch.zeros_like(cy)
    Ry = torch.stack([torch.stack([cy, zero, sy]), torch.stack([zero, one, zero]), torch.stack([-sy, zero, cy])])
    Rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, cp, sp]), torch.stack([zero, -sp, cp])])  # y down: +pitch looks up
    Rz = torch.stack([torch.stack([cr, -sr, zero]), torch.stack([sr, cr, zero]), torch.stack([zero, zero, one])])
    return Ry @ Rx @ Rz

class Lens:
    """params: dict with f0,cx0,cy0,f1,cx1,cy1 (pixels, full res 3840), k0,k1, ry1,rp1,rr1 (deg), gain1"""
    def __init__(self, p, device='cuda'):
        self.p = {k: torch.tensor(float(v), device=device) for k, v in p.items()}

def project(d, f, cx, cy, k, size):
    # d (...,3) lens-local unit dirs, z forward, x right, y down -> normalized grid coords
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    theta = torch.acos(z.clamp(-1, 1))
    rxy = torch.sqrt(x * x + y * y).clamp_min(1e-9)
    r = f * (theta + k * theta ** 3)
    u = cx + r * x / rxy
    v = cy + r * y / rxy
    gx = (2 * u + 1) / size - 1
    gy = (2 * v + 1) / size - 1
    return torch.stack([gx, gy], -1), theta

def world_to_lens1(d, p):
    d1 = torch.stack([-d[..., 0], d[..., 1], -d[..., 2]], -1)
    R = rotmat(p['ry1'] * math.pi / 180, p['rp1'] * math.pi / 180, p['rr1'] * math.pi / 180)
    return d1 @ R.T

def sample(d, tex0, tex1, p, size=3840, blend_deg=8.0, mode='bilinear', separate=False):
    """d: (H,W,3) world unit directions. world front (+z) = lens0 (facing the holder). tex: (1,3,S,S) float"""
    g0, th0 = project(d, p['f0'], p['cx0'], p['cy0'], p['k0'], size)
    d1 = world_to_lens1(d, p)
    g1, th1 = project(d1, p['f1'], p['cx1'], p['cy1'], p['k1'], size)
    c0 = F.grid_sample(tex0, g0[None], mode=mode, padding_mode='border', align_corners=False)[0]
    c1 = F.grid_sample(tex1, g1[None], mode=mode, padding_mode='border', align_corners=False)[0] * p['gain1']
    if separate:
        return c0, c1, th0, th1
    # weight: front lens if th0 < th1, smooth across seam
    b = blend_deg * math.pi / 180
    w0 = ((th1 - th0) / b + 0.5).clamp(0, 1)
    return c0 * w0 + c1 * (1 - w0)

def equirect_dirs(W, H, device='cuda'):
    lon = (torch.arange(W, device=device) + 0.5) / W * 2 * math.pi - math.pi   # -pi..pi, 0 = front(lens0)
    lat = math.pi / 2 - (torch.arange(H, device=device) + 0.5) / H * math.pi   # +pi/2 top
    lat, lon = torch.meshgrid(lat, lon, indexing='ij')
    x = torch.cos(lat) * torch.sin(lon)
    y = -torch.sin(lat)
    z = torch.cos(lat) * torch.cos(lon)
    return torch.stack([x, y, z], -1)

def persp_dirs(W, H, yaw, pitch, roll, hfov, device='cuda'):
    # yaw/pitch/roll/hfov in degrees; yaw 0 = lens0 axis, +yaw to the right; +pitch up
    fx = (W / 2) / math.tan(math.radians(hfov) / 2)
    xs = (torch.arange(W, device=device) + 0.5 - W / 2) / fx
    ys = (torch.arange(H, device=device) + 0.5 - H / 2) / fx
    ys, xs = torch.meshgrid(ys, xs, indexing='ij')
    d = torch.stack([xs, ys, torch.ones_like(xs)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    t = lambda a: torch.tensor(math.radians(a), device=device)
    R = rotmat(t(yaw), t(-pitch), t(roll))  # +pitch = look up
    return d @ R.T

DEFAULT = dict(f0=1920/math.radians(95), cx0=1920, cy0=1920, k0=0.0,
               f1=1920/math.radians(95), cx1=1920, cy1=1920, k1=0.0,
               ry1=0.0, rp1=0.0, rr1=0.0, gain1=1.0)
