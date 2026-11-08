# Read the camera's own fused attitude (gyro) from a DJI Osmo 360 .OSV, without DJI software.
# Written from the format notes in OpenOSV docs/FORMAT.md and proto/dvtm_osmo360.proto (Apache-2.0);
# no code from that project is used. Pure Python: MP4 box walk + hand-written protobuf reader.
# usage: osvmeta.py <osv> <out.json>
# output per frame: frame_ts (microseconds since power-up), att_ts (attitude clock), q = list of [w,x,y,z]
import sys, struct, json

def boxes(f, start, end):
    p = start
    while p + 8 <= end:
        f.seek(p); size, typ = struct.unpack('>I4s', f.read(8)); hdr = 8
        if size == 1: size = struct.unpack('>Q', f.read(8))[0]; hdr = 16
        elif size == 0: size = end - p
        yield typ.decode('latin1'), p + hdr, p + size
        p += size

def find(f, start, end, path):
    for typ, s, e in boxes(f, start, end):
        if typ == path[0]:
            if len(path) == 1: yield s, e
            else: yield from find(f, s, e, path[1:])

def read(f, s, e): f.seek(s); return f.read(e - s)

def track_samples(f, trak_s, trak_e):
    stbl = next(find(f, trak_s, trak_e, ['mdia', 'minf', 'stbl']))
    kids = {t: (s, e) for t, s, e in boxes(f, *stbl)}
    stsd = read(f, *kids['stsd']); fmt = stsd[12:16].decode('latin1')
    stsz = read(f, *kids['stsz']); ssize, cnt = struct.unpack('>II', stsz[4:12])
    sizes = [ssize] * cnt if ssize else list(struct.unpack(f'>{cnt}I', stsz[12:12 + 4 * cnt]))
    if 'stco' in kids:
        b = read(f, *kids['stco']); n = struct.unpack('>I', b[4:8])[0]; offs = list(struct.unpack(f'>{n}I', b[8:8 + 4 * n]))
    else:
        b = read(f, *kids['co64']); n = struct.unpack('>I', b[4:8])[0]; offs = list(struct.unpack(f'>{n}Q', b[8:8 + 8 * n]))
    b = read(f, *kids['stsc']); n = struct.unpack('>I', b[4:8])[0]
    runs = [struct.unpack('>III', b[8 + 12 * i:20 + 12 * i]) for i in range(n)]
    out = []; si = 0
    for ci, off in enumerate(offs, 1):
        spc = [r[1] for r in runs if r[0] <= ci][-1]
        for _ in range(spc):
            if si >= cnt: break
            out.append((off, sizes[si])); off += sizes[si]; si += 1
    return fmt, out

def varint(b, i):
    v = 0; sh = 0
    while True:
        c = b[i]; i += 1; v |= (c & 0x7f) << sh; sh += 7
        if c < 0x80: return v, i

def pb(b):
    """protobuf message -> {field: [values]}; wire type 2 stays bytes, 5 -> float, 1 -> double"""
    d = {}; i = 0; n = len(b)
    while i < n:
        key, i = varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0: v, i = varint(b, i)
        elif wt == 1: v = struct.unpack('<d', b[i:i + 8])[0]; i += 8
        elif wt == 2: ln, i = varint(b, i); v = b[i:i + ln]; i += ln
        elif wt == 5: v = struct.unpack('<f', b[i:i + 4])[0]; i += 4
        else: raise ValueError(f'wire type {wt}')
        d.setdefault(fn, []).append(v)
    return d

def sub(d, *path):
    for p in path:
        if d is None or p not in d: return None
        d = pb(d[p][0])
    return d

def quat(b):
    q = pb(b); return [q.get(k, [0.0])[0] for k in (1, 2, 3, 4)]  # w, x, y, z

if __name__ == '__main__':
    osv, outp = sys.argv[1], sys.argv[2]
    with open(osv, 'rb') as f:
        f.seek(0, 2); end = f.tell()
        moov = next(find(f, 0, end, ['moov']))
        traks = [(s, e) for t, s, e in boxes(f, *moov) if t == 'trak']
        tr = [track_samples(f, s, e) for s, e in traks]
        print('tracks:', [(fmt, len(smp)) for fmt, smp in tr])
        djmd = [smp for fmt, smp in tr if fmt == 'djmd']
        frames = []; clip = {}
        for k, (off, size) in enumerate(djmd[0]):
            f.seek(off); m = pb(f.read(size))
            if k == 0 and 1 in m:
                c = pb(m[1][0])
                for name, fld in (('readout', 4), ('read_dir', 5), ('imu_hz', 10), ('sensor_fps', 11), ('digital_focal', 8)):
                    if fld in c: clip[name] = pb(c[fld][0]).get(1, [None])[0]
            fm = pb(m[3][0])
            hdr = pb(fm[1][0])
            rec = {'seq': hdr.get(1, [None])[0], 'frame_ts': hdr.get(2, [None])[0]}
            da = sub(fm, 3, 2, 1)
            if da:
                rec['att_ts'] = da.get(1, [None])[0]; rec['vsync'] = da.get(2, [None])[0]
                rec['offset'] = da.get(4, [None])[0]
                rec['q'] = [quat(x) for x in da.get(3, [])]
            vs = sub(fm, 3, 3)
            if vs: rec['vsync_pos'] = vs.get(1, [None])[0]
            cam = sub(fm, 2)
            if cam and 9 in cam: rec['cam_q'] = quat(cam[9][0])
            frames.append(rec)
    json.dump({'clip': clip, 'frames': frames}, open(outp, 'w'))
    nq = [len(r.get('q', [])) for r in frames]
    print(outp, len(frames), 'frames; clip', clip, '; quats/frame min', min(nq), 'max', max(nq))
    print('first', json.dumps(frames[0])[:400]); print('second', json.dumps(frames[1])[:400])
