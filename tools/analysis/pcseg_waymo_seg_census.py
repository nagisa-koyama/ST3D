"""Census of the 3D semantic segmentation labels in raw Waymo Perception tfrecords (open item O3, 20261116_01 §9).

No tensorflow, no waymo_open_dataset: a minimal protobuf wire-format walker that SEEKS past the camera JPEGs and range
images, so a 1 GB segment costs a few MB of reads unless a frame is decoded.

Per tfrecord it records:
  * frames, and frames whose lasers[*].ri_return{1,2}.segmentation_label_compressed is non-empty, per laser;
  * version fingerprints (field numbers from waymo-open-dataset master dataset.proto / label.proto):
      Label.num_top_lidar_points_in_box = 13          -> v1.3.2+
      CameraImage.camera_segmentation_label = 10      -> v1.4.0+
      CameraSegmentationLabel.num_cameras_covered = 5 -> v1.4.2+
      Frame.map_features = 10, Frame.map_pose_offset = 11 -> v1.4.2+
  * with --decode: for every labelled TOP frame, the semantic-class histogram over VALID returns (range > 0, as
    Pointcept's preprocess_waymo.py masks them), for return 1 and return 2 separately, plus sanity counts (labels on
    invalid pixels, classes outside the Segmentation.Type enum 0..22, points in no-label zones).

Usage (inside the container; numpy only):
  python3 analysis/pcseg_waymo_seg_census.py census --root <individual_files> --splits training validation \
      --shard 0 --nshards 2 --workers 4 --decode --out /home/koyama/data/pcseg/waymo_check/census_shard0.json
  python3 analysis/pcseg_waymo_seg_census.py aggregate --out_dir /home/koyama/data/pcseg/waymo_check
Reads only; never writes next to the data.
"""
import argparse, glob, json, os, struct, sys, time, zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

N_CLASSES = 23  # Segmentation.Type: TYPE_UNDEFINED = 0 ... TYPE_SIDEWALK = 22
CLASS_NAMES = ['UNDEFINED', 'CAR', 'TRUCK', 'BUS', 'OTHER_VEHICLE', 'MOTORCYCLIST', 'BICYCLIST', 'PEDESTRIAN', 'SIGN',
               'TRAFFIC_LIGHT', 'POLE', 'CONSTRUCTION_CONE', 'BICYCLE', 'MOTORCYCLE', 'BUILDING', 'VEGETATION',
               'TREE_TRUNK', 'CURB', 'ROAD', 'LANE_MARKER', 'OTHER_GROUND', 'WALKABLE', 'SIDEWALK']
LASER = {0: 'UNKNOWN', 1: 'TOP', 2: 'FRONT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT', 5: 'REAR'}


# ----------------------------------------------------------------------------- wire format, from a file (seeking)
def _fvarint(f):
    v = 0; s = 0
    while True:
        c = f.read(1)
        if not c:
            raise EOFError
        x = c[0]; v |= (x & 0x7f) << s
        if not x & 0x80:
            return v
        s += 7


def _ffields(f, end):
    """Yield (field, wiretype, value_or_length) for the message spanning [f.tell(), end). For wt 2 the caller either
    reads the payload or leaves it: the generator seeks to the payload's end before the next key either way."""
    while f.tell() < end:
        key = _fvarint(f); fn, wt = key >> 3, key & 7
        if wt == 0:
            yield fn, wt, _fvarint(f)
        elif wt == 1:
            f.seek(8, 1); yield fn, wt, None
        elif wt == 5:
            f.seek(4, 1); yield fn, wt, None
        elif wt == 2:
            ln = _fvarint(f); start = f.tell()
            yield fn, wt, (start, ln)
            f.seek(start + ln)
        else:
            raise ValueError('wire type %d at %d' % (wt, f.tell()))


# ----------------------------------------------------------------------------- wire format, in memory
def _varint(b, i):
    v = 0; s = 0
    while True:
        x = b[i]; i += 1; v |= (x & 0x7f) << s
        if not x & 0x80:
            return v, i
        s += 7


def _fields(b):
    i = 0; n = len(b)
    while i < n:
        key, i = _varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i); yield fn, wt, v
        elif wt == 1:
            yield fn, wt, b[i:i + 8]; i += 8
        elif wt == 5:
            yield fn, wt, b[i:i + 4]; i += 4
        elif wt == 2:
            ln, i = _varint(b, i); yield fn, wt, b[i:i + ln]; i += ln
        else:
            raise ValueError('wire type %d' % wt)


def packed_varints(buf):
    """Vectorised decode of a packed varint field (int32: negatives are 10-byte two's complement)."""
    b = np.frombuffer(buf, np.uint8)
    if b.size == 0:
        return np.zeros(0, np.int64)
    ends = np.flatnonzero(b < 0x80)
    starts = np.r_[0, ends[:-1] + 1]
    grp = np.repeat(np.arange(ends.size), ends - starts + 1)
    pos = np.arange(b.size) - starts[grp]
    vals = (b & 0x7f).astype(np.uint64) << (7 * pos).astype(np.uint64)
    return np.add.reduceat(vals, starts).view(np.int64)


def _dims(shape_blob):
    dims = []
    for fn, wt, v in _fields(shape_blob):
        if fn == 1 and wt == 0:
            dims.append(int(v))
        elif fn == 1 and wt == 2:
            dims += [int(x) for x in packed_varints(v)]
    return dims


def matrix(blob, kind):
    """MatrixFloat / MatrixInt32: data = 1 (packed), shape = 2 -> MatrixShape.dims = 1."""
    data = None; dims = []
    for fn, wt, v in _fields(blob):
        if fn == 1 and wt == 2:
            data = np.frombuffer(v, '<f4') if kind == 'f' else packed_varints(v).astype(np.int32)
        elif fn == 2 and wt == 2:
            dims = _dims(v)
    return data.reshape(dims)


# ----------------------------------------------------------------------------- one segment
def _read(f, span):
    start, ln = span; here = f.tell(); f.seek(start); b = f.read(ln); f.seek(here); return b


def _range_image(f, span, want):
    """RangeImage: range_image_compressed = 2, segmentation_label_compressed = 6. Returns (seg_len, seg_bytes, ri_bytes);
    the range image is read only when the frame carries a segmentation label and `want` is set."""
    seg_len = 0; seg = None; ri = None; ri_span = None
    for fn, wt, v in _ffields(f, span[0] + span[1]):
        if wt != 2:
            continue
        if fn == 6:
            seg_len = v[1]
            if want and v[1]:
                seg = f.read(v[1])
        elif fn == 2:
            ri_span = v
    if want and seg_len and ri_span:
        ri = _read(f, ri_span)
    return seg_len, seg, ri


def _decode_top(seg_c, ri_c, acc, key):
    seg = matrix(zlib.decompress(seg_c), 'i')            # [H, W, 2] = (instance_id, semantic_class)
    ri = matrix(zlib.decompress(ri_c), 'f')              # [H, W, 4] = (range, intensity, elongation, in-NLZ)
    a = acc[key]
    if seg.shape[:2] != ri.shape[:2]:
        a['shape_mismatch'] += 1
        return
    valid = ri[..., 0] > 0
    sem = seg[..., 1]; inst = seg[..., 0]
    sv = sem[valid]
    bad = (sv < 0) | (sv >= N_CLASSES)
    a['bad_class'] += int(bad.sum())
    a['hist'] += np.bincount(np.clip(sv, 0, N_CLASSES - 1)[~bad], minlength=N_CLASSES)
    a['valid'] += int(valid.sum())
    a['invalid_nonzero_label'] += int(((sem != 0) & ~valid).sum())
    a['instance_nonzero'] += int((inst[valid] > 0).sum())
    a['max_instance'] = max(a['max_instance'], int(inst.max()) if inst.size else 0)
    nlz = valid & (ri[..., 3] > 0)
    a['nlz_valid'] += int(nlz.sum()); a['nlz_label0'] += int((sem[nlz] == 0).sum())
    a['shape'] = list(seg.shape)


def census_file(args):
    path, decode = args
    t0 = time.time()
    rec = dict(file=os.path.basename(path), split=os.path.basename(os.path.dirname(path)), frames=0,
               seg_frames={}, seg_frames_r2={}, labelled=[], context_names=[], n_label_with_top_pts=0, n_labels=0,
               n_labels_first_frame=0,
               frames_with_cam_seg=0, cam_seg_with_num_cameras_covered=0, frames_with_map_features=0,
               frames_with_map_pose_offset=0, frame_fields=set(), error=None)
    acc = {k: dict(hist=np.zeros(N_CLASSES, np.int64), valid=0, bad_class=0, invalid_nonzero_label=0,
                   instance_nonzero=0, max_instance=0, nlz_valid=0, nlz_label0=0, shape=None, shape_mismatch=0,
                   frames=0) for k in ('r1', 'r2')}
    try:
        size = os.path.getsize(path)
        with open(path, 'rb', buffering=1 << 16) as f:
            while f.tell() < size:
                hdr = f.read(12)
                if len(hdr) < 12:
                    break
                ln = struct.unpack('<Q', hdr[:8])[0]; start = f.tell(); end = start + ln
                rec['frames'] += 1
                ts = None; ctx_name = None; top = {}
                any_cam_seg = False; has_mf = False
                for fn, wt, v in _ffields(f, end):
                    rec['frame_fields'].add(fn)
                    if fn == 1 and wt == 2:                                   # context
                        for f2, w2, v2 in _ffields(f, v[0] + v[1]):
                            if f2 == 1 and w2 == 2:
                                ctx_name = f.read(v2[1]).decode(); break
                    elif fn == 2 and wt == 0:                                 # timestamp_micros
                        ts = v
                    elif fn == 4 and wt == 2:                                 # images (CameraImage)
                        for f2, w2, v2 in _ffields(f, v[0] + v[1]):
                            if f2 == 10 and w2 == 2:                          # camera_segmentation_label
                                any_cam_seg = True
                                for f3, w3, v3 in _ffields(f, v2[0] + v2[1]):
                                    if f3 == 5:                               # num_cameras_covered
                                        rec['cam_seg_with_num_cameras_covered'] += 1
                    elif fn == 5 and wt == 2:                                 # lasers
                        name = None; spans = {}
                        for f2, w2, v2 in _ffields(f, v[0] + v[1]):
                            if f2 == 1 and w2 == 0:
                                name = v2
                            elif f2 in (2, 3) and w2 == 2:
                                spans[f2] = v2
                        lname = LASER.get(name, str(name))
                        for f2, key, store in ((2, 'r1', 'seg_frames'), (3, 'r2', 'seg_frames_r2')):
                            if f2 not in spans:
                                continue
                            here = f.tell(); f.seek(spans[f2][0])
                            seg_len, seg, ri = _range_image(f, spans[f2], decode and lname == 'TOP')
                            f.seek(here)
                            if seg_len:
                                rec[store][lname] = rec[store].get(lname, 0) + 1
                            if lname == 'TOP':
                                top[key] = (seg_len, seg, ri)
                    elif fn == 6 and wt == 2:                                 # laser_labels (Label)
                        rec['n_labels'] += 1
                        if rec['frames'] == 1:
                            rec['n_labels_first_frame'] += 1
                            for f2, w2, v2 in _ffields(f, v[0] + v[1]):
                                if f2 == 13:                                  # num_top_lidar_points_in_box
                                    rec['n_label_with_top_pts'] += 1
                    elif fn == 10:                                            # map_features
                        has_mf = True
                    elif fn == 11:                                            # map_pose_offset
                        rec['frames_with_map_pose_offset'] += 1
                f.seek(end + 4)                                               # trailing data CRC
                if any_cam_seg:
                    rec['frames_with_cam_seg'] += 1
                if has_mf:
                    rec['frames_with_map_features'] += 1
                if ctx_name and ctx_name not in rec['context_names']:
                    rec['context_names'].append(ctx_name)
                if 'r1' in top and top['r1'][0]:
                    rec['labelled'].append([rec['frames'] - 1, ts])
                    if decode:
                        for key in ('r1', 'r2'):
                            if key in top and top[key][1] is not None and top[key][2] is not None:
                                _decode_top(top[key][1], top[key][2], acc, key); acc[key]['frames'] += 1
    except Exception as e:  # keep the census going; the record says what failed
        rec['error'] = '%s: %s' % (type(e).__name__, e)
    rec['frame_fields'] = sorted(rec['frame_fields'])
    for k in acc:
        acc[k]['hist'] = acc[k]['hist'].tolist()
    rec['decode'] = acc if decode else None
    rec['seconds'] = round(time.time() - t0, 1)
    return rec


def cmd_census(a):
    files = []
    for s in a.splits:
        files += sorted(glob.glob(os.path.join(a.root, s, '*.tfrecord')))
    files = files[a.shard::a.nshards]
    if a.max_files:
        files = files[:a.max_files]
    print('%d files (shard %d/%d), decode=%s, %d workers' % (len(files), a.shard, a.nshards, a.decode, a.workers),
          flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = []; t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(census_file, [(p, a.decode) for p in files])):
            out.append(r)
            if (i + 1) % 10 == 0 or i + 1 == len(files) or r['error']:
                print('[%4d/%d] %s frames=%d seg_top=%d %s (%.0f s)' % (i + 1, len(files), r['file'][:40], r['frames'],
                      r['seg_frames'].get('TOP', 0), r['error'] or '', time.time() - t0), flush=True)
    tmp = a.out + '.tmp'
    json.dump(dict(args=vars(a), files=out), open(tmp, 'w'))
    os.replace(tmp, a.out)
    print('wrote %s in %.0f s' % (a.out, time.time() - t0))


def cmd_aggregate(a):
    recs = []
    for p in sorted(glob.glob(os.path.join(a.out_dir, 'census_shard*.json'))):
        recs += json.load(open(p))['files']
    summary = {}
    for split in sorted({r['split'] for r in recs}):
        rs = [r for r in recs if r['split'] == split]
        s = dict(files=len(rs), errors=[(r['file'], r['error']) for r in rs if r['error']],
                 frames=sum(r['frames'] for r in rs),
                 seg_frames_top_r1=sum(r['seg_frames'].get('TOP', 0) for r in rs),
                 seg_frames_top_r2=sum(r['seg_frames_r2'].get('TOP', 0) for r in rs),
                 seg_frames_other_lasers={k: sum(r['seg_frames'].get(k, 0) + r['seg_frames_r2'].get(k, 0) for r in rs)
                                          for k in ('FRONT', 'SIDE_LEFT', 'SIDE_RIGHT', 'REAR')},
                 segments_with_seg=sum(1 for r in rs if r['seg_frames'].get('TOP', 0)),
                 context_name_matches_file=sum(1 for r in rs if r['context_names'] and
                                               r['file'].startswith('segment-' + r['context_names'][0])),
                 multi_context_files=sum(1 for r in rs if len(r['context_names']) > 1),
                 frames_with_cam_seg=sum(r['frames_with_cam_seg'] for r in rs),
                 segments_with_cam_seg=sum(1 for r in rs if r['frames_with_cam_seg']),
                 cam_seg_with_num_cameras_covered=sum(r['cam_seg_with_num_cameras_covered'] for r in rs),
                 frames_with_map_features=sum(r['frames_with_map_features'] for r in rs),
                 frames_with_map_pose_offset=sum(r['frames_with_map_pose_offset'] for r in rs),
                 laser_labels_all_frames=sum(r['n_labels'] for r in rs),
                 laser_labels_first_frames=sum(r['n_labels_first_frame'] for r in rs),
                 first_frame_labels_with_num_top_lidar_points=sum(r['n_label_with_top_pts'] for r in rs),
                 frame_fields_seen=sorted({x for r in rs for x in r['frame_fields']}),
                 seg_frames_per_segment=dict(zip(*[x.tolist() for x in np.unique(
                     [r['seg_frames'].get('TOP', 0) for r in rs], return_counts=True)])))
        strides = np.concatenate([np.diff([i for i, _ in r['labelled']]) for r in rs if len(r['labelled']) > 1] or [[]])
        if strides.size:
            u, c = np.unique(strides.astype(int), return_counts=True)
            s['labelled_frame_index_stride'] = dict(zip(u.tolist(), c.tolist()))
        if rs[0].get('decode'):
            for key in ('r1', 'r2'):
                h = np.sum([r['decode'][key]['hist'] for r in rs], 0)
                tot = int(h.sum())
                d = {k: sum(r['decode'][key][k] for r in rs) for k in
                     ('frames', 'valid', 'bad_class', 'invalid_nonzero_label', 'instance_nonzero', 'nlz_valid',
                      'nlz_label0', 'shape_mismatch')}
                d['max_instance'] = max(r['decode'][key]['max_instance'] for r in rs)
                d['shapes'] = sorted({str(r['decode'][key]['shape']) for r in rs if r['decode'][key]['shape']})
                d['class_points'] = {CLASS_NAMES[i]: int(h[i]) for i in range(N_CLASSES)}
                d['class_share_pct'] = {CLASS_NAMES[i]: round(100 * h[i] / max(tot, 1), 4) for i in range(N_CLASSES)}
                d['classes_absent'] = [CLASS_NAMES[i] for i in range(N_CLASSES) if h[i] == 0]
                s['decode_' + key] = d
        summary[split] = s
        with open(os.path.join(a.out_dir, 'labelled_frames_%s.txt' % split), 'w') as fo:
            for r in sorted(rs, key=lambda r: r['file']):
                for _, ts in r['labelled']:
                    fo.write('%s,%d\n' % (r['context_names'][0] if r['context_names'] else r['file'], ts))
    json.dump(summary, open(os.path.join(a.out_dir, 'census_summary.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('census')
    c.add_argument('--root', default='/home/koyama/data/waymo_open_dataset_v_1_4_0/individual_files')
    c.add_argument('--splits', nargs='+', default=['training', 'validation'])
    c.add_argument('--shard', type=int, default=0); c.add_argument('--nshards', type=int, default=1)
    c.add_argument('--workers', type=int, default=4); c.add_argument('--max_files', type=int, default=0)
    c.add_argument('--decode', action='store_true'); c.add_argument('--out', required=True)
    g = sub.add_parser('aggregate'); g.add_argument('--out_dir', required=True)
    a = ap.parse_args()
    {'census': cmd_census, 'aggregate': cmd_aggregate}[a.cmd](a)
