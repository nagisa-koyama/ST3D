"""Reader of the offline, frame-keyed S3b render cache (experiments_md episodes/20261116_01 §8.9, "b1/b2 preparation").

A RENDER op inside DATA.FACTOR_SWAP (train set) or DATA.EVAL_PERTURB (val set) replaces each frame's cloud by a cached
re-render. The cache is built offline by tools/analysis/pcseg_render_cache.py (pcseg.render, time-aware z-buffer, no
fill) and holds, per frame f of a sequence, for every output point:

  src_off  int8   g - f, the source frame it came from (depth 3: -1, 0, +1);
  src_idx  int32  its index in that frame's points of the source device, in pandaset_seg.read_frame's order;
  xyz      float32 [M, 3] the rendered point, on the target ray at the winner's range, in frame f's normative ego frame
           (read_frame's frame, no SHIFT_COOR);
  n_src    int32 [3] the source frames' point counts for offsets -1, 0, +1 (-1 = frame not used), checked at load;
  mixed    bool   diagnostic only (another class among the cell's candidates near the winner's range); never read here.

Labels and intensities are never cached. They follow the index: `load` reads the source frames with read_frame and takes
raw label and intensity at (src_off, src_idx), so labels move by the same index as the points (§4). The renderer makes
no point (no fill), so there is no -1 index; a source label that maps to IGNORE (-1) stays -1. The cache never saw a
label of the frame it renders onto.
"""
import hashlib
import json
import os

import numpy as np

OFFSETS = (-1, 0, 1)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class RenderCache:
    """spec: {CACHE: <dir of one kind, holding manifest.json and <seq>/<ff>.npz>}."""

    def __init__(self, spec, device):
        spec = dict(spec or {})
        unknown = sorted(set(spec) - {'CACHE'})
        if unknown or not spec.get('CACHE'):
            raise ValueError('RENDER: needs exactly {CACHE: <dir>}; unknown keys %s' % unknown)
        self.dir = str(spec['CACHE'])
        path = os.path.join(self.dir, 'manifest.json')
        if not os.path.isfile(path):
            raise FileNotFoundError('RENDER: no manifest.json in %s' % self.dir)
        with open(path) as f:
            self.manifest = json.load(f)
        self.manifest_sha256 = _sha256(path)
        self.device = int(device)
        if int(self.manifest['source_device']) != self.device:
            raise ValueError('RENDER: cache %s renders device %s, the dataset reads device %d'
                             % (self.dir, self.manifest['source_device'], self.device))

    def describe(self):
        m = self.manifest
        return ('RENDER %s: device %s onto %s, depth %s, tolerance %s deg (manifest sha256 %s...)'
                % (m['kind'], m['source_device'], m['target'], m['depth'], m['tolerance_deg'],
                   self.manifest_sha256[:12]))

    def path(self, seq, frame):
        return os.path.join(self.dir, seq, '%02d.npz' % frame)

    def record(self, seq, frame):
        p = self.path(seq, frame)
        if not os.path.isfile(p):
            raise FileNotFoundError('RENDER: %s/%02d is not in the cache (%s)' % (seq, frame, p))
        with np.load(p) as z:
            rec = {k: z[k] for k in ('src_off', 'src_idx', 'xyz', 'n_src')}
        m = len(rec['xyz'])
        if rec['xyz'].shape != (m, 3) or len(rec['src_off']) != m or len(rec['src_idx']) != m:
            raise ValueError('RENDER: %s has inconsistent array lengths' % p)
        if m == 0:
            raise ValueError('RENDER: %s renders no point' % p)
        if not np.isfinite(rec['xyz']).all():
            raise ValueError('RENDER: %s has non-finite coordinates' % p)
        return rec

    def load(self, read, seq, frame):
        """read(g) -> (points [N, 4] float32 of source frame g, raw labels [N]) as pandaset_seg.read_frame returns them.
        Returns (points [M, 4] float32: cached xyz + the source point's intensity, raw labels [M] int64)."""
        rec = self.record(seq, frame)
        off = rec['src_off'].astype(np.int64)
        idx = rec['src_idx'].astype(np.int64)
        inten = np.empty(len(idx), dtype=np.float32)
        raw = np.empty(len(idx), dtype=np.int64)
        done = np.zeros(len(idx), dtype=bool)
        for k, o in enumerate(OFFSETS):
            sel = off == o
            n_expect = int(rec['n_src'][k])
            if not sel.any():
                continue
            if n_expect < 0:
                raise ValueError('RENDER: %s/%02d uses offset %d but records no source frame for it' % (seq, frame, o))
            pts, lab = read(frame + o)
            if len(pts) != n_expect:
                raise ValueError('RENDER: %s/%02d offset %d: the source frame has %d points, the cache was built on %d'
                                 % (seq, frame, o, len(pts), n_expect))
            i = idx[sel]
            if i.min() < 0 or i.max() >= len(pts):
                raise ValueError('RENDER: %s/%02d offset %d: index out of range [0, %d)' % (seq, frame, o, len(pts)))
            inten[sel] = pts[i, 3]
            raw[sel] = lab[i]
            done |= sel
        if not done.all():
            raise ValueError('RENDER: %s/%02d: offsets %s are not in %s'
                             % (seq, frame, sorted(set(off[~done].tolist())), OFFSETS))
        points = np.concatenate([rec['xyz'].astype(np.float32), inten[:, None]], axis=1)
        return points, raw
