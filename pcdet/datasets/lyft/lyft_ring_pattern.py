"""RING_PATTERN and KEEP_FRAMES for the Lyft loader (experiments_md 20261024_01; design 20261003_05 §5 'User 10-24
~02:40' item 3). Both opt-in, training only, absent = unchanged.

The rings come from a per-frame cache written by tools/analysis/lyft_ring_cache.py (gate run 2's recovery,
lyft_rings.deskew_and_rings over every 40-beam TRAIN frame). A Lyft LIDAR_TOP scan is stored laser-major, so the
recovered rings are stored-order segments: ring id per raw row is a non-decreasing step function, cached as points per
ring (`counts`) beside the ring elevations (`lev`, de-skewed, about the sensor at firing time). An entry also holds the
raw row count `n` and the crc32 of the raw x, y, z columns (float32): a cloud that is not the cached one is refused,
so the ring ids can never slide against the points.

The thinning itself is lyft_rings.ring_pattern_mask, unchanged (the gate's): keep the rings nearest a vertical lattice
of SPACING_DEG (random phase per sample) and one point per kept ring per AZ_RES_DEG azimuth bin (random phase), over
the RAW sensor-frame cloud, i.e. before LEVEL_COOR, SHIFT_COOR and any augmentation.
"""
import hashlib
import pickle
import zlib

import numpy as np

from . import lyft_rings


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def xyz_crc32(points):
    """crc32 of the x, y, z columns as float32 in C order (the cache tool's definition)."""
    return zlib.crc32(np.ascontiguousarray(np.asarray(points)[:, :3], dtype=np.float32).tobytes()) & 0xFFFFFFFF


def ring_ids(counts):
    """Ring id per stored row from points per ring (rings are stored-order segments)."""
    counts = np.asarray(counts, dtype=np.int64)
    assert counts.ndim == 1 and np.all(counts > 0), 'ring counts must be positive'
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)


def read_keep_list(path, sha256=None):
    """Frame stems, one per line. With `sha256`, the file must hash to it (the committed list, nothing else)."""
    if sha256 is not None:
        got = file_sha256(path)
        assert got == sha256, 'KEEP_FRAMES list %s hashes to %s, the config expects %s' % (path, got, sha256)
    with open(path) as f:
        stems = [s.strip() for s in f if s.strip()]
    assert len(set(stems)) == len(stems), 'KEEP_FRAMES list %s repeats a frame' % path
    return stems


def load_ring_cache(path, sha256=None):
    if sha256 is not None:
        got = file_sha256(path)
        assert got == sha256, 'ring cache %s hashes to %s, the config expects %s' % (path, got, sha256)
    with open(path, 'rb') as f:
        cache = pickle.load(f)
    assert isinstance(cache, dict) and 'frames' in cache, 'not a lyft ring cache: %s' % path
    return cache['frames']


def check_entry(stem, entry, points, n_rings):
    """Refuse a cache entry that does not belong to this cloud or does not hold exactly `n_rings` rings."""
    assert entry is not None, 'frame %s has no entry in the ring cache' % stem
    assert int(entry['n']) == len(points), 'ring cache for %s holds %d rows, the cloud has %d' % (
        stem, int(entry['n']), len(points))
    assert int(np.sum(entry['counts'])) == len(points), 'ring cache counts for %s do not sum to the cloud' % stem
    assert xyz_crc32(points) == int(entry['crc32']), 'ring cache for %s was made from a different cloud' % stem
    assert len(entry['lev']) == len(entry['counts']) == n_rings, \
        'frame %s has %d recovered rings, RING_PATTERN.N_RINGS is %d (KEEP_FRAMES keeps exact counts only)' % (
            stem, len(entry['lev']), n_rings)


def ring_pattern_keep(stem, points, entry, cfg, rng=None):
    """Keep mask of RING_PATTERN over one RAW Lyft LIDAR_TOP cloud (stored order, sensor frame, before any shift).
    Phases: two draws from `rng` (np.random by default, as the PandaSet loader), vertical first, as the gate."""
    check_entry(stem, entry, points, int(cfg['N_RINGS']))
    rng = np.random if rng is None else rng
    draw = rng.random_sample if hasattr(rng, 'random_sample') else rng.random
    phase_v, phase_h = draw(), draw()
    keep, _ = lyft_rings.ring_pattern_mask(np.asarray(points)[:, :3], ring_ids(entry['counts']),
                                           np.asarray(entry['lev'], dtype=np.float64), float(cfg['SPACING_DEG']),
                                           float(cfg['AZ_RES_DEG']), phase_v, phase_h)
    return keep
