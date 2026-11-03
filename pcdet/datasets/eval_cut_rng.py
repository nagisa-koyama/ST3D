"""The per-frame generator of the evaluation-time scan-pattern cuts (ANALYSIS, evaluation only): Waymo EVAL_TOP_THIN
and EVAL_RING_THIN on nuScenes, KITTI and PandaSet. Each loader keys a frame by a string of its own (Waymo frame_id,
nuScenes lidar_path, KITTI sample_idx, PandaSet '<sequence>/<frame_idx:02d>') and seeds the generator with that key's
crc32, so every arm and every rescoring of one checkpoint cuts a frame the same way.

DRAW (opt-in key of the cut's config block; experiments_md 20261003_05 §5 item 30, the random-cut redraws): absent or
0 is the original seeding, `default_rng(crc32(key))`, bit for bit, so every config written before the key reproduces.
DRAW k >= 1 seeds `default_rng([crc32(key), k])`, a different stream per (frame, k). Only the count-matched random
arms (and the 'pattern' phases) consume the generator; a structured cut (rows, cols, azbin) draws nothing, so its twin
is the same at every k.
"""
import zlib

import numpy as np


def eval_cut_rng(key, draw=None):
    """Generator for one evaluation frame's cut: `key` the loader's frame key (str() is taken), `draw` the config's
    DRAW (None / 0 = the original seed)."""
    draw = 0 if draw is None else draw
    if isinstance(draw, bool) or not isinstance(draw, (int, np.integer)) or draw < 0:
        raise ValueError('DRAW must be a non-negative integer, got %r' % (draw,))
    crc = zlib.crc32(str(key).encode())
    return np.random.default_rng(crc if draw == 0 else [crc, int(draw)])
