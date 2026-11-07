"""Side-effect-free helpers of reflection_r11_target_dump.py (importable by tests/test_reflection_drop.py)."""
import numpy as np


def gone_mask(final_boxes, emptied_raw):
    """Which of the collated (final) boxes are among the raw boxes the arm emptied, matched on x, y, z, dx, dy, dz.
    Compared in float32: raw boxes can be float64 (Waymo's annos) while the collated ones are float32, and an exact
    float64 comparison then never matches (the 2026-11-07 smoke of the N->W dump failed on that)."""
    final_boxes = np.asarray(final_boxes)
    if len(final_boxes) == 0 or len(emptied_raw) == 0:
        return np.zeros(len(final_boxes), dtype=bool)
    e = np.asarray(emptied_raw)[:, :6].astype(np.float32)
    f = final_boxes[:, :6].astype(np.float32)
    return np.array([bool((e == x).all(axis=1).any()) for x in f], dtype=bool)
