"""Write the L0 weights: a trained sampler with its last layer zeroed, i.e. the per-bin rule exactly.

    python sampler_init_weights.py <trained.npz> <init.npz>
"""
import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import _init_path  # noqa: F401,E402
from pcdet.datasets.processor.point_sampler import LearnedPointSampler  # noqa: E402

s = LearnedPointSampler.load(sys.argv[1])
s.w3 = np.zeros_like(s.w3); s.b3 = np.zeros_like(s.b3)
s.save(sys.argv[2]); print('wrote', sys.argv[2])
