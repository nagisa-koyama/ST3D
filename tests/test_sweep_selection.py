"""select_sweep_indices: consecutive by default; displacement mode spreads the same count over the span."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import _init_path  # noqa
from pcdet.datasets.nuscenes.nuscenes_dataset import select_sweep_indices


def sweeps_with_displacements(d):
    out = []
    for x in d:
        m = np.eye(4); m[0, 3] = x
        out.append({'transform_matrix': m})
    return out


def test_default_is_consecutive():
    sw = sweeps_with_displacements(np.arange(1, 200) * 0.5)
    assert select_sweep_indices(sw, 15) == list(range(14))
    assert select_sweep_indices(sw, 15, {'MODE': 'consecutive'}) == list(range(14))
    assert select_sweep_indices(sw, 1) == []


def test_displacement_spreads_over_span():
    sw = sweeps_with_displacements(np.arange(1, 200) * 0.5)  # 0.5 m per sweep, 99.5 m available
    idx = select_sweep_indices(sw, 15, {'MODE': 'displacement', 'SPAN_M': 28.0})
    assert len(idx) == 14 and len(set(idx)) == 14
    disp = np.array([sw[k]['transform_matrix'][0, 3] for k in idx])
    assert np.allclose(disp, np.linspace(0, 28, 15)[1:], atol=0.26)


def test_displacement_falls_back_to_available_span():
    sw = sweeps_with_displacements(np.arange(1, 30) * 0.2)  # only 5.8 m available
    idx = select_sweep_indices(sw, 15, {'MODE': 'displacement', 'SPAN_M': 28.0})
    assert len(idx) == 14 and len(set(idx)) == 14 and max(idx) == 28


def test_stationary_ego_uses_distinct_sweeps():
    sw = sweeps_with_displacements(np.zeros(199))
    idx = select_sweep_indices(sw, 15, {'MODE': 'displacement', 'SPAN_M': 28.0})
    assert len(idx) == 14 and len(set(idx)) == 14
