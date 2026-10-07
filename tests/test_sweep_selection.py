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


def sweeps_two_sided(past, future):
    out = []
    for x in past:
        m = np.eye(4); m[0, 3] = -x
        out.append({'transform_matrix': m, 'time_lag': x / 10.0})
    for x in future:
        m = np.eye(4); m[0, 3] = x
        out.append({'transform_matrix': m, 'time_lag': -x / 10.0})
    return out


def test_future_off_never_picks_future():
    sw = sweeps_two_sided(np.arange(1, 50) * 0.5, np.arange(1, 50) * 0.5)
    idx = select_sweep_indices(sw, 15, {'MODE': 'displacement', 'SPAN_M': 20.0})
    assert all(sw[k]['time_lag'] > 0 for k in idx) and len(idx) == 14


def test_future_splits_by_reach():
    sw = sweeps_two_sided(np.arange(1, 21) * 0.5, np.arange(1, 81) * 0.5)  # past reaches 10 m, future 40 m
    idx = select_sweep_indices(sw, 15, {'MODE': 'displacement', 'SPAN_M': 30.0, 'FUTURE': True})
    fut = [k for k in idx if sw[k]['time_lag'] < 0]
    assert len(idx) == 14 and len(set(idx)) == 14
    assert len(fut) == round(14 * 30 / 40)  # future reach 30 (capped by SPAN), past 10
    assert max(np.linalg.norm(sw[k]['transform_matrix'][:3, 3]) for k in fut) <= 30.26


def test_future_sweeps_chain_on_real_infos():
    """future_sweeps maps a later keyframe's sweeps into the anchor frame with the same chain the past sweeps use:
    the later keyframe stores the ANCHOR's own lidar file among its sweeps, and that entry must map to identity."""
    import pickle
    from types import SimpleNamespace
    from pathlib import Path
    import pytest
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    p = Path('/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval/nuscenes_infos_200sweeps_val.pkl')
    if not p.exists():
        pytest.skip('nuScenes infos not available')
    infos = pickle.load(open(p, 'rb'))[:80]
    frames, tok2frame, scene_order, scene = [], {}, {}, 0
    for k, inf in enumerate(infos):
        t = float(inf['timestamp'])
        if k and t - frames[-1]['t'] > 0.6:
            scene += 1
        tok2frame[inf['token']] = len(frames)
        frames.append({'t': t, 'S': np.asarray(inf['ref_from_car']) @ np.asarray(inf['car_from_global']), 'scene': scene})
        scene_order.setdefault(scene, []).append(len(frames) - 1)
    fake = SimpleNamespace(_sweep_compensator=SimpleNamespace(tok2frame=tok2frame, frames=frames, scene_order=scene_order),
                           infos=infos, _frame_to_info=None)
    a = next(k for k in range(len(infos) - 1) if frames[k + 1]['scene'] == frames[k]['scene'])
    fs = NuScenesDataset.future_sweeps(fake, infos[a], 1e9)
    assert fs and all(s['time_lag'] < 0 for s in fs)
    lags = [s['time_lag'] for s in fs]
    assert lags == sorted(lags, reverse=True)                     # oldest (closest to the anchor) first
    assert len({s['lidar_path'] for s in fs}) == len(fs)          # nothing twice
    assert infos[a]['lidar_path'] not in {s['lidar_path'] for s in fs}
    f = infos[a + 1]
    A = frames[a]['S'] @ np.linalg.inv(frames[a + 1]['S'])
    own = [s for s in f['sweeps'] if s['lidar_path'] == infos[a]['lidar_path']]
    assert own, 'the next keyframe should store the anchor lidar among its sweeps'
    assert np.abs(A @ own[0]['transform_matrix'] - np.eye(4)).max() < 1e-3


def test_distinct_real_sweeps_drops_padding():
    from pcdet.datasets.nuscenes.nuscenes_dataset import distinct_real_sweeps
    m = np.eye(4); m[0, 3] = -1.0
    pad = [{'lidar_path': 'a.bin', 'transform_matrix': None, 'time_lag': 0.0}] * 3
    real = [{'lidar_path': 's1.bin', 'transform_matrix': m, 'time_lag': 0.05}] * 2 + \
           [{'lidar_path': 's2.bin', 'transform_matrix': m, 'time_lag': 0.10}]
    out = distinct_real_sweeps(pad + real, 'a.bin')
    assert [s['lidar_path'] for s in out] == ['s1.bin', 's2.bin']
