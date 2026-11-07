"""Configs of experiments_md 20261003_05 §5 item 37 (count-matched structure terms). CPU only, configs only.

37a: the eval-only twin that scores the P->N count-matched control L (31539 / 31550) on Pandar64 spin val must be L's
resolved config with only DATA_CONFIG_TAR replaced, and that block must be the PandaSet oracle's (spin2spin). A twin that
kept nuScenes' DATA_CONFIG_TAR, or changed the model, would score a different detector or a different val.
"""
import sys
from pathlib import Path

from easydict import EasyDict

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
import _init_path  # noqa: E402,F401

L_PANDASET = 'cfgs/da-ieee-access/centerpoint-global-fovcut-recount-latcount-pandaset2nuscenes.yaml'
TWIN_37A = 'cfgs/da-ieee-access-analysis/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-l-on-spin.yaml'
SPIN = 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml'
P_TWIN = 'cfgs/da-ieee-access-analysis/factorswap-pandaset/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml'


def plain(x):
    if isinstance(x, dict):
        return {k: plain(v) for k, v in x.items() if k not in ('_BASE_CONFIG_', 'ROOT_DIR', 'LOCAL_RANK')}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    return x


def resolved(path):
    from pcdet.config import cfg_from_yaml_file
    return plain(cfg_from_yaml_file(path, EasyDict()))


def test_37a_twin_is_l_with_the_spin_oracle_eval_block(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, src, spin = resolved(TWIN_37A), resolved(L_PANDASET), resolved(SPIN)
    assert twin['DATA_CONFIG_TAR'] == spin['DATA_CONFIG_TAR']
    assert twin['DATA_CONFIG_TAR'] != src['DATA_CONFIG_TAR']
    for k in set(twin) | set(src):
        if k != 'DATA_CONFIG_TAR':
            assert twin.get(k) == src.get(k), k


def test_37a_twin_scores_the_same_model_as_the_p_twin(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, p = resolved(TWIN_37A), resolved(P_TWIN)
    for k in ('MODEL', 'CLASS_NAMES', 'DATA_CONFIG_TAR'):
        assert twin[k] == p[k], k


# ---- 37b: the Lyft-40 count-matched control and the multi-source reader it needs

def test_source_data_config_reads_single_and_multi_source_forms():
    from pcdet.datasets.point_calibration import source_data_config
    single = EasyDict({'DATA_CONFIG': {'A': 1}})
    multi = EasyDict({'DATA_CONFIGS': {'LYFT_40BEAM': {'A': 2}}})
    assert source_data_config(single).A == 1
    assert source_data_config(multi).A == 2


def test_source_data_config_refuses_two_sources():
    import pytest
    from pcdet.datasets.point_calibration import source_data_config
    with pytest.raises(AssertionError):
        source_data_config(EasyDict({'DATA_CONFIGS': {'A': {}, 'B': {}}}))
    with pytest.raises(AssertionError):
        source_data_config(EasyDict({}))


def test_build_reference_source_reads_a_data_configs_row(monkeypatch):
    """Before item 37b build_reference_source read ref.DATA_CONFIG only, so a Lyft-40 (DATA_CONFIGS) lattice row as the
    reference raised AttributeError; it must hand the single DATA_CONFIGS block to build_dataloader."""
    import pcdet.datasets as D
    from pcdet.datasets import point_calibration as PC
    monkeypatch.chdir(TOOLS)
    seen = {}

    def fake(dataset_cfg, **kw):
        seen['cfg'] = dataset_cfg
        return 'dataset', None, None
    monkeypatch.setattr(D, 'build_dataloader', fake)
    assert PC.build_reference_source('cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml') == 'dataset'
    assert 'RING_PATTERN' in seen['cfg'] and seen['cfg'].get('LIDAR_CONFIG') in (40, '40')


L_LYFT40 = 'cfgs/da-ieee-access/centerpoint-global-recount-latcount-lyft40-lyft2nuscenes.yaml'
C1_LYFT40 = 'cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml'
P_LYFT40 = 'cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml'
TWIN_37B = 'cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-evalonly-l-on-lyft40.yaml'
O_L40 = 'cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-oracle.yaml'


def test_37b_l_is_c1_matched_to_the_lattice_arm(monkeypatch):
    monkeypatch.chdir(TOOLS)
    l, c1 = resolved(L_LYFT40), resolved(C1_LYFT40)
    lb, cb = l['DATA_CONFIGS']['LYFT_40BEAM'], c1['DATA_CONFIGS']['LYFT_40BEAM']
    assert lb.pop('HIST_DIST_REFERENCE_CFG') == P_LYFT40 and lb.pop('HIST_DIST_IN_GRID') is True
    assert lb == cb
    for k in set(l) | set(c1):
        if k != 'DATA_CONFIGS':
            assert l.get(k) == c1.get(k), k


def test_37b_twin_is_l_with_the_lyft40_oracle_eval_block(monkeypatch):
    monkeypatch.chdir(TOOLS)
    twin, src, o = resolved(TWIN_37B), resolved(L_LYFT40), resolved(O_L40)
    assert twin['DATA_CONFIG_TAR'] == o['DATA_CONFIG_TAR'] != src['DATA_CONFIG_TAR']
    for k in set(twin) | set(src):
        if k != 'DATA_CONFIG_TAR':
            assert twin.get(k) == src.get(k), k


# ---- 37c: the KITTI pattern row P_K (TRAIN_FACTOR_SWAP RING_THIN MODE pattern) and its config

import numpy as np  # noqa: E402

P_K = 'cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml'
KITTI_ORACLE = 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml'
PATTERN = {'RING_THIN': {'MODE': 'pattern', 'SPACING_DEG': 1.33, 'AZ_RES_DEG': 0.332}}


def kitti_like_scan(n_rings=64, step_deg=0.17, r=20.0):
    """A raw HDL-64E-like scan in stored order: one ring's full turn after another, azimuth rising from ~0 deg, ring
    elevations from +2 down to -24.8 deg (the order kitti_rings.rings_from_order reads)."""
    el = np.linspace(2.0, -24.8, n_rings)
    az = np.arange(0.05, 360.0, step_deg)
    pts = []
    for e in el:
        a = np.radians(az)
        pts.append(np.stack([r * np.cos(a), r * np.sin(a), np.full_like(a, r * np.tan(np.radians(e))),
                             np.full_like(a, 0.5)], 1))
    return np.concatenate(pts).astype(np.float32), el


def test_37c_pattern_arm_is_the_frame_seeded_lattice():
    from pcdet.datasets.kitti import kitti_factor_swap, kitti_rings
    pts, el = kitti_like_scan()
    a = kitti_factor_swap.apply_points(pts, EasyDict(PATTERN), '000042')
    b = kitti_factor_swap.apply_points(pts, EasyDict(PATTERN), '000042')
    assert np.array_equal(a, b)                                    # a fixed dataset: the same cut every epoch
    m = kitti_rings.eval_ring_thin_mask(pts, PATTERN['RING_THIN'], kitti_factor_swap.frame_rng('000042', 'RING_THIN'))
    assert np.array_equal(a, pts[m])
    ring, lev = kitti_rings.rings_from_order(pts[:, :3])
    kept = np.unique(ring[m])
    assert 18 <= len(kept) <= 22                                   # 26.8 deg of KITTI FOV / 1.33 deg
    assert abs(np.median(np.diff(np.sort(lev[kept]))) - 1.33) < 0.3
    az = np.degrees(np.arctan2(pts[m, 1], pts[m, 0]))
    per_ring = [np.sum(ring[m] == k) for k in kept]
    assert max(per_ring) <= int(np.ceil(360.0 / 0.332)) + 1       # one point per 0.332 deg bin, not every 0.17 deg return
    assert min(per_ring) >= 0.9 * 360.0 / 0.332


def test_37c_p_k_config_is_the_oracle_plus_the_pattern_arm(monkeypatch):
    monkeypatch.chdir(TOOLS)
    p, o = resolved(P_K), resolved(KITTI_ORACLE)
    swap = p['DATA_CONFIG'].pop('TRAIN_FACTOR_SWAP')
    assert swap == plain(PATTERN)
    assert p == o


# ---- 37d: pitch rows on Lyft, whose LIDAR_TOP x points backward

def test_37d_lyft_accepts_the_pitch_arm():
    from pcdet.datasets import loader_factor_swap as lfs
    assert lfs.arm_of(EasyDict({'PITCH_DEG': [-3.0, -2.0], 'PITCH_PROB': [0.5, 0.5], 'PIVOT': [0, 0, 2.0]}), 'lyft') \
        == 'PITCH_DEG'


def test_37d_lyft_pitch_sign_raises_the_ground_ahead_to_nuscenes_slope(monkeypatch):
    """A flat ground falling 1.03 deg ahead (ahead = -x on Lyft) must, after the row's draw, rise ahead by nuScenes'
    slope (2.29 or 1.35). With the sign of the angle wrong it would fall by ~4.3 deg instead."""
    from pcdet.datasets.waymo import waymo_factor_swap as wfs
    monkeypatch.chdir(TOOLS)
    for target, lyft_fwd in (('lyft64', None), ('lyft40', None)):
        cfg = resolved('cfgs/da-ieee-access-analysis/factorswap-%s/centerpoint-factorswap-%s-r5-pitch.yaml'
                       % (target, target))
        block = cfg['DATA_CONFIG'] if 'DATA_CONFIG' in cfg else cfg['DATA_CONFIGS']['LYFT_40BEAM']
        swap = EasyDict(block['TRAIN_FACTOR_SWAP'])
        import json
        st = json.load(open('/home/koyama/data/factorswap_lyft/stats_%s.json' % target))
        lyft_fwd = st['ground']['forward_slope_deg']['median']          # rise ahead, already sign-corrected
        x = np.linspace(-40.0, 40.0, 401)
        # ground in the loader frame: z rises along -x (ahead) by lyft_fwd, through the point under the sensor
        ground = np.stack([x, np.zeros_like(x), np.tan(np.radians(lyft_fwd)) * (-x)], 1)
        for k, want in enumerate((2.29, 1.35)):
            out = wfs.pitch_about(ground, float(swap.PITCH_DEG[k]), swap.PIVOT)
            rise_ahead = np.degrees(np.arctan(np.polyfit(-out[:, 0], out[:, 2], 1)[0]))
            assert abs(rise_ahead - want) < 0.02, (target, k, rise_ahead)
        assert swap.PIVOT == [0.0, 0.0, float(st['shift_coor'][2])]


def test_37c_l_k_config_is_the_oracle_plus_the_measured_rates(monkeypatch):
    import json
    monkeypatch.chdir(TOOLS)
    lk = resolved('cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-l-latcount.yaml')
    o = resolved(KITTI_ORACLE)
    swap = lk['DATA_CONFIG'].pop('TRAIN_FACTOR_SWAP')
    assert lk == o
    r = json.load(open('/home/koyama/data/factorswap_s2/kitti_pattern_rates.json'))
    assert swap['RANGE_KEEP']['RATE'] == plain(r['RATE']) and swap['RANGE_KEEP']['EDGES'] == plain(r['EDGES'])
    assert len(r['EDGES']) == len(r['RATE']) + 1 and all(0 < x <= 1 for x in r['RATE'])
