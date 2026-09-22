"""Every da-ieee-access config spends the same optimisation budget.

The family fixes the budget in SAMPLE PRESENTATIONS - NUM_EPOCHS x source frames - anchored at
nuScenes 20 epochs = 562,600, so that no config's result is partly a training-length effect. The
epoch count therefore has to be read off the SOURCE's row, which is easy to get wrong when the
source and target are different datasets: a Lyft -> nuScenes config wants Lyft's 30, not
nuScenes' 20.
"""
import glob
import re
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG_DIR = ROOT / 'tools/cfgs/da-ieee-access'
ANCHOR = 562600          # 20 epochs x 28,130 nuScenes frames
TOL = 0.02               # whole-epoch rounding costs at most 1.7%

FRAMES = {'kitti': 3712, 'pandaset': 4880, 'lyft': 18900, 'nuscenes': 28130, 'waymo': 79041}


def _source_of(name):
    """The SOURCE dataset, which for an 'a2b' name is a, not b."""
    stem = Path(name).stem
    head = re.split(r'2(?:nuscenes|kitti)', stem)[0]
    for key in ('pandaset', 'nuscenes', 'kitti', 'waymo', 'lyft'):
        if key in head:
            return key
    return None


CONFIGS = [f for f in sorted(glob.glob(str(CFG_DIR / 'centerpoint-*.yaml')))
           if _source_of(f) is not None]


@pytest.mark.parametrize('path', CONFIGS, ids=[Path(p).stem for p in CONFIGS])
def test_budget_is_within_tolerance_of_the_anchor(path):
    cfg = yaml.safe_load(open(path))
    if 'OPTIMIZATION' not in cfg or 'NUM_EPOCHS' not in cfg['OPTIMIZATION']:
        pytest.skip('no NUM_EPOCHS of its own')
    src = _source_of(path)
    got = cfg['OPTIMIZATION']['NUM_EPOCHS'] * FRAMES[src]
    assert abs(got - ANCHOR) / ANCHOR <= TOL, (
        '%s: source %s (%d frames) x %d epochs = %d presentations, %.1f%% off the %d anchor. '
        'Read NUM_EPOCHS off the SOURCE row, not the target.'
        % (Path(path).name, src, FRAMES[src], cfg['OPTIMIZATION']['NUM_EPOCHS'], got,
           100 * (got - ANCHOR) / ANCHOR, ANCHOR))


@pytest.mark.parametrize('path', CONFIGS, ids=[Path(p).stem for p in CONFIGS])
def test_epoch_keyed_schedules_fit_inside_the_run(path):
    """PROG_AUG.UPDATE_AUG holds epoch INDICES, so they must scale with NUM_EPOCHS."""
    cfg = yaml.safe_load(open(path))
    st = cfg.get('SELF_TRAIN')
    if not st or not st.get('PROG_AUG', {}).get('ENABLED'):
        pytest.skip('no progressive augmentation')
    n = cfg['OPTIMIZATION']['NUM_EPOCHS']
    aug = st['PROG_AUG']['UPDATE_AUG']
    assert max(aug) < n, '%s: UPDATE_AUG %s never fires within %d epochs' % (Path(path).name, aug, n)
    assert max(aug) >= 0.6 * n, (
        '%s: UPDATE_AUG %s all land in the first %.0f%% of a %d-epoch run, so the ramp stops early'
        % (Path(path).name, aug, 100 * max(aug) / n, n))
