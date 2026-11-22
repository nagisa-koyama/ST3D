"""Item 53 D2 reader (tools/analysis/item53_d2_read.py): log parsing and the fixed reading rule. CPU only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools' / 'analysis'))
import item53_d2_read as r  # noqa: E402

LOG = """Car AP_R40@0.70, 0.70, 0.70:
bbox AP:1.0, 2.0, 3.0
bev  AP:10.0, 20.0, 30.0
3d   AP:1.5, 2.5, 3.5
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:1.0, 2.0, 3.0
bev  AP:40.0, 50.0, 60.0
3d   AP:4.5, 5.5, 6.5
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:1.0, 2.0, 3.0
bev  AP:11.0, 21.0, 31.0
3d   AP:1.0, 2.0, 3.0
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:1.0, 2.0, 3.0
bev  AP:41.0, 51.0, 61.0
3d   AP:4.0, 5.0, 6.0
"""


def test_car_ap_reads_the_last_block_moderate_column():
    assert r.car_ap(LOG) == (21.0, 2.0, 51.0)


def test_reading_rule():
    assert r.reading([0.5, 0.9]) == 'collapse gone'
    assert r.reading([0.49, 0.1]) == 'collapse remains'
    assert r.reading([0.6, 0.4]) == 'mixed'


def test_the_d2_script_runs_every_layer_group_for_both_seeds():
    """Job 35169 stopped after its first eval: the group list was named GROUPS, which bash keeps as the user's group ids
    and silently refuses to assign, so RE[<gid>] was unbound. RUN / READ replace the eval and read commands here."""
    import os
    import subprocess
    script = os.environ.get('ITEM53_D2_SCRIPT',
                            str(Path(__file__).resolve().parents[1] / 'tools' / 'scripts' / '_item53_d2_evals.sh'))
    env = dict(os.environ, RUN='echo EVAL', READ='echo READ')
    p = subprocess.run(['bash', script], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    evals = [l for l in p.stdout.splitlines() if l.startswith('EVAL ')]
    assert len(evals) == 14
    for s in (1, 2):
        mine = [l for l in evals if '--extra_tag i53_s%d' % s in l]
        assert sum('--layers' not in l for l in mine) == 1                       # full AdaBN, the procedure check
        for g in r.GROUPS:
            assert sum(('--layers_name %s ' % g) in l for l in mine) == 1
    assert '^(?!point_head\\.box_center_layers)' in p.stdout and '^(?!backbone_3d\\.SA_modules\\.4\\.)' in p.stdout
    assert p.stdout.splitlines()[-1].startswith('READ analysis/item53_d2_read.py')
