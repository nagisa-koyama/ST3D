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
