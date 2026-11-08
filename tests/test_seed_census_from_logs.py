"""analysis/seed_census_from_logs.py (experiments_md 20261108_03 §2 A): pairs each evaluated epoch with the Car block printed
before it and reads the moderate column. CPU only, synthetic log text."""
import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
spec = importlib.util.spec_from_file_location('seed_census', TOOLS / 'analysis/seed_census_from_logs.py')
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def block(bev, d3):
    return ['Car AP_R40@0.70, 0.70, 0.70:', 'bbox AP:90.0, 85.0, 80.0', 'bev  AP:%s' % bev, '3d   AP:%s' % d3]


def test_epochs_take_the_block_before_their_end_line_and_the_moderate_column():
    lines = (['wandb: Run data is saved locally in /storage/wandb/run-20261101_000000-abc123',
              '[2026-11-01 00:00:00,000  train.py 172  INFO]  seed             667']
             + ['Car AP_R40@0.70, 0.50, 0.50:', 'bbox AP:1, 2, 3', 'bev  AP:9.0, 9.0, 9.0', '3d   AP:9.0, 9.0, 9.0']
             + block('80.0, 70.5, 60.0', '60.0, 50.25, 40.0')
             + ['*************** EPOCH 19 of kitti EVALUATION end.*****************']
             + block('81.0, 71.5, 61.0', '61.0, 51.25, 41.0')
             + ['*************** EPOCH 20 of kitti EVALUATION end.*****************'])
    ep, run, seed = census.parse(lines)
    assert run == 'abc123' and seed == 667
    assert ep == [{'epoch': 19, 'dataset': 'kitti', 'bev': 70.5, '3d': 50.25},
                  {'epoch': 20, 'dataset': 'kitti', 'bev': 71.5, '3d': 51.25}]


def test_identical_columns_and_an_end_line_without_a_block():
    lines = (['*************** EPOCH 5 of waymo EVALUATION end.*****************']
             + block('65.6, 65.6, 65.6', '51.1, 51.1, 51.1')
             + ['*************** EPOCH 7 of waymo EVALUATION end.*****************'])
    ep, run, seed = census.parse(lines)
    assert run is None and seed is None
    assert ep == [{'epoch': 7, 'dataset': 'waymo', 'bev': 65.6, '3d': 51.1}]
