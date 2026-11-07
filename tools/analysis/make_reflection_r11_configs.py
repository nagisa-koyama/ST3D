"""Write the six reflection-row configs of experiments_md 20261003_05 §5 item 35g (R11 threshold / R11r random twin on
S2 KITTI, N->W Waymo TOP, PandaSet Pandar64) from the swap values of tools/analysis/reflection_r11_stats.py
(/home/koyama/data/reflection_r11/r11_stats.json). Every value goes into the config explicitly, so a run is reproducible
from its config alone. Floats are written with repr (exact round trip): T must equal the loader's intensity values.

    python3 analysis/make_reflection_r11_configs.py [--stats /home/koyama/data/reflection_r11/r11_stats.json]
"""
import argparse
import json
import os
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
A = os.path.join(TOOLS, 'cfgs', 'da-ieee-access-analysis')
TOP_CALIB = '/home/koyama/data/waymo_top_calib/top_calib.pkl'
ROWS = {   # M1 sensor: (directory, oracle config, oracle line, sensor line)
    'kitti': ('factorswap-s2', 'cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml',
              'FACTOR-SWAP S2: the KITTI oracle recipe 26843 (80.33 / 70.59, Car BEV / 3D moderate)',
              'KITTI HDL-64E, the whole raw scan about the velodyne origin (kitti_factor_swap point arm)'),
    'waymo': ('factorswap-n2w', 'cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml',
              'FACTOR-SWAP N->W: the Waymo oracle recipe O2 31624 (65.63 Car BEV on Waymo val, every 5th frame)',
              'Waymo TOP returns of the stored first-return cloud about the segment\'s TOP position (TOP_CALIB); the side '
              'lidars untouched; differential recount of the boxes it empties (waymo_factor_swap)'),
    'pandar64': ('factorswap-pandaset', 'cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml',
                 'FACTOR-SWAP Pandar64: the spin oracle O_P 26389 (81.67 / 69.26 rule A, spin val)',
                 'Pandar64 device 0 in PandaSet ego axes about the fitted sensor origin, before the axis swap '
                 '(pandaset_dataset._get_lidar_points)'),
}


def fmt(v):
    return 'null' if v is None else repr(float(v))


def write(sensor, mode, cfgv, stats_path, agg):
    d, base, oracle, sensor_line = ROWS[sensor]
    row = 'r11-reflection' if mode == 'threshold' else 'r11r-reflection-random'
    name = 'centerpoint-%s-%s.yaml' % (d, row)
    arm = ('R11, threshold model: in each (range bin, height class) with E > 0 the returns with intensity < T are dropped '
           'and those at exactly T with probability Q, so the darkest share E goes, as a sensor that loses its weakest '
           'returns first.' if mode == 'threshold' else
           'R11r, count-matched random twin of R11: each return of a (range bin, height class) dropped with probability E, '
           'whatever its intensity; R11r - R11 separates WHICH returns are lost from HOW MANY.')
    text = [
        'ANALYSIS training row, %s, with ONE factor of its TRAINING data set to the nuScenes value (experiments_md '
        '20261003_05 §5 item 35g; design fixed 2026-11-07 13:11 JST, R11 approved by the user that day), training only '
        '(TRAIN_FACTOR_SWAP.REFLECTION_DROP, pcdet/datasets/reflection_drop.py), scored on the target\'s UNPERTURBED val '
        'in-run (DATA_CONFIG_TAR is the oracle\'s). Contribution = AP(oracle) - AP(this row), Car BEV R40 IoU 0.7, one '
        'seed; read if >= 3 AP. Target TRAIN labels train this row: labelled analysis, never a method.' % oracle,
        'This arm: reflection characteristics (a sensor-device sub-factor, item 35a). %s' % arm,
        'E = M1\'s excess drop e_T(b, c) = (h_nuS - h_T) / (1 - h_T) (job 32100, /home/koyama/data/reflection_m1/m1.json), '
        'applied where e > 0 and both sensors have >= 2,000 eligible slots in the cell (else 0). Bins 0-10, 10-20, 20-30, '
        '30-40, 40-50, 50-60, 60-75 m (3D range about the sensor); classes ground |dz| <= 0.3 m, above-ground 0.3-3.0 m '
        'over the frame\'s label-free ground plane; ego (< 3 m), beyond 75 m and outside both classes untouched.',
        'Returns: %s.' % sensor_line,
        'T / Q from %s (tools/analysis/reflection_r11_stats.py, Slurm CPU job, M1\'s drawn TRAIN frames). Expected '
        'dropped share of all returns %.4f; untouched share %.3f (on those frames).' % (
            stats_path, agg['expected_drop_share'], agg['untouched_share']),
    ]
    lines = []
    for para in text:
        lines += textwrap.wrap(para, width=118, initial_indent='# ', subsequent_indent='# ',
                               break_long_words=False, break_on_hyphens=False)
    lines += [
        '_BASE_CONFIG_: %s' % base,
        'DATA_CONFIG:',
        '    TRAIN_FACTOR_SWAP:',
        '        REFLECTION_DROP:',
        '            MODE: %s' % mode,
    ]
    keys = ['E'] + (['T', 'Q'] if mode == 'threshold' else [])
    for k in keys:
        lines.append('            %s:' % k)
        for c in ('ground', 'above'):
            lines.append('                %s: [%s]' % (c, ', '.join(fmt(v) for v in cfgv[k][c])))
    if sensor == 'waymo':
        lines.append('            TOP_CALIB: %s   # per-segment TOP extrinsic: the TOP position' % TOP_CALIB)
    path = os.path.join(A, d, name)
    open(path, 'w').write('\n'.join(lines) + '\n')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stats', default='/home/koyama/data/reflection_r11/r11_stats.json')
    a = ap.parse_args()
    st = json.load(open(a.stats))
    for sensor in ROWS:
        t = st['targets'][sensor]
        for mode in ('threshold', 'random'):
            print(write(sensor, mode, t['configs'][mode], a.stats, t['aggregate']))


if __name__ == '__main__':
    main()
