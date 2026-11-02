"""tools/analysis/factorswap_s2_readout.py on synthetic logs in the real format.

The metric block is copied from tools/logs/error_26843_oracleZsh.txt (epoch 152): the
`Car AP_R40@0.70, 0.70, 0.70:` block gives Car_bev/moderate_R40 etc. (eval.py ret_dict keys),
the `Car AP_R40@0.70, 0.50, 0.50:` block the IoU-0.5 reading. Pins: the parsed values, the
W&B run id, the R1 (R2 − R1) and R10 (R3 − R10) exceptions, the core sum, both verdicts and
the one-seed reading, plus pending rows under --missing-ok.
"""
import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(HERE, '..', 'tools', 'analysis', 'factorswap_s2_readout.py')
spec = importlib.util.spec_from_file_location('factorswap_s2_readout', TOOL)
fs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fs)

HEAD = ("wandb: Run data is saved locally in /storage/wandb/run-20261001_005323-{run_id}\n"
        "[2026-10-01 00:53:28,607  config.py 13  INFO]  cfg.OPTIMIZATION.NUM_EPOCHS: 152\n")
# tqdm lines end with \r, like the real stderr log
PROGRESS = "epochs:   8%|▊         | 12/152 [02:00<23:00,  5.9s/it]\repochs:  10%|▉         | 15/152\r"


def eval_block(epoch, bev07, d307, bev05=89.1438, d305=88.5180, easy=88.7970, hard=80.1218):
    return (
        "[2026-10-01 18:40:00,661  eval_utils.py 106  INFO]  *************** Performance of EPOCH %d *****************\n"
        "[2026-10-01 18:40:00,662  eval_utils.py 108  INFO]  Generate label finished(sec_per_example: 0.0800 second).\n"
        "[2026-10-01 18:40:00,664  eval_utils.py 126  INFO]  recall_rcnn_0.7: 0.636063\n"
        "[2026-10-01 18:42:04,826  eval_utils.py 189  INFO]  Car AP@0.70, 0.70, 0.70:\n"
        "bbox AP:88.4470, 85.2147, 85.2082\n"
        "bev  AP:86.4983, 77.4350, 77.5833\n"
        "3d   AP:81.9832, 71.0704, 67.2886\n"
        "aos  AP:88.42, 85.03, 84.86\n"
        "Car AP_R40@0.70, 0.70, 0.70:\n"
        "bbox AP:91.8136, 86.5900, 86.7175\n"
        "bev  AP:%.4f, %.4f, %.4f\n"
        "3d   AP:82.5437, %.4f, 69.3276\n"
        "aos  AP:91.78, 86.39, 86.35\n"
        "Car AP@0.70, 0.50, 0.50:\n"
        "bbox AP:88.4470, 85.2147, 85.2082\n"
        "bev  AP:89.7761, 86.6309, 86.7737\n"
        "3d   AP:89.7529, 86.3860, 86.5144\n"
        "aos  AP:88.42, 85.03, 84.86\n"
        "Car AP_R40@0.70, 0.50, 0.50:\n"
        "bbox AP:91.8136, 86.5900, 86.7175\n"
        "bev  AP:92.5182, %.4f, 89.4227\n"
        "3d   AP:92.4263, %.4f, 88.7684\n"
        "aos  AP:91.78, 86.39, 86.35\n"
        "Pedestrian AP@0.50, 0.50, 0.50:\n"
        "bbox AP:59.5989, 56.7790, 53.9819\n"
        "bev  AP:49.5142, 45.2426, 41.8193\n"
        "3d   AP:45.1453, 40.8059, 37.2848\n"
        "aos  AP:56.44, 52.87, 49.83\n"
        "Pedestrian AP_R40@0.50, 0.50, 0.50:\n"
        "bbox AP:58.7764, 56.1405, 52.8024\n"
        "bev  AP:48.0384, 43.5586, 39.7661\n"
        "3d   AP:43.4371, 38.8687, 34.8813\n"
        "aos  AP:55.45, 51.82, 48.22\n"
        "\n"
        "[2026-10-01 18:42:04,837  eval_utils.py 196  INFO]  *************** EPOCH %d of kitti EVALUATION end.*****************\n"
    ) % (epoch, easy, bev07, hard, d307, bev05, d305, epoch)


def write_job(log_dir, job, tag, run_id, final=None, earlier=None):
    text = HEAD.format(run_id=run_id) + PROGRESS
    if earlier is not None:
        text += eval_block(*earlier)
    if final is not None:
        text += eval_block(*final)
    with open(os.path.join(log_dir, 'error_%d_%s.txt' % (job, tag)), 'w') as f:
        f.write(text)
    with open(os.path.join(log_dir, 'output_%d_%s.txt' % (job, tag)), 'w') as f:
        f.write('W&B run directory: /storage/wandb/run-20261001_005323-%s/files\n' % run_id)


# row -> (job, run_id, final BEV@0.7 moderate, 3D@0.7 moderate)
ROWS = {
    'O': (26843, '06iwywhs', 80.3301, 70.5866),
    'R1': (31409, 'r1r1r1r1', 60.00, 50.0),   # term = R2 − R1 = 70 − 60 = 10
    'R2': (31410, 'r2r2r2r2', 70.00, 60.0),   # control; O − R2 = 10.33 (not summed)
    'R3': (31414, 'r3r3r3r3', 72.33, 62.0),   # O − R3 = 8.00
    'R4': (31411, 'r4r4r4r4', 78.33, 68.0),   # 2.00 -> not detectable
    'R5': (31415, 'r5r5r5r5', 74.33, 64.0),   # 6.00
    'R6': (31416, 'r6r6r6r6', 70.33, 60.0),   # 10.00
    'R7': (31417, 'r7r7r7r7', 77.33, 67.0),   # 3.00 -> read (boundary)
    'R8': (31412, 'r8r8r8r8', 79.33, 69.0),
    'R9': (31413, 'r9r9r9r9', 79.00, 69.0),
    'R10': (31420, 'r10r10r1', 65.33, 55.0),  # term = R3 − R10 = 72.33 − 65.33 = 7.00
}
# core sum = 10 + 8 + 2 + 6 + 10 + 3 = 39.00; |39.00 − 38.76| <= 5


def make_logs(tmp_path, skip=()):
    d = tmp_path / 'logs'
    d.mkdir()
    for row, (job, rid, bev, d3) in ROWS.items():
        if row in skip:
            write_job(str(d), job, row, rid, final=None)
        else:
            # an earlier, worse epoch first: the tool must take the LAST block
            write_job(str(d), job, row, rid, final=(152, bev, d3), earlier=(151, bev - 5.0, d3 - 5.0))
    return str(d)


def test_parse_real_format_oracle(tmp_path):
    d = make_logs(tmp_path)
    r = fs.read_job(26843, d)
    assert r['run_id'] == '06iwywhs'
    assert r['epoch'] == 152
    m = r['metrics']
    assert m['Car_bev/moderate_R40'] == pytest.approx(80.3301)
    assert m['Car_3d/moderate_R40'] == pytest.approx(70.5866)
    assert m['Car_bev/easy_R40'] == pytest.approx(88.7970)
    assert m['Car_bev/hard_R40'] == pytest.approx(80.1218)
    assert m['Car_bev/moderate_R40@0.5'] == pytest.approx(89.1438)
    assert m['Car_3d/moderate_R40@0.5'] == pytest.approx(88.5180)
    # the 11-point block and other classes must not leak into the Car R40 keys
    assert m['Car_image/moderate_R40'] == pytest.approx(86.5900)
    assert 'Pedestrian_bev/moderate_R40' in m and m['Pedestrian_bev/moderate_R40'] == pytest.approx(43.5586)
    assert r['progress'] == '15/152 epochs'


def test_tr_cr_semantics():
    lines = fs.split_log_lines('a\rb\nc\r\nd')
    assert lines == ['a', 'b', 'c', '', 'd']


def test_table_terms_sum_verdict(tmp_path):
    d = make_logs(tmp_path)
    results = {r: fs.read_job(fs.DEFAULT_JOBS[r][0], d) for r in fs.ROW_ORDER}
    table, terms = fs.build_table(results)
    by_row = {t['row']: t for t in table}
    assert terms['O'] is None
    assert terms['R1'] == pytest.approx(10.0)       # R2 − R1, not O − R1 (= 20.33)
    assert terms['R10'] == pytest.approx(7.0)       # R3 − R10, not O − R10 (= 15.0)
    assert terms['R2'] == pytest.approx(80.3301 - 70.0)
    assert terms['R3'] == pytest.approx(8.0, abs=1e-3)
    assert terms['R4'] == pytest.approx(2.0, abs=1e-3)
    assert 'not detectable at one seed' in by_row['R4']['term_s']
    assert by_row['R7']['term_s'].endswith('(read)')
    assert by_row['R1']['term_s'].startswith('R2 − R1')
    assert by_row['R10']['term_s'].startswith('R3 − R10')
    assert by_row['R3']['run_id'] == 'r3r3r3r3'
    assert by_row['R1']['bev07'] == pytest.approx(60.0)
    s, missing, verdict = fs.summarize(terms)
    assert missing == []
    assert s == pytest.approx(39.0, abs=1e-3)
    assert verdict.endswith('consistent at one seed')
    text = fs.render(table, terms, missing_ok=False)
    assert '| R10 | fg + bg (read against R3) | 31420 | r10r10r1 | 65.33 / 55.00 |' in text
    assert 'Sum = 39.00; gap O − control = 38.8; ' in text
    assert 'consistent at one seed' in text


def test_verdict_interaction():
    terms = {'R1': 10.0, 'R3': 8.0, 'R4': 2.0, 'R5': 6.0, 'R6': 10.0, 'R7': 9.0}   # 45 > 38.8 + 5
    s, missing, verdict = fs.summarize(terms)
    assert s == pytest.approx(45.0)
    assert verdict.endswith('interaction + unmeasured')
    terms['R7'] = -2.0                                                             # 34 -> within 5
    assert fs.summarize(terms)[2].endswith('consistent at one seed')
    terms['R7'] = -6.0                                                             # 30 -> outside
    assert fs.summarize(terms)[2].endswith('interaction + unmeasured')


def test_missing_ok_pending(tmp_path, capsys):
    d = make_logs(tmp_path, skip=('R3', 'R10'))
    rc = fs.main(['--missing-ok', '--log-dir', d])
    out = capsys.readouterr().out
    assert rc == 0
    assert '| R3 | density | 31414 | r3r3r3r3 | - / - | - / - | - / - | pending (15/152 epochs) | O − R3: pending |' in out
    assert 'R3 − R10: pending' in out
    assert 'verdict pending (R3 not final)' in out
    assert 'Sum so far: 31.00 (missing R3)' in out
    # without the flag the pending rows are an error
    rc = fs.main(['--log-dir', d])
    assert rc == 1
    assert 'no final evaluation yet for R3, R10' in capsys.readouterr().err


def test_oracle_fallback_when_oracle_log_missing(tmp_path):
    d = make_logs(tmp_path, skip=('O',))
    results = {r: fs.read_job(fs.DEFAULT_JOBS[r][0], d) for r in fs.ROW_ORDER}
    table, terms = fs.build_table(results)
    o = [t for t in table if t['row'] == 'O'][0]
    assert o['bev07'] == pytest.approx(80.33) and o['3d07'] == pytest.approx(70.59)
    assert o['status'].startswith('hardcoded')
    assert terms['R3'] == pytest.approx(80.33 - 72.33, abs=1e-3)
