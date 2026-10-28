"""COND-TAB: weather / illumination tags of Waymo and nuScenes per split and cell (ANALYSIS; experiments_md 20261028_01).

Design FIXED in experiments_md T05 §5, entry "User 10-28 (received between 07:50 and 11:39)", 11:45 JST, before any
number; applied here unchanged. Tags are metadata; GT boxes are counted for diagnosis only; nothing here selects or
feeds a method value (memory: uda_legality_rule.md, analysis exception). CPU only.

Waymo (raw v1.4 tfrecords, training / validation / domain_adaptation):
- Tags from Frame.context.stats (time_of_day = 2, location = 3, weather = 4; laser_object_counts = 1), parsed straight
  from the protobuf wire format with `waymo_calib._fields` (no tensorflow / waymo_open_dataset: neither is in the
  container). Context.name (= 1) is checked against the file name for every segment.
- The context of EVERY frame of EVERY segment is read (Frame.context is field 1, written first), so the constancy check
  covers all frames, not a sample; the tag of a segment is its FIRST frame's.
- Frames per segment = records in the tfrecord (headers walked with pread).
- Join with the processed infos the loader reads (per-sequence `waymo_processed_data/<seq>/<seq>.pkl`, ImageSets
  order, sampling over the concatenated list exactly as `WaymoDataset.include_waymo_data`). GT Car = annos.name ==
  'Vehicle' (Vehicle -> Car, as `map_name_to_kitti`), counted (a) raw, all stored boxes (what the KITTI-metric
  evaluation reads as GT), and (b) stored num_points_in_gt >= 1 (MIN_POINTS_OF_GT, what training keeps before the
  loader's post-augmentation range filter, which is not applied here).
nuScenes (v1.0-trainval): tag rule copied from E2 (`nuscenes_condition_ap.frame_conditions`): `rain` / `night` as
substrings of the lower-cased scene.description; city from log.location (boston-seaport = Boston, singapore-* =
Singapore). Joined to the 10-sweep train / val infos of the nuScenes configs. GT Car = gt_names == 'car', raw and with
num_lidar_pts >= 1 (FILTER_MIN_POINTS_IN_GT 1). A word-boundary variant of the rule is computed only to list scenes
where it would differ.

    singularity exec --bind /home/koyama/data/:/storage <sif> python3 analysis/condition_tags.py \\
        --out /home/koyama/data/cond_tab_20261028 --workers 8

Run from ST3D/tools. Gates (exit 3 if any fails, outputs still written): the loader totals of the logs reproduce
(Waymo train 158,081 / interval 2 79,041 / interval 10 15,809; val 39,987 / interval 5 7,998; nuScenes 28,130 / 6,019);
every Waymo pkl exists; and the E2 val cells reproduce.
"""
import argparse
import csv
import json
import os
import pickle
import re
import struct
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from waymo_calib import _fields, _varint  # noqa: E402  (protobuf wire-format reader, ST3D adda12b)

WAYMO = Path('/home/koyama/data/waymo_open_dataset_v_1_4_0')
RAW = WAYMO / 'individual_files'
PCDET = WAYMO / 'pcdet_structure'
NUSC = Path('/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval')
META = NUSC / 'v1.0-trainval'

# (split label, directory, labelled) for every tfrecord set read
WAYMO_SETS = [
    ('train', RAW / 'training', True),
    ('val', RAW / 'validation', True),
    ('da_train', RAW / 'domain_adaptation' / 'training', True),
    ('da_train_unlabeled', RAW / 'domain_adaptation' / 'training' / 'unlabeled', False),
    ('da_val', RAW / 'domain_adaptation' / 'validation', True),
    ('da_val_unlabeled', RAW / 'domain_adaptation' / 'validation' / 'unlabeled', False),
    ('da_test', RAW / 'domain_adaptation' / 'testing', False),
]
# Sampling intervals, read from the logs (quoted in the report):
#  oracle training 26398 (cfgs/da-ieee-access/centerpoint-sourceonly-waymo.yaml) and its eval 27179
#  (centerpoint-sourceonly-waymo2waymo.yaml): train 2 -> 79,041 of 158,081; test 5 -> 7,998 of 39,987.
#  27704 (centerpoint-gblobs-accum-legaldepth-st3d-nuscenes2waymo.yaml) / PL-GT: DATA_CONFIG_TAR train 10 -> 15,809;
#  test 5 -> 7,998.
INTERVALS = {'train': (1, 2, 10), 'val': (1, 5)}
LOADER_TOTALS = {('train', 1): 158081, ('train', 2): 79041, ('train', 10): 15809, ('val', 1): 39987, ('val', 5): 7998}
NUSC_TOTALS = {'train': 28130, 'val': 6019}
PANDASET_TRAIN = 4880   # 'Total samples for PandaSet dataset (train): 4880' (26389, 26974, 29327, 29330, 28034)
MIN_VAL_FRAMES = 300
# E2 val cells (experiments_md 20261024_02 §5): (city, cell) -> (scenes, frames, GT cars)
E2_VAL = {('Boston', 'clear'): (53, 2122, 31040), ('Boston', 'rain'): (24, 968, 12740),
          ('Boston', 'night'): (0, 0, 0), ('Boston', 'both'): (0, 0, 0),
          ('Singapore', 'clear'): (58, 2327, 18715), ('Singapore', 'rain'): (0, 0, 0),
          ('Singapore', 'night'): (12, 482, 2566), ('Singapore', 'both'): (3, 120, 404)}
CELLS = ('clear', 'rain', 'night', 'both')
VEHICLE_TYPE = 1        # Label.Type TYPE_VEHICLE


# ----------------------------------------------------------------------------------------------- Waymo tfrecords
def _context(payload_head, fd, off_payload):
    """Frame.context bytes: field 1 (wire type 2) is serialized first; returns (bytes, ok)."""
    key, i = _varint(payload_head, 0)
    if key != (1 << 3 | 2):
        return None
    ln, i = _varint(payload_head, i)
    return os.pread(fd, ln, off_payload + i)


def _parse_context(ctx):
    name, tod, loc, wea, veh = None, None, None, None, 0
    for fn, wt, v in _fields(ctx):
        if fn == 1 and wt == 2:
            name = bytes(v).decode()
        elif fn == 4 and wt == 2:
            for f2, w2, v2 in _fields(v):
                if f2 == 2 and w2 == 2:
                    tod = bytes(v2).decode()
                elif f2 == 3 and w2 == 2:
                    loc = bytes(v2).decode()
                elif f2 == 4 and w2 == 2:
                    wea = bytes(v2).decode()
                elif f2 == 1 and w2 == 2:
                    d = {a: b for a, _, b in _fields(v2)}
                    if d.get(1, 0) == VEHICLE_TYPE:
                        veh += d.get(2, 0)
    return name, tod, loc, wea, veh


def scan_tfrecord(args):
    """Every frame's (name, time_of_day, location, weather, laser Vehicle count) and the record count."""
    split, path = args
    fd = os.open(path, os.O_RDONLY)
    frames = []
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        size, off = os.fstat(fd).st_size, 0
        while off < size:
            head = os.pread(fd, 12 + 16, off)
            ln = struct.unpack('<Q', head[:8])[0]
            ctx = _context(head[12:], fd, off + 12)
            if ctx is None:                       # not expected; fall back to the whole record
                ctx = next(v for fn, wt, v in _fields(os.pread(fd, ln, off + 12)) if fn == 1 and wt == 2)
            frames.append(_parse_context(ctx))
            off += 8 + 4 + ln + 4
        assert off == size, (path, off, size)
    finally:
        os.close(fd)
    seg = Path(path).stem
    expect = seg.replace('segment-', '', 1).replace('_with_camera_labels', '')
    tags = [f[:4] for f in frames]
    first = tags[0]
    return dict(split=split, segment=seg, path=str(path), frames=len(frames), context_name=first[0],
                name_ok=first[0] == expect, time_of_day=first[1], location=first[2], weather=first[3],
                constant=all(t == first for t in tags),
                n_distinct=len(set(tags)), stats_vehicle=[f[4] for f in frames])


def waymo_scan(workers, cache):
    if cache.exists():
        print(f'reusing the tfrecord scan {cache}', flush=True)
        return json.load(open(cache))
    jobs = []
    for split, d, _ in WAYMO_SETS:
        files = sorted(p for p in d.iterdir() if p.suffix == '.tfrecord')
        print(f'{split}: {len(files)} tfrecords in {d}', flush=True)
        jobs += [(split, str(p)) for p in files]
    t0, out = time.time(), []
    with Pool(workers) as pool:
        for k, r in enumerate(pool.imap(scan_tfrecord, jobs, chunksize=2)):
            out.append(r)
            if (k + 1) % 100 == 0:
                print(f'  scanned {k + 1}/{len(jobs)} ({time.time() - t0:.0f} s)', flush=True)
    json.dump(out, open(cache, 'w'))
    print(f'scan done: {len(out)} segments, {time.time() - t0:.0f} s -> {cache}', flush=True)
    return out


# ----------------------------------------------------------------------------------------------- Waymo infos
def waymo_join(scan):
    """Per frame of the loader's global list: (segment, sample_idx, Vehicle raw, Vehicle pts>=1, all boxes)."""
    by_seg = {s['segment']: s for s in scan if s['split'] in ('train', 'val')}
    out, checks = {}, {}
    for split in ('train', 'val'):
        seqs = [os.path.splitext(x.strip())[0] for x in open(PCDET / 'ImageSets' / f'{split}.txt')]
        rows, missing, pkl_vs_tf, frame_mismatch = [], [], [], 0
        for seq in seqs:
            p = PCDET / 'waymo_processed_data' / seq / f'{seq}.pkl'
            if not p.exists():
                missing.append(seq)
                continue
            infos = pickle.load(open(p, 'rb'))
            sv = by_seg[seq]['stats_vehicle'] if seq in by_seg else None
            if seq not in by_seg or len(infos) != by_seg[seq]['frames']:
                pkl_vs_tf.append((seq, len(infos), by_seg[seq]['frames'] if seq in by_seg else None))
            for info in infos:
                a = info['annos']
                veh = a['name'] == 'Vehicle'
                n_raw = int(veh.sum())
                n_pts = int((veh & (a['num_points_in_gt'] >= 1)).sum())
                idx = int(info['point_cloud']['sample_idx'])
                if sv is not None and (idx >= len(sv) or sv[idx] != n_raw):
                    frame_mismatch += 1
                rows.append((seq, idx, n_raw, n_pts, int(len(a['name']))))
        out[split] = rows
        checks[split] = dict(sequences=len(seqs), missing_pkl=missing, pkl_frames_ne_tfrecord=pkl_vs_tf,
                             frames_stats_vehicle_ne_pkl=frame_mismatch, frames=len(rows))
        print(f'{split}: {len(seqs)} sequences, {len(missing)} pkl missing, {len(rows)} frames, '
              f'{len(pkl_vs_tf)} with pkl frames != tfrecord records, {frame_mismatch} frames whose '
              f'context.stats Vehicle count != the pkl Vehicle count', flush=True)
    return out, checks


# ----------------------------------------------------------------------------------------------- nuScenes
def nuscenes_tags():
    logs = {l['token']: l for l in json.load(open(META / 'log.json'))}
    scenes = {s['token']: s for s in json.load(open(META / 'scene.json'))}
    sample_scene = {s['token']: s['scene_token'] for s in json.load(open(META / 'sample.json'))}
    tag = {}
    for tok, sc in scenes.items():
        loc = logs[sc['log_token']]['location']
        assert loc == 'boston-seaport' or loc.startswith('singapore-'), loc
        d = sc['description'].lower()
        rain, night = 'rain' in d, 'night' in d                  # E2's rule, unchanged
        w_rain, w_night = bool(re.search(r'\brain\b', d)), bool(re.search(r'\bnight\b', d))
        cell = {(False, False): 'clear', (True, False): 'rain', (False, True): 'night', (True, True): 'both'}[
            (rain, night)]
        tag[tok] = dict(scene=sc['name'], city='Boston' if loc == 'boston-seaport' else 'Singapore', location=loc,
                        description=sc['description'], cell=cell, rule_word_differs=(rain, night) != (w_rain, w_night),
                        nbr_samples=sc['nbr_samples'])
    return tag, sample_scene


def nuscenes_rows(tag, sample_scene):
    out = {}
    for split in ('train', 'val'):
        infos = pickle.load(open(NUSC / f'nuscenes_infos_10sweeps_{split}.pkl', 'rb'))
        rows = []
        for info in infos:
            car = info['gt_names'] == 'car'
            rows.append((sample_scene[info['token']], int(car.sum()), int((car & (info['num_lidar_pts'] >= 1)).sum())))
        out[split] = rows
        print(f'nuScenes {split}: {len(rows)} keyframes', flush=True)
    return out


# ----------------------------------------------------------------------------------------------- tables
def waymo_tables(scan, joined):
    seg = {s['segment']: s for s in scan}
    tables = {}
    for split in ('train', 'val'):
        rows = joined[split]
        acc = defaultdict(lambda: defaultdict(int))
        segs = defaultdict(set)
        for k, (sq, idx, n_raw, n_pts, n_all) in enumerate(rows):
            s = seg[sq]
            key = (s['location'], s['time_of_day'], s['weather'])
            segs[key].add(sq)
            for iv in INTERVALS[split]:
                if k % iv == 0:
                    c = acc[key]
                    c[f'frames_i{iv}'] += 1
                    c[f'car_raw_i{iv}'] += n_raw
                    c[f'car_pts1_i{iv}'] += n_pts
                    c[f'boxes_all_i{iv}'] += n_all
        tables[split] = {'|'.join(k): dict(segments=len(segs[k]), **v) for k, v in sorted(acc.items())}
    for split in ('da_train', 'da_train_unlabeled', 'da_val', 'da_val_unlabeled', 'da_test'):
        ss = sorted((s for s in scan if s['split'] == split), key=lambda s: s['segment'])
        acc = defaultdict(lambda: defaultdict(int))
        k = 0
        for s in ss:
            key = '|'.join((s['location'], s['time_of_day'], s['weather']))
            c = acc[key]
            c['segments'] += 1
            c['frames_i1'] += s['frames']
            c['stats_vehicle_i1'] += sum(s['stats_vehicle'])
            for j in range(s['frames']):          # hypothetical: as the loader would sample a sorted list
                for iv in (2, 5, 10):
                    if (k + j) % iv == 0:
                        c[f'frames_i{iv}'] += 1
            k += s['frames']
        tables[split] = {kk: dict(v) for kk, v in sorted(acc.items())}
    return tables


def margins(table, fields):
    """Pool a location|tod|weather table over location -> tod|weather."""
    out = defaultdict(lambda: defaultdict(int))
    for k, v in table.items():
        _, tod, wea = k.split('|')
        for f in fields:
            out[f'{tod}|{wea}'][f] += v.get(f, 0)
    return {k: dict(v) for k, v in sorted(out.items())}


def nuscenes_tables(tag, rows):
    tables = {}
    for split in ('train', 'val'):
        acc = defaultdict(lambda: defaultdict(int))
        scenes = defaultdict(set)
        for sc, n_raw, n_pts in rows[split]:
            t = tag[sc]
            key = f"{t['city']}|{t['cell']}"
            scenes[key].add(sc)
            acc[key]['frames'] += 1
            acc[key]['car_raw'] += n_raw
            acc[key]['car_pts1'] += n_pts
        tables[split] = {f'{city}|{cell}': dict(scenes=len(scenes[f'{city}|{cell}']), **acc[f'{city}|{cell}'])
                         for city in ('Boston', 'Singapore') for cell in CELLS}
        for k, v in tables[split].items():
            v.setdefault('frames', 0)
            v.setdefault('car_raw', 0)
            v.setdefault('car_pts1', 0)
    return tables


def feasibility(val_frames, train_frames):
    return dict(stratified=('yes' if val_frames >= MIN_VAL_FRAMES else 'no'),
                condition_oracle=('yes' if train_frames >= PANDASET_TRAIN else 'no'))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fails = []

    # ---- Waymo
    scan = waymo_scan(a.workers, out / 'waymo_tfrecord_scan.json')
    vocab = {f: dict(Counter(s[f] for s in scan)) for f in ('time_of_day', 'location', 'weather')}
    bad_name = [s['segment'] for s in scan if not s['name_ok']]
    not_const = [dict(segment=s['segment'], split=s['split'], n_distinct=s['n_distinct']) for s in scan
                 if not s['constant']]
    per_split = {sp: sum(1 for s in scan if s['split'] == sp) for sp, _, _ in WAYMO_SETS}
    frames_split = {sp: sum(s['frames'] for s in scan if s['split'] == sp) for sp, _, _ in WAYMO_SETS}
    print(f'Waymo segments per set {per_split}; frames {frames_split}', flush=True)
    print(f'vocabulary {vocab}', flush=True)
    print(f'context.name != file name: {len(bad_name)}; segments whose tags vary across frames: {len(not_const)} '
          f'(frames checked {sum(s["frames"] for s in scan)})', flush=True)
    if bad_name:
        fails.append(f'context.name mismatch on {len(bad_name)} segments')
    joined, jchecks = waymo_join(scan)
    for (split, iv), n in LOADER_TOTALS.items():
        got = len(range(0, len(joined[split]), iv))
        ok = got == n
        print(f'gate: Waymo {split} interval {iv}: {got} frames vs the log {n} -> {"PASS" if ok else "FAIL"}', flush=True)
        if not ok:
            fails.append(f'Waymo {split} interval {iv}: {got} != {n}')
    for split in ('train', 'val'):
        if jchecks[split]['missing_pkl']:
            fails.append(f'Waymo {split}: {len(jchecks[split]["missing_pkl"])} pkl missing')
    wt = waymo_tables(scan, joined)
    wm = {'train': margins(wt['train'], ['segments', 'frames_i1', 'frames_i2', 'frames_i10', 'car_raw_i1',
                                         'car_raw_i2', 'car_raw_i10', 'car_pts1_i2', 'car_pts1_i10']),
          'val': margins(wt['val'], ['segments', 'frames_i1', 'frames_i5', 'car_raw_i1', 'car_raw_i5', 'car_pts1_i5'])}
    for kind, tbl in (('cell', wt), ('margin', wm)):
        for split in ('train', 'val'):
            print(f'-- Waymo {split} per {"location|tod|weather" if kind == "cell" else "tod|weather (pooled)"}',
                  flush=True)
            for k, v in tbl[split].items():
                print(f'   {k:36s} {json.dumps(v)}', flush=True)
    for split in ('da_train', 'da_train_unlabeled', 'da_val', 'da_val_unlabeled', 'da_test'):
        print(f'-- Waymo {split}', flush=True)
        for k, v in wt[split].items():
            print(f'   {k:36s} {json.dumps(v)}', flush=True)
    # val all-class boxes at interval 5 against 27179's 'total num of gt: 484979'
    val_boxes_i5 = sum(v['boxes_all_i5'] for v in wt['val'].values())
    print(f'check: Waymo val boxes of every class at interval 5: {val_boxes_i5} (27179 logs "total num of gt: 484979")',
          flush=True)

    # ---- nuScenes
    tag, sample_scene = nuscenes_tags()
    nrows = nuscenes_rows(tag, sample_scene)
    for split, n in NUSC_TOTALS.items():
        ok = len(nrows[split]) == n
        print(f'gate: nuScenes {split}: {len(nrows[split])} keyframes vs the log {n} -> {"PASS" if ok else "FAIL"}',
              flush=True)
        if not ok:
            fails.append(f'nuScenes {split}: {len(nrows[split])} != {n}')
    nt = nuscenes_tables(tag, nrows)
    e2 = {}
    for (city, cell), (sc, fr, gt) in E2_VAL.items():
        v = nt['val'][f'{city}|{cell}']
        ok = (v['scenes'], v['frames']) == (sc, fr)          # the gate: scenes and frames
        gt_ok = v['car_raw'] == gt                            # reported beside, not gating
        e2[f'{city}|{cell}'] = dict(expected=[sc, fr, gt], got=[v['scenes'], v['frames'], v['car_raw']], ok=ok,
                                    gt_cars_ok=gt_ok)
        print(f'gate E2: val {city} {cell}: scenes / frames {v["scenes"]} / {v["frames"]} vs E2 {sc} / {fr} -> '
              f'{"PASS" if ok else "FAIL"}; GT cars {v["car_raw"]} vs E2 {gt} -> {"same" if gt_ok else "DIFFERENT"}',
              flush=True)
        if not ok:
            fails.append(f'E2 {city} {cell}')
    split_scenes = {sp: sorted({r[0] for r in nrows[sp]}) for sp in ('train', 'val')}
    word_diff = [dict(split=sp, scene=tag[s]['scene'], description=tag[s]['description'], cell=tag[s]['cell'])
                 for sp in ('train', 'val') for s in split_scenes[sp] if tag[s]['rule_word_differs']]
    print(f'nuScenes scenes where a word-boundary rule would differ from E2\'s substring rule: {len(word_diff)}',
          flush=True)
    for w in word_diff:
        print(f'   {w}', flush=True)
    for split in ('train', 'val'):
        print(f'-- nuScenes {split} per city|cell', flush=True)
        for k, v in nt[split].items():
            print(f'   {k:20s} {json.dumps(v)}', flush=True)

    # ---- feasibility (pre-declared rule, applied mechanically)
    feas = dict(waymo_cell={}, waymo_margin={}, waymo_da={}, nuscenes_cell={}, nuscenes_pooled={})
    for k in sorted(set(wt['train']) | set(wt['val'])):
        tr = wt['train'].get(k, {}).get('frames_i2', 0)
        va = wt['val'].get(k, {}).get('frames_i5', 0)
        feas['waymo_cell'][k] = dict(train_i2=tr, val_i5=va, **feasibility(va, tr))
    for k in sorted(set(wm['train']) | set(wm['val'])):
        tr = wm['train'].get(k, {}).get('frames_i2', 0)
        va = wm['val'].get(k, {}).get('frames_i5', 0)
        feas['waymo_margin'][k] = dict(train_i2=tr, val_i5=va, **feasibility(va, tr))
    for k in sorted(set(wt['da_train']) | set(wt['da_val'])):
        tr = wt['da_train'].get(k, {}).get('frames_i2', 0)
        va = wt['da_val'].get(k, {}).get('frames_i5', 0)
        feas['waymo_da'][k] = dict(train_i2=tr, val_i5=va, **feasibility(va, tr), note='no processed infos')
    for k in nt['train']:
        tr, va = nt['train'][k]['frames'], nt['val'][k]['frames']
        feas['nuscenes_cell'][k] = dict(train=tr, val=va, **feasibility(va, tr))
    for cell in CELLS:
        tr = sum(nt['train'][f'{c}|{cell}']['frames'] for c in ('Boston', 'Singapore'))
        va = sum(nt['val'][f'{c}|{cell}']['frames'] for c in ('Boston', 'Singapore'))
        feas['nuscenes_pooled'][cell] = dict(train=tr, val=va, **feasibility(va, tr))
    print('-- feasibility (val >= 300 frames at eval sampling; train at the oracle sampling >= 4,880)', flush=True)
    for kind, d in feas.items():
        for k, v in d.items():
            print(f'   {kind:16s} {k:36s} {json.dumps(v)}', flush=True)

    # ---- outputs
    seg_rows = []
    pkl_frames = defaultdict(int)
    for split in ('train', 'val'):
        for sq, *_ in joined[split]:
            pkl_frames[sq] += 1
    for s in scan:
        seg_rows.append(dict(split=s['split'], segment=s['segment'], context_name=s['context_name'],
                             location=s['location'], time_of_day=s['time_of_day'], weather=s['weather'],
                             frames_tfrecord=s['frames'], tags_constant=s['constant'],
                             stats_vehicle_sum=sum(s['stats_vehicle']), pkl_frames=pkl_frames.get(s['segment'], '')))
    write_csv(out / 'waymo_segments.csv', seg_rows, list(seg_rows[0]))
    sc_rows = []
    for sp in ('train', 'val'):
        fr = Counter(r[0] for r in nrows[sp])
        cr = Counter()
        for r in nrows[sp]:
            cr[r[0]] += r[1]
        for s in split_scenes[sp]:
            t = tag[s]
            sc_rows.append(dict(split=sp, scene=t['scene'], scene_token=s, city=t['city'], location=t['location'],
                                cell=t['cell'], keyframes=fr[s], car_raw=cr[s], description=t['description']))
    write_csv(out / 'nuscenes_scenes.csv', sc_rows, list(sc_rows[0]))
    result = dict(
        waymo=dict(tables=wt, margins=wm, vocab=vocab, segments_per_set=per_split, frames_per_set=frames_split,
                   context_name_mismatch=bad_name, tags_not_constant=not_const,
                   frames_checked=sum(s['frames'] for s in scan), join_checks=jchecks,
                   val_boxes_all_i5=val_boxes_i5, intervals=INTERVALS, loader_totals={f'{k[0]}_i{k[1]}': v for k, v in
                                                                                         LOADER_TOTALS.items()}),
        nuscenes=dict(tables=nt, e2_check=e2, rule_word_differs=word_diff, totals=NUSC_TOTALS),
        feasibility=feas, pandaset_train=PANDASET_TRAIN, min_val_frames=MIN_VAL_FRAMES, gate_failures=fails)
    with open(out / 'cond_tab.json', 'w') as f:
        json.dump(result, f, indent=1)
    print(f'wrote {out}/cond_tab.json, waymo_segments.csv, nuscenes_scenes.csv', flush=True)
    if fails:
        print(f'GATE FAILURES: {fails}', flush=True)
        sys.exit(3)
    print('all gates PASS', flush=True)


if __name__ == '__main__':
    main()
