"""P2-B condition oracles (ANALYSIS, target labels trained on; never a method): build the four arms' TRAIN frame sets.

Design: experiments_md 20261003_05 (T05) §5 "PHASE 2, design FIXED 12:04", design commit af7680c; report
experiments_md 20261028_03. Tags and counts: experiments_md 20261028_01 (COND-TAB, ST3D 6acd5bd, job 30914).

    python analysis/condoracle_build.py [--out /home/koyama/data/cond_oracle_20261028]

Data-side only; no loader code changes:
- Waymo: a split list (ImageSets/train.txt lines, kept in train.txt order) that DATA_SPLIT.train names by ABSOLUTE
  path. WaymoDataset reads `root_path / 'ImageSets' / (split + '.txt')`; a pathlib join with an absolute path yields
  that path, and `split` is used nowhere else on the training path.
- nuScenes: a filtered copy of nuscenes_infos_10sweeps_train.pkl (original order, info dicts untouched) that
  INFO_PATH.train names by ABSOLUTE path (`root_path / info_path` likewise).

Arms (seed 0 draws are recorded in arms.json next to the outputs):
- Waymo Night: every train segment tagged time_of_day Night, weather sunny (78: sf 36, phx 29, other 13).
- Waymo Day: Day | sunny train segments drawn per location to Night's counts. numpy default_rng(0), ONE generator,
  locations in the order sf, phx, other; candidates of a location sorted by segment name; rng.choice(len, n,
  replace=False); the drawn list is then written in train.txt order.
- nuScenes Boston rain: every Boston rain-only train scene (125).
- nuScenes Boston clear: 125 of the 265 Boston clear train scenes, default_rng(0).choice(265, 125, replace=False)
  over the scenes sorted by name.

The loader samples Waymo every SAMPLED_INTERVAL-th frame of the CONCATENATED list, so an arm's frame count at
interval 2 is ceil(arm frames / 2), not the count those segments contribute under the full train list's phase
(COND-TAB's 7,707 for Night). Both expectations are written to arms.json.
"""
import argparse, csv, hashlib, json, math, os, pickle, sys
from collections import Counter
import numpy as np

TAB = '/home/koyama/data/cond_tab_20261028'
WAYMO_ROOT = '/home/koyama/data/waymo_open_dataset_v_1_4_0/pcdet_structure'
NUSC_ROOT = '/home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval'
LOCS = ['location_sf', 'location_phx', 'location_other']


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/home/koyama/data/cond_oracle_20261028')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rec = {'design': 'T05 §5 PHASE 2 P2-B, af7680c', 'inputs': {}}

    # ------------------------------------------------------------------ Waymo
    wcsv = os.path.join(TAB, 'waymo_segments.csv')
    rec['inputs']['waymo_segments.csv'] = md5(wcsv)
    rows = [r for r in csv.DictReader(open(wcsv)) if r['split'] == 'train']
    train_lines = [l.strip() for l in open(os.path.join(WAYMO_ROOT, 'ImageSets', 'train.txt')) if l.strip()]
    val_lines = [l.strip() for l in open(os.path.join(WAYMO_ROOT, 'ImageSets', 'val.txt')) if l.strip()]
    stem = lambda l: os.path.splitext(l)[0]
    line_of = {stem(l): l for l in train_lines}
    order = {stem(l): i for i, l in enumerate(train_lines)}
    assert len(rows) == 798 and set(r['segment'] for r in rows) == set(line_of), 'COND-TAB train != ImageSets/train.txt'
    frames_of = {r['segment']: int(r['pkl_frames']) for r in rows}

    night = [r for r in rows if r['time_of_day'] == 'Night' and r['weather'] == 'sunny']
    n_per_loc = Counter(r['location'] for r in night)
    assert len(night) == 78 and n_per_loc == Counter({'location_sf': 36, 'location_phx': 29, 'location_other': 13}), n_per_loc
    rng = np.random.default_rng(0)
    day = []
    for loc in LOCS:
        cands = sorted(r['segment'] for r in rows
                       if r['time_of_day'] == 'Day' and r['weather'] == 'sunny' and r['location'] == loc)
        idx = rng.choice(len(cands), size=n_per_loc[loc], replace=False)
        day += [cands[i] for i in idx]
        rec.setdefault('waymo_day_candidates', {})[loc] = len(cands)
    loc_of = {r['segment']: r['location'] for r in rows}
    arms = {'waymo_night': sorted((r['segment'] for r in night), key=order.get),
            'waymo_day': sorted(day, key=order.get)}
    assert not set(arms['waymo_night']) & set(arms['waymo_day'])
    for arm, segs in arms.items():
        path = os.path.join(args.out, f'{arm}_sunny_train.txt')
        with open(path, 'w') as f:
            f.writelines(line_of[s] + '\n' for s in segs)
        nf = sum(frames_of[s] for s in segs)
        rec[arm] = {
            'split_file': path, 'DATA_SPLIT.train': path[:-len('.txt')], 'segments': len(segs),
            'per_location': dict(Counter(loc_of[s] for s in segs)), 'frames_all': nf,
            'frames_expected_at_interval2_own_list': math.ceil(nf / 2),
            'disjoint_from_val': not ({line_of[s] for s in segs} & set(val_lines)), 'list': segs}
    # the oracle's global-phase count of the Night segments, for the record (COND-TAB 7,707)
    gpos, kept = 0, Counter()
    for l in train_lines:
        n = frames_of[stem(l)]
        k = len(range((-gpos) % 2, n, 2))
        kept[stem(l)] = k
        gpos += n
    for arm in ('waymo_night', 'waymo_day'):
        rec[arm]['frames_at_interval2_full_list_phase'] = sum(kept[s] for s in arms[arm])

    # ------------------------------------------------------------------ nuScenes
    ncsv = os.path.join(TAB, 'nuscenes_scenes.csv')
    rec['inputs']['nuscenes_scenes.csv'] = md5(ncsv)
    srows = list(csv.DictReader(open(ncsv)))
    rain = sorted(r['scene'] for r in srows if r['split'] == 'train' and r['city'] == 'Boston' and r['cell'] == 'rain')
    clear_all = sorted(r['scene'] for r in srows if r['split'] == 'train' and r['city'] == 'Boston' and r['cell'] == 'clear')
    val_scenes = {r['scene'] for r in srows if r['split'] == 'val'}
    assert len(rain) == 125 and len(clear_all) == 265, (len(rain), len(clear_all))
    rng = np.random.default_rng(0)
    clear = sorted(clear_all[i] for i in rng.choice(len(clear_all), size=125, replace=False))
    meta = os.path.join(NUSC_ROOT, 'v1.0-trainval')
    scene_name = {s['token']: s['name'] for s in json.load(open(os.path.join(meta, 'scene.json')))}
    scene_of_sample = {s['token']: scene_name[s['scene_token']] for s in json.load(open(os.path.join(meta, 'sample.json')))}
    src = os.path.join(NUSC_ROOT, 'nuscenes_infos_10sweeps_train.pkl')
    rec['inputs']['nuscenes_infos_10sweeps_train.pkl'] = md5(src)
    infos = pickle.load(open(src, 'rb'))
    assert len(infos) == 28130, len(infos)
    kf = {'clear': csv_kf(srows, clear), 'rain': csv_kf(srows, rain)}
    for arm, scenes in (('nuscenes_boston_rain', rain), ('nuscenes_boston_clear', clear)):
        want = set(scenes)
        sub = [i for i in infos if scene_of_sample[i['token']] in want]
        got = {scene_of_sample[i['token']] for i in sub}
        assert got == want, (arm, len(got), len(want))
        path = os.path.join(args.out, f'{arm}_train_infos.pkl')
        with open(path, 'wb') as f:
            pickle.dump(sub, f)
        cars = sum(int((np.asarray(i['gt_names']) == 'car').sum()) for i in sub)
        rec[arm] = {'infos_file': path, 'scenes': len(scenes), 'keyframes': len(sub),
                    'keyframes_from_csv': kf['rain' if 'rain' in arm else 'clear'],
                    'gt_car_raw': cars, 'disjoint_from_val': not (want & val_scenes), 'list': sorted(scenes)}
    rec['nuscenes_boston_clear']['candidates'] = len(clear_all)
    with open(os.path.join(args.out, 'arms.json'), 'w') as f:
        json.dump(rec, f, indent=1)
    for arm in ('waymo_night', 'waymo_day', 'nuscenes_boston_rain', 'nuscenes_boston_clear'):
        print(arm, {k: v for k, v in rec[arm].items() if k != 'list'})
    print('wrote', os.path.join(args.out, 'arms.json'))


def csv_kf(srows, scenes):
    s = set(scenes)
    return sum(int(r['keyframes']) for r in srows if r['split'] == 'train' and r['scene'] in s)


if __name__ == '__main__':
    sys.exit(main())
