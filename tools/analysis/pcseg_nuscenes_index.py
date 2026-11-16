"""Build pcseg's nuScenes-lidarseg frame index (G0, experiments_md episodes/20261116_01 §5): one row per labelled
LIDAR_TOP keyframe, in Pointcept's order and split (pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py:
every sample of v1.0-trainval in sample.json order; train = the devkit's splits.train scenes, val = the rest).

Each row: [lidar sample_data token, lidar file relative to the detection tree, label file relative to the lidarseg tree,
scene name]. The detection tree is only read; the index goes to its own file.

    python3 analysis/pcseg_nuscenes_index.py --nuscenes /home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval \
        --lidarseg /home/koyama/data/nuscenes_lidarseg_v1_0 --out /home/koyama/data/pcseg/nuscenes_lidarseg_index_v1.json
"""
import argparse
import json
import os


def build(nuscenes, lidarseg, version='v1.0-trainval'):
    from nuscenes.utils import splits
    tab = os.path.join(nuscenes, version)
    load = lambda name: json.load(open(os.path.join(tab, name + '.json')))  # noqa: E731
    scenes = {s['token']: s['name'] for s in load('scene')}
    samples = load('sample')
    # sample.json carries no 'data' field; the devkit builds sample['data'][channel] from the keyframe sample_data
    # records (NuScenes.__make_reverse_index__). Keyframe files sit in samples/<channel>/.
    lidar_of = {}
    for r in load('sample_data'):
        if r['is_key_frame'] and r['filename'].startswith('samples/LIDAR_TOP/'):
            if r['sample_token'] in lidar_of:
                raise ValueError('sample %s: two LIDAR_TOP keyframes' % r['sample_token'])
            lidar_of[r['sample_token']] = r
    seg = {e['sample_data_token']: e['filename'] for e in json.load(open(os.path.join(lidarseg, version, 'lidarseg.json')))}
    train_names = set(splits.train)
    rows = {'train': [], 'val': []}
    for s in samples:                                                     # sample.json order, as Pointcept
        if s['token'] not in lidar_of:
            raise ValueError('sample %s: no LIDAR_TOP keyframe' % s['token'])
        rec = lidar_of[s['token']]
        tok = rec['token']
        if tok not in seg:
            raise ValueError('sample %s: no lidarseg entry for %s' % (s['token'], tok))
        name = scenes[s['scene_token']]
        rows['train' if name in train_names else 'val'].append([tok, rec['filename'], seg[tok], name])
    meta = {'version': version, 'nuscenes': nuscenes, 'lidarseg': lidarseg, 'order': 'sample.json',
            'split_rule': 'nuscenes.utils.splits.train scenes -> train, other trainval scenes -> val (Pointcept)',
            'counts': {k: len(v) for k, v in rows.items()},
            'scenes': {k: len({r[3] for r in v}) for k, v in rows.items()},
            'val_scenes_equal_devkit_val': sorted({r[3] for r in rows['val']}) == sorted(splits.val)}
    return dict(meta=meta, **rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nuscenes', required=True, help='detection tree root holding samples/ and v1.0-trainval/')
    ap.add_argument('--lidarseg', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = build(args.nuscenes, args.lidarseg)
    missing = [r[2] for k in ('train', 'val') for r in out[k]
               if not os.path.isfile(os.path.join(args.lidarseg, r[2]))]
    out['meta']['label_files_missing'] = len(missing)
    if os.path.exists(args.out):
        raise FileExistsError('%s exists; the index is versioned (write a new _vN file instead)' % args.out)
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(json.dumps(out['meta'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()
