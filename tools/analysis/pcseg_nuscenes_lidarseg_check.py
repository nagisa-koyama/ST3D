"""Check an extracted nuScenes-lidarseg tree against the detection tree's nuScenes (experiments_md episodes/20261116_01
§9, O1). The lidarseg tree is kept apart from the detection tree, because its v1.0-*/category.json would replace the
23-category file the detection pipeline reads.

For every split directory in the lidarseg tree that has a lidarseg.json:
  - every entry's label file exists, and every label file has an entry;
  - every sample_data token is a LIDAR_TOP keyframe of the detection tree's sample_data.json (trainval only);
  - for --n_points entries spread over the split, the label count equals the point count of the lidar .pcd.bin
    (5 float32 per point) and every label is a category index in category.json;
  - the category table (index -> name) is printed.

    python3 analysis/pcseg_nuscenes_lidarseg_check.py --lidarseg /home/koyama/data/nuscenes_lidarseg_v1_0 \
        --nuscenes /home/koyama/data/nuscenes_full_v_1_0/v1.0-trainval --out <json>
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lidarseg', required=True)
    ap.add_argument('--nuscenes', required=True, help='detection tree root holding samples/ and v1.0-trainval/')
    ap.add_argument('--n_points', type=int, default=400)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    report = {}
    for split in sorted(d for d in os.listdir(args.lidarseg) if d.startswith('v1.0-')):
        lj = os.path.join(args.lidarseg, split, 'lidarseg.json')
        if not os.path.isfile(lj):
            report[split] = {'lidarseg_json': False}
            continue
        entries = json.load(open(lj))
        cats = json.load(open(os.path.join(args.lidarseg, split, 'category.json')))
        index = {c['index']: c['name'] for c in cats}
        files = {e['filename'] for e in entries}
        on_disk = set()
        label_dir = os.path.join(args.lidarseg, 'lidarseg', split)
        if os.path.isdir(label_dir):
            on_disk = {os.path.join('lidarseg', split, f) for f in os.listdir(label_dir)}
        r = {'entries': len(entries), 'label_files_on_disk': len(on_disk),
             'entries_without_file': len(files - on_disk), 'files_without_entry': len(on_disk - files),
             'categories': {int(k): v for k, v in sorted(index.items())}}
        if split == 'v1.0-trainval':
            sd = json.load(open(os.path.join(args.nuscenes, 'v1.0-trainval', 'sample_data.json')))
            by_token = {s['token']: s for s in sd}
            missing = [e['sample_data_token'] for e in entries if e['sample_data_token'] not in by_token]
            not_key = [e['sample_data_token'] for e in entries if e['sample_data_token'] in by_token
                       and not (by_token[e['sample_data_token']]['is_key_frame']
                                and 'LIDAR_TOP' in by_token[e['sample_data_token']]['filename'])]
            r['tokens_missing_in_detection_tree'] = len(missing)
            r['tokens_not_lidar_top_keyframes'] = len(not_key)
            idx = np.linspace(0, len(entries) - 1, num=min(args.n_points, len(entries))).round().astype(int)
            bad_count, bad_label, n_pts, hist = [], [], 0, np.zeros(256, dtype=np.int64)
            for i in idx:
                e = entries[int(i)]
                lab = np.fromfile(os.path.join(args.lidarseg, e['filename']), dtype=np.uint8)
                pc = os.path.join(args.nuscenes, by_token[e['sample_data_token']]['filename'])
                npts = os.path.getsize(pc) // (5 * 4)
                if len(lab) != npts:
                    bad_count.append((e['filename'], len(lab), npts))
                if not set(np.unique(lab).tolist()) <= set(index):
                    bad_label.append(e['filename'])
                n_pts += len(lab)
                hist += np.bincount(lab, minlength=256)
            r.update({'checked_entries': len(idx), 'checked_points': int(n_pts), 'count_mismatches': bad_count[:20],
                      'n_count_mismatches': len(bad_count), 'files_with_unknown_index': len(bad_label),
                      'label_fraction': {index.get(k, str(k)): float(hist[k] / max(n_pts, 1))
                                         for k in np.flatnonzero(hist)}})
        report[split] = r
        print(split, {k: v for k, v in r.items() if k not in ('categories', 'label_fraction')})
    with open(args.out, 'w') as f:
        json.dump(report, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
