"""Plan of the full LiT translation (experiments_md 20261015_02 §7). Plain python + numpy, no LiT import.

    python3 -I lit_port/lit_plan.py --list <ImageSets/train.txt> --data <waymo_processed_data> --out <plan.json> [--interval 2]

Reproduces `WaymoDataset.include_waymo_data` for the TRAIN split: per-sequence info pickles `<data>/<seq>/<seq>.pkl` are
concatenated in train.txt order (a sequence whose pickle is missing is skipped, as there), and SAMPLED_INTERVAL keeps every
`interval`-th entry of that GLOBAL list. So which frames of a sequence the detector sees depends on the lengths of the
sequences before it; the plan records, per sequence, its frame count and the sample_idx values that are sampled. Only these
are ray cast (the reconstructions use all frames). No label of any target split is read.
"""
import argparse
import json
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--interval', type=int, default=2)
    a = ap.parse_args()
    names = [os.path.splitext(x.strip())[0] for x in open(a.list).readlines()]
    seqs, skipped, g = [], [], 0
    for n in names:
        p = os.path.join(a.data, n, n + '.pkl')
        if not os.path.exists(p):
            skipped.append(n)
            continue
        with open(p, 'rb') as f:
            infos = pickle.load(f)
        idx = [i['point_cloud']['sample_idx'] for i in infos]
        assert idx == list(range(len(infos))), f'{n}: sample_idx is not 0..N-1'
        assert all(i['point_cloud']['lidar_sequence'] == n for i in infos), f'{n}: lidar_sequence mismatch'
        sampled = [k for k in range(len(infos)) if (g + k) % a.interval == 0]
        seqs.append({'name': n, 'n_frames': len(infos), 'sampled': sampled})
        g += len(infos)
    plan = {'interval': a.interval, 'n_listed': len(names), 'skipped': skipped, 'total_frames': g,
            'total_sampled': sum(len(s['sampled']) for s in seqs), 'sequences': seqs}
    with open(a.out, 'w') as f:
        json.dump(plan, f)
    print({k: v for k, v in plan.items() if k != 'sequences'}, 'n_sequences', len(seqs))


if __name__ == '__main__':
    main()
