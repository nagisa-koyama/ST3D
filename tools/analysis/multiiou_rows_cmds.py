"""Commands for the IoU 0.7 / 0.5 rescoring of the paper's rows (experiments_md 20261013_02): one line per (row, class),
each an oracle_box_error_multiiou.py call with --arms base inside the container. Car at 0.7,0.5 for every row;
Pedestrian at 0.5,0.25 for the rows marked `ped`. The job script runs the lines with `xargs -P`.

    python3 analysis/multiiou_rows_cmds.py <manifest.json> <out_dir> > cmds.txt
"""
import json
import re
import sys

SIF = '/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif'


def main(manifest, out_dir):
    for k, r in enumerate(json.load(open(manifest))):
        x = r.get('extra', '')
        dev = re.search(r'device=(\d)', x)
        cone = re.search(r'cone=(\d)', x)
        classes = [('Car', '0.7,0.5')] + ([('Pedestrian', '0.5,0.25')] if re.search(r'(^|;)ped(;|$)', x) else [])
        for cls, th in classes:
            label = f"r{k:02d}_{r['setting']}_{r['job']}_{cls}"
            args = [f"--dataset {r['dataset']}", f'--cls {cls}', f'--thresholds {th}', '--arms base',
                    f"--result {r['result_dir']}/result.pkl", f'--label {label}', f'--out {out_dir}/{label}.jsonl']
            if r['dataset'] in ('nuscenes', 'waymo'):
                args.append(f"--cfg {r['cfg']}")
            if r['dataset'] == 'pandaset':
                args += [f'--device {dev.group(1)}', f'--cone {cone.group(1)}']
            print(f"singularity exec --bind /home/koyama/data/:/storage {SIF} python3 analysis/oracle_box_error_multiiou.py "
                  + ' '.join(args) + f' > {out_dir}/{label}.log 2>&1')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
