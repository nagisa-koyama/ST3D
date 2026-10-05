#!/usr/bin/env bash
# S2 teachers AS TRAINED (no AdaBN) on the labelled SOURCE val (nuScenes val), score floor 0.0001, for the
# pre-declared Pedestrian / Cyclist cut rule (experiments_md/20261004_01 §5c, source_precision_cut.py).
# One GPU, sequential. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
S=/storage/wandb
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
gen() {  # cfg ckpt out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python $P/generate_pseudo_labels.py --cfg_file "$1" --teacher_ckpt "$2" --out_dir "$3" --thresh 0.0001
  echo "gen $3 exit $?"
}
gen $P/cfgs/teacher_s2_on_nuscenes_val_15sweeps.yaml $S/run-20261001_022243-2da6oz6e/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/2da6oz6e_ep20_nuscenes_val_15sweeps_thr0.0001
gen $P/cfgs/teacher_s2_on_nuscenes_val_1sweep.yaml $S/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/wfdcq75s_ep20_nuscenes_val_1sweep_thr0.0001
echo "=== done ==="
