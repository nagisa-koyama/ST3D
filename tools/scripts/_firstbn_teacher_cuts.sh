#!/usr/bin/env bash
# Pseudo-label passes with FIRST-BN-ONLY adapted teachers (score floor 0.0001, no W&B), for the
# label-free count-balance cuts of the per-domain-BN pseudo-labelling rows (experiments_md/20261004_01
# §5b, §2.7). One GPU, sequential. Run inside an sbatch allocation: --wrap="bash scripts/_firstbn_teacher_cuts.sh"
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
S=/storage/wandb
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
FIRST='^backbone_3d\.conv_input\.1$'
gen() {  # cfg ckpt out
  singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python analysis/pseudo_label_threshold/generate_pseudo_labels.py \
      --cfg_file "$1" --teacher_ckpt "$2" --out_dir "$3" --thresh 0.0001 --teacher_adabn 1.0 --adabn_layers "$FIRST"
  echo "gen $3 exit $?"
}
gen cfgs/da-ieee-access/centerpoint-st3d-lyft2nuscenes.yaml $S/run-20260924_111041-ldb35c2o/files/ckpt/checkpoint_epoch_30.pth \
    /storage/pseudo_labels/ldb35c2o_ep30_nuscenes_train_thr0.0001_adabn_firstbn
gen cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml $S/run-20261001_022243-2da6oz6e/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/2da6oz6e_ep20_kitti_train_thr0.0001_adabn_firstbn
gen cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml $S/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/wfdcq75s_ep20_kitti_train_thr0.0001_adabn_firstbn
echo "=== done ==="
