#!/usr/bin/env bash
# One 2-GPU allocation for the per-domain-BN / BN-adapted-teacher pseudo-labelling rows
# (experiments_md/20261004_01). Run INSIDE an sbatch allocation (submit with --wrap="bash scripts/...").
#
#  1. DDP smoke test of centerpoint-st3d-dsbn-adabnteacher-lyft2nuscenes through the real 2-GPU launcher
#     (snapshot, torch.distributed, auto eval recovery): DSNorm student + teacher, the teacher's TARGET
#     statistics re-estimated before the pseudo-label pass, both domains through DSNorm under DDP, the
#     final checkpoint scored with the target set. 16-frame subset, 2 epochs: not a measurement.
#  3. (on the first GPU, after its pass) AdaBN eval arms on the nuScenes -> Waymo checkpoints.
#  2. Pseudo-label passes with the BN-ADAPTED teacher at score floor 0.0001 (no W&B), one per GPU in
#     parallel, for the label-free count-balance cuts of the S2 rows and the S1 label-quality diagnosis:
#       S1  ldb35c2o ep30 on nuScenes train
#       S2  2da6oz6e ep20 (26814, the headline's teacher) and wfdcq75s ep20 (single-sweep control) on KITTI train
#
# Do not edit while a job runs it (bash reads scripts incrementally).
set -u
cd /home/koyama/code/ST3D/tools
S=/storage/wandb
L=$S/run-20260924_111041-ldb35c2o/files/ckpt/checkpoint_epoch_30.pth
echo "=== 1. DDP smoke: per-domain BN + BN-adapted teacher ==="
WANDB_NOTES="${WANDB_NOTES:-Smoke}" bash scripts/run_sourceonly_2gpu.sh \
  cfgs/da-ieee-access/centerpoint-st3d-dsbn-adabnteacher-lyft2nuscenes.yaml smoke_dsbn smoke_dsbn_adabnteacher_lyft2nuscenes \
  --pretrained_model "$L" --pretrained_model_teacher "$L" \
  --epochs 2 --use_subset --num_epochs_to_eval 0 --workers 2
echo "=== smoke exit: $? ==="

echo "=== 2. pseudo-label passes with the BN-adapted teacher ==="
read -r -a G <<< "${CUDA_VISIBLE_DEVICES//,/ }"
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
gen() {  # gpu cfg ckpt out
  CUDA_VISIBLE_DEVICES=$1 singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
    python analysis/pseudo_label_threshold/generate_pseudo_labels.py \
      --cfg_file "$2" --teacher_ckpt "$3" --out_dir "$4" --thresh 0.0001 --teacher_adabn 1.0
}
N=$S/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth
A=$S/run-20260928_050622-exbb9chu/files/ckpt/checkpoint_epoch_20.pth
( gen "${G[0]}" cfgs/da-ieee-access/centerpoint-st3d-lyft2nuscenes.yaml "$L" \
    /storage/pseudo_labels/ldb35c2o_ep30_nuscenes_train_thr0.0001_adabn; echo "S1 gen exit $?"
  # 3. AdaBN on the nuScenes -> Waymo checkpoints (the arms 27289 lost to Waymo's missing INFO_PATH)
  for arm in "centerpoint-sourceonly-nuscenes2waymo $N target 1.0 adabn_NWctrl_target" \
             "centerpoint-sourceonly-nuscenes2waymo $N target 0.5 adabn_NWctrl_mix05" \
             "centerpoint-accum-nuscenes2waymo $A target 1.0 adabn_NWaccum_target"; do
    set -- $arm
    CUDA_VISIBLE_DEVICES=${G[0]} bash analysis/adabn_eval.sh cfgs/da-ieee-access/$1.yaml $2 $3 $4 $5
    echo "Waymo arm $5 exit $?"
  done ) &
( gen "${G[1]:-${G[0]}}" cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml \
    $S/run-20261001_022243-2da6oz6e/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/2da6oz6e_ep20_kitti_train_thr0.0001_adabn; echo "S2 26814 gen exit $?"
  gen "${G[1]:-${G[0]}}" cfgs/da-ieee-access/centerpoint-accum-st3d-nuscenes2kitti.yaml \
    $S/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth \
    /storage/pseudo_labels/wfdcq75s_ep20_kitti_train_thr0.0001_adabn; echo "S2 control gen exit $?" ) &
wait
echo "=== done ==="
