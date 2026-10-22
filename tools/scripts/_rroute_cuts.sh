#!/usr/bin/env bash
# R-ROUTE cut passes, GPU part (experiments_md 20261022_07 §2.1; rules pre-declared in 20261004_01 §5e). Teacher 27807
# (scan-pattern thinning, W&B 87h61vfd, epoch 7) ADAPTED: every BN layer's statistics re-estimated on 1,000 strided nuScenes
# TRAIN frames (MIX 1.0, eval-mode view, no labels), exactly what train.py's SELF_TRAIN.TEACHER_ADABN does for the R-ROUTE
# config. Score floor 0.0001, no W&B, ONE GPU (the Waymo pass must not run under 2-GPU DDP: memory/repo/index.md):
#   - nuScenes TRAIN through the R-ROUTE config's own target block; the config's TEACHER_ADABN adapts the teacher first
#                                                                              -> rule A (NEG_THRESH)
#   - Waymo VAL (SAMPLED_INTERVAL 5, 7,998 frames) through the flat config of 29316 (RING_PATTERN.APPLY_AT_EVAL), the
#     teacher adapted on the R-ROUTE config's nuScenes TRAIN (gen_adapted_teacher_pass.py --stats_cfg)
#                                                                              -> rule B (SCORE_THRESH)
# The rules, the cut application and the target dump run afterwards on CPU: scripts/_rroute_cut_checks.sh.
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_rroute_cuts.sh <27807 checkpoint_epoch_7.pth>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _rroute_cuts.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-dsbn-adabnteacher-waymo2nuscenes.yaml
OUT=/storage/pseudo_labels/87h61vfd_ep7_adabn
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" \
  --out_dir ${OUT}_nuscenes_train_thr0.0001 --thresh 0.0001
echo "gen nuscenes train (adapted teacher) exit $?"
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/gen_adapted_teacher_pass.py --cfg_file $P/cfgs/teacher_27807_on_waymo_val_int5_ringpattern.yaml \
  --stats_cfg "$ST_CFG" --teacher_ckpt "$CKPT" --out_dir ${OUT}_waymo_val_int5_ringpattern_thr0.0001 --thresh 0.0001
echo "gen waymo val (adapted teacher) exit $?"
ls -la ${OUT/\/storage/\/home\/koyama\/data}_nuscenes_train_thr0.0001/ps_label_e0.pkl \
       ${OUT/\/storage/\/home\/koyama\/data}_waymo_val_int5_ringpattern_thr0.0001/ps_label_e0.pkl
echo "=== done ==="
