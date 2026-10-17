#!/usr/bin/env bash
# A3 cut passes, GPU part (experiments_md 20261017_06 §3; rules pre-declared in 20261004_01 §5e). Teacher 27807
# (scan-pattern thinning, W&B 87h61vfd, epoch 7) AS TRAINED (no AdaBN), score floor 0.0001, no W&B, ONE GPU (the Waymo
# pass must not run under 2-GPU DDP: memory/repo/index.md, Waymo NCCL timeout):
#   - nuScenes TRAIN through the A3 config's own target block            -> rule A (NEG_THRESH)
#   - Waymo VAL (SAMPLED_INTERVAL 5, 7,998 frames) through the flat config, RING_PATTERN.APPLY_AT_EVAL, i.e. the
#     teacher's own training input (TOP rows on the 1.33 deg lattice, 0.332 deg azimuth bins, side lidars dropped)
#                                                                         -> rule B (SCORE_THRESH)
# The rules, the cut application and the target dump run afterwards on CPU: scripts/_a3_cut_checks.sh.
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_a3_cuts_w2n_ringpattern.sh <27807 checkpoint_epoch_7.pth>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _a3_cuts_w2n_ringpattern.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-waymo2nuscenes.yaml
OUT=/storage/pseudo_labels/87h61vfd_ep7
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" \
  --out_dir ${OUT}_nuscenes_train_thr0.0001 --thresh 0.0001
echo "gen nuscenes train exit $?"
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file $P/cfgs/teacher_27807_on_waymo_val_int5_ringpattern.yaml \
  --teacher_ckpt "$CKPT" --out_dir ${OUT}_waymo_val_int5_ringpattern_thr0.0001 --thresh 0.0001
echo "gen waymo val exit $?"
ls -la ${OUT/\/storage/\/home\/koyama\/data}_nuscenes_train_thr0.0001/ps_label_e0.pkl \
       ${OUT/\/storage/\/home\/koyama\/data}_waymo_val_int5_ringpattern_thr0.0001/ps_label_e0.pkl
echo "=== done ==="
