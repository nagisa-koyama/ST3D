#!/usr/bin/env bash
# Legal chain STEP 2 for Waymo -> nuScenes pseudo-labelling (experiments_md/20261005_01 §14; rules pre-declared in
# 20261004_01 §5e). Teacher 27599 (per-bin global thinning, W&B puo3m3qt) AS TRAINED (no AdaBN), score floor 0.0001,
# no W&B:
#   - nuScenes TRAIN, through the pseudo-labelling config's own target block -> rule A (background plateau's upper
#     half-maximum) -> NEG_THRESH
#   - Waymo VAL (the labelled SOURCE val, SAMPLED_INTERVAL 5 = 7,998 frames) on the teacher's TRAINING input: the
#     thinning measured as train.py measures it for the config, installed on the val clouds (--calib_from_cfg) ->
#     rule B (source precision, all classes; Vehicle -> Car) -> SCORE_THRESH
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_legal_chain_step2_w2n.sh <27599 checkpoint_epoch_7.pth>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _legal_chain_step2_w2n.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-global-st3d-waymo2nuscenes.yaml
OUT=/storage/pseudo_labels/puo3m3qt_ep7
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" \
  --out_dir ${OUT}_nuscenes_train_thr0.0001 --thresh 0.0001
echo "gen nuscenes train exit $?"
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file $P/cfgs/teacher_27599_on_waymo_val_int5_thinned.yaml \
  --teacher_ckpt "$CKPT" --out_dir ${OUT}_waymo_val_int5_thinned_thr0.0001 --thresh 0.0001 --calib_from_cfg "$ST_CFG"
echo "gen waymo val exit $?"
echo "=== rule B: SCORE_THRESH (source-val precision on Waymo val, all classes) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" bash -c "cd $P && python source_precision_cut.py \
  ${OUT}_waymo_val_int5_thinned_thr0.0001/ps_label_e0.pkl --dataset waymo \
  --infos /home/koyama/code/ST3D/data/waymo/waymo_infos_val.pkl --classes Car Pedestrian Cyclist"
echo "=== rule A: NEG_THRESH (background plateau, nuScenes train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py \
  ${OUT}_nuscenes_train_thr0.0001/ps_label_e0.pkl
echo "=== done ==="
