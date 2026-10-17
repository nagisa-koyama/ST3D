#!/usr/bin/env bash
# A3 cut passes, CPU part (experiments_md 20261017_06 §3, §5), after scripts/_a3_cuts_w2n_ringpattern.sh:
#   1. rule B: source_precision_cut.py on the Waymo val pass (Vehicle -> Car; GT >= 1 point by num_points_in_gt)
#   2. rule A: plateau_neg_thresh.py on the nuScenes TRAIN pass
#   3. apply_pseudo_label_cuts.py: rule A step 6, the cut values as the config will carry them, the per-frame first-pass
#      label counts the row should log, and the cut label file
#   4. target_dump.py on 20 nuScenes TRAIN frames with the cut labels (bug-prevention checklist rule 4)
#   5. generate.log of both passes: teacher weights loaded, nothing "Not updated"
# Outputs on NFS: /home/koyama/data/pseudo_labels/87h61vfd_ep7_cuts/. No GPU. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-waymo2nuscenes.yaml
H=/home/koyama/data/pseudo_labels/87h61vfd_ep7
S=/storage/pseudo_labels/87h61vfd_ep7
mkdir -p ${H}_cuts
RUN="singularity exec --bind /home/koyama/data/:/storage $SIF"
echo "=== 1. rule B: SCORE_THRESH (source-val precision on Waymo val through the scan pattern) ==="
$RUN bash -c "cd $P && python source_precision_cut.py ${S}_waymo_val_int5_ringpattern_thr0.0001/ps_label_e0.pkl \
  --dataset waymo --infos /home/koyama/code/ST3D/data/waymo/waymo_infos_val.pkl --classes Car Pedestrian Cyclist" \
  | tee ${H}_cuts/rule_b.txt
echo "=== 2. rule A: NEG_THRESH (background plateau, nuScenes train) ==="
$RUN python $P/plateau_neg_thresh.py ${S}_nuscenes_train_thr0.0001/ps_label_e0.pkl | tee ${H}_cuts/rule_a.txt
echo "=== 3. cuts applied to the nuScenes TRAIN pass ==="
$RUN python $P/apply_pseudo_label_cuts.py ${S}_nuscenes_train_thr0.0001/ps_label_e0.pkl \
  --rule_b_log ${S}_cuts/rule_b.txt --rule_a_log ${S}_cuts/rule_a.txt --out ${S}_cuts/ps_label_e0_cut.pkl \
  | tee ${H}_cuts/apply.txt
echo "=== 4. target dump, 20 nuScenes TRAIN frames, cut labels ==="
$RUN python analysis/target_dump.py "$ST_CFG" TAR 20 ${S}_cuts/ps_label_e0_cut.pkl 2>&1 \
  | /usr/bin/grep -v " is loaded" | tee ${H}_cuts/target_dump.txt
echo "=== 5. teacher load in both passes ==="
for d in ${H}_nuscenes_train_thr0.0001 ${H}_waymo_val_int5_ringpattern_thr0.0001; do
  echo "$d: $(/usr/bin/grep -c 'Not updated' $d/generate.log) 'Not updated' lines; $(/usr/bin/grep -o '==> Done (loaded [0-9/]*)' $d/generate.log | head -1)"
  /usr/bin/grep -h "wrote .*ps_label_e0.pkl" $d/generate.log
done | tee ${H}_cuts/teacher_load.txt
echo "=== done ==="
