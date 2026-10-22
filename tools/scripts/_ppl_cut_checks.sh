#!/usr/bin/env bash
# P-PL cut passes, CPU part (experiments_md 20261022_07 §3.1), after scripts/_ppl_cuts.sh:
#   1. rule B: already computed in the GPU pass (rule_b_pandaset_val.py); its `cuts: {...}` line is copied here
#   2. rule A: plateau_neg_thresh.py on the nuScenes TRAIN pass
#   3. apply_pseudo_label_cuts.py: rule A step 6, the cut values as the config will carry them, the per-frame first-pass
#      label counts the row should log, and the cut label file
#   4. target_dump.py on 20 nuScenes TRAIN frames with the cut labels, through the P-PL config (bug-prevention rule 4)
#   5. generate.log / rule_b.log: teacher weights loaded, nothing "Not updated", the pre-check line, frames written
# Outputs on NFS: /home/koyama/data/pseudo_labels/lf9c36yn_ep115_cuts/. No GPU. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml
H=/home/koyama/data/pseudo_labels/lf9c36yn_ep115
S=/storage/pseudo_labels/lf9c36yn_ep115
mkdir -p ${H}_cuts
RUN="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $SIF"
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
echo "=== 1. rule B: SCORE_THRESH (teacher's precision on PandaSet val through the scan pattern; from the GPU pass) ==="
cp ${H}_pandaset_val_ringpattern/rule_b.txt ${H}_cuts/rule_b.txt && cat ${H}_cuts/rule_b.txt
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
{
  d=${H}_nuscenes_train_thr0.0001
  echo "$d: $(/usr/bin/grep -c 'Not updated' $d/generate.log) 'Not updated' lines; $(/usr/bin/grep -o '==> Done (loaded [0-9/]*)' $d/generate.log | head -1)"
  /usr/bin/grep -h "wrote .*ps_label_e0.pkl" $d/generate.log
  d=${H}_pandaset_val_ringpattern
  echo "$d: $(/usr/bin/grep -c 'Not updated' $d/rule_b.log) 'Not updated' lines; $(/usr/bin/grep -o '==> Done (loaded [0-9/]*)' $d/rule_b.log | head -1)"
  /usr/bin/grep -h "pre-check\|wrote .* frames\|rank cap" $d/rule_b.log
} | tee ${H}_cuts/teacher_load.txt
echo "=== done ==="
