#!/usr/bin/env bash
# P-PL rule-B pass (experiments_md 20261022_07 §3.1, §9): teacher 28034 ep115 (lf9c36yn) AS TRAINED on PandaSet val
# through its own RING_PATTERN thinning, rule B in the pass (rule_b_pandaset_val.py). Replaces the --wrap of job 30220,
# which died on a tuple key turned into a list by pin_memory and whose dash shell could not read PIPESTATUS.
#   1. smoke: the same command on 2 batches into a scratch dir (fails fast on any code-path error after the model pass);
#   2. the full pass into lf9c36yn_ep115_pandaset_val_ringpattern/ (rule_b.txt, rule_b.log, rule_b_pass.pkl).
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_ppl_rule_b.sh <28034 checkpoint_epoch_115.pth>
# Do not edit while a job runs it.
set -u -o pipefail
CKPT=${1:?usage: _ppl_rule_b.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml
BIND="--bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D"
H=/home/koyama/data/pseudo_labels/lf9c36yn_ep115_pandaset_val_ringpattern
S=/storage/pseudo_labels/lf9c36yn_ep115_pandaset_val_ringpattern
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
mkdir -p ${H}_smoke $H
echo "=== 1. smoke, 2 batches ==="
singularity exec --nv $BIND "$SIF" python analysis/pseudo_label_threshold/rule_b_pandaset_val.py \
  --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" --out_dir ${S}_smoke --check 4 --max_batches 2 | tee ${H}_smoke/rule_b.txt
rc=$?; echo "smoke exit $rc"; [ $rc -eq 0 ] || exit $rc
echo "=== 2. full pass ==="
singularity exec --nv $BIND "$SIF" python analysis/pseudo_label_threshold/rule_b_pandaset_val.py \
  --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" --out_dir $S --check 20 | tee $H/rule_b.txt
rc=$?; echo "rule B pandaset val exit $rc"; [ $rc -eq 0 ] || exit $rc
ls -la $H/rule_b_pass.pkl
echo "=== done ==="
