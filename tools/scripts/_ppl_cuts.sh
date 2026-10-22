#!/usr/bin/env bash
# P-PL cut passes, GPU part (experiments_md 20261022_07 §3.1; rules pre-declared in 20261004_01 §5e). Teacher 28034
# (PandaSet -> nuScenes scan-pattern thinning, W&B lf9c36yn, epoch 115) AS TRAINED, score floor 0.0001, no W&B, ONE GPU:
#   - nuScenes TRAIN through the P-PL config's own target block                      -> rule A (NEG_THRESH)
#   - PandaSet VAL (20 sequences, 1,600 frames) through the teacher's own RING_PATTERN thinning, GT in the loader's
#     frame, rule B computed in the same pass (rule_b_pandaset_val.py; see its docstring for why the Waymo tool chain
#     cannot serve PandaSet)                                                            -> rule B (SCORE_THRESH)
# Rule A, the cut application and the target dump run afterwards on CPU: scripts/_ppl_cut_checks.sh.
# PandaSet infos bake absolute /root/ST3D paths: the /root/ST3D bind is required (AGENTS.md).
# Usage (inside an sbatch allocation, 1 GPU):  bash scripts/_ppl_cuts.sh <28034 checkpoint_epoch_115.pth>
# Do not edit while a job runs it.
set -u
CKPT=${1:?usage: _ppl_cuts.sh <teacher checkpoint>}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
ST_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-st3d-pandaset2nuscenes.yaml
OUT=/storage/pseudo_labels/lf9c36yn_ep115
BIND="--bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D"
echo "=== live HEAD $(git rev-parse --short HEAD) ==="
singularity exec --nv $BIND "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" \
  --out_dir ${OUT}_nuscenes_train_thr0.0001 --thresh 0.0001
echo "gen nuscenes train exit $?"
mkdir -p ${OUT/\/storage/\/home\/koyama\/data}_pandaset_val_ringpattern
singularity exec --nv $BIND "$SIF" \
  python $P/rule_b_pandaset_val.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" \
  --out_dir ${OUT}_pandaset_val_ringpattern --check 20 \
  | tee ${OUT/\/storage/\/home\/koyama\/data}_pandaset_val_ringpattern/rule_b.txt
echo "rule B pandaset val exit ${PIPESTATUS[0]}"
ls -la ${OUT/\/storage/\/home\/koyama\/data}_nuscenes_train_thr0.0001/ps_label_e0.pkl \
       ${OUT/\/storage/\/home\/koyama\/data}_pandaset_val_ringpattern/rule_b_pass.pkl
echo "=== done ==="
