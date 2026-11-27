#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 65′ (design fixed 2026-11-27 15:47 JST, before any number): item 65's TENT-style gamma /
# beta of P_K (SEED) on unlabelled KITTI testing clouds with the loss H + KL (per-class top-500 entropy plus SHOT's prior term,
# the marginal held at its start; analysis/item65_tent.py --variant prior), the bitwise check, then each saved checkpoint
# (epochs 1 and 10) rescored under global AdaBN as 59a. Exit 3 from training = the step-50 guard held the row.
# Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item65p_tent_20261127.sh SEED [extra item65_tent.py args, smoke only]
# Do not edit while a job runs it.
set -u
SEED=$1; shift
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
SRC_CFG=cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml
EVAL_CFG=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml
RUN="singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $SIF"
PK=$($RUN python3 analysis/item65_tent.py pk "$SEED" | tail -1)
echo "=== item65p TENT + prior, seed $SEED ($(date '+%F %T')) ==="
WANDB_NOTES="Item 65′ (T05 §5, design 2026-11-27 15:47 JST), ANALYSIS: TENT-style BN gamma / beta of P_K seed $SEED on KITTI testing clouds (no labels), loss = per-class top-500 entropy + SHOT prior KL (marginal held at its start), Adam 0.001, 10 epochs" \
  $RUN python3 analysis/item65_tent.py --variant prior --ckpt "$PK" --seed "$SEED" --out_dir "/home/koyama/data/item65p/s$SEED" \
    --run_name "i65p_tent_s$SEED" "$@" < /dev/null
s=$?
[ $s -eq 3 ] && { echo "item65p: HELD by the step-50 guard (a class below 0.5 of its prior)"; exit 3; }
[ $s -eq 0 ] || { echo "item65p: TENT failed (exit $s)"; exit 2; }
$RUN python3 analysis/item65_tent.py check "$SEED" --variant prior || { echo "item65p: bitwise check FAILED, not rescored"; exit 2; }
F=0
for e in 1 10; do
  ck=$($RUN python3 analysis/item65_tent.py ckpt "$SEED" "$e" --variant prior | tail -1)
  [ -f "$ck" ] || { echo "exit 2 (no checkpoint $ck)"; F=$((F+1)); continue; }
  t=i65p_e${e}_s${SEED}
  echo "=== item65p rescore $t ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 65′ rescore: TENT + prior checkpoint epoch $e seed $SEED under global AdaBN (1000 strided KITTI train frames), full KITTI val" \
    $RUN python3 analysis/rs_adabn_eval.py --cfg_file "$EVAL_CFG" --src_cfg "$SRC_CFG" --ckpt "$ck" --knots global \
      --extra_tag "$t" --eval_tag "$t" --run_name "$t" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
echo "=== item65p seed $SEED done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
