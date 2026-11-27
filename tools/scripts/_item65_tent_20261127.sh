#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 65 (design fixed 2026-11-27 13:02 JST, before any number): TENT-style gamma / beta of P_K
# (SEED) on unlabelled KITTI testing clouds (analysis/item65_tent.py), the bitwise check, then each saved checkpoint (epochs 1
# and 10) rescored under global AdaBN as 59a. Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item65_tent_20261127.sh SEED [extra item65_tent.py args, smoke only]
# Do not edit while a job runs it.
set -u
SEED=$1; shift
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
SRC_CFG=cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml
EVAL_CFG=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml
RUN="singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $SIF"
PK=$($RUN python3 analysis/item65_tent.py pk "$SEED" | tail -1)
echo "=== item65 TENT, seed $SEED ($(date '+%F %T')) ==="
WANDB_NOTES="Item 65 (T05 §5, design 2026-11-27 13:02 JST), ANALYSIS: TENT-style BN gamma / beta of P_K seed $SEED on KITTI testing clouds (no labels), top-500 heatmap entropy, Adam 0.001, 10 epochs" \
  $RUN python3 analysis/item65_tent.py --ckpt "$PK" --seed "$SEED" --out_dir "/home/koyama/data/item65/s$SEED" \
    --run_name "i65_tent_s$SEED" "$@" < /dev/null || { echo "item65: TENT failed"; exit 2; }
$RUN python3 analysis/item65_tent.py check "$SEED" || { echo "item65: bitwise check FAILED, not rescored"; exit 2; }
F=0
for e in 1 10; do
  ck=$($RUN python3 analysis/item65_tent.py ckpt "$SEED" "$e" | tail -1)
  [ -f "$ck" ] || { echo "exit 2 (no checkpoint $ck)"; F=$((F+1)); continue; }
  t=i65_e${e}_s${SEED}
  echo "=== item65 rescore $t ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 65 rescore: TENT checkpoint epoch $e seed $SEED under global AdaBN (1000 strided KITTI train frames), full KITTI val" \
    $RUN python3 analysis/rs_adabn_eval.py --cfg_file "$EVAL_CFG" --src_cfg "$SRC_CFG" --ckpt "$ck" --knots global \
      --extra_tag "$t" --eval_tag "$t" --run_name "$t" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
echo "=== item65 seed $SEED done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
