#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 62 (design fixed 2026-11-27 12:53 JST, before any number): P_K of SEED with its BN
# statistics re-estimated on 1,000 strided KITTI train clouds over a subpopulation of each layer's elements (all = global
# AdaBN, the procedure check; far20; tbox, teacher boxes at 59b's rule-A cuts; gtbox, labelled reference), then full KITTI
# val. Usage (from ST3D/tools, inside a 1-GPU allocation): bash scripts/_item62_pop_20261127.sh SEED
# Each arm logs its exit status; the next runs regardless; exit 1 if any failed. Do not edit while a job runs it.
set -u
SEED=$1
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
EVAL_CFG=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml
CK=$(python3 analysis/item62_pop.py ckpt "$SEED")
F=0
for arm in all far20 tbox gtbox; do
  echo "=== item62 $arm (seed $SEED) ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 62 (T05 §5, design 2026-11-27 12:53 JST), labelled ANALYSIS, rescore only: P_K seed $SEED, BN statistics over the subpopulation $arm of each layer's elements (1000 strided KITTI train clouds), full KITTI val. Readings fixed in T05 item 62." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/item62_pop_adabn_eval.py --cfg_file "$EVAL_CFG" --ckpt "$CK" --arm "$arm" --seed "$SEED" \
        --out_dir "/home/koyama/data/item62/${arm}_s${SEED}" --run_name "i62_${arm}_s${SEED}" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
echo "=== item62 seed $SEED done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
