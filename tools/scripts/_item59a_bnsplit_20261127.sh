#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 59a (design fixed 2026-11-27 08:15 JST, before any number): where item 58's G correction
# sits. For one seed: item 57's smoke, then four global-AdaBN rescoring runs on full KITTI val, which differ only in gamma / beta:
# rs1 (P_K's own, repeats item 57's RS-1), aff_all / aff_3d / aff_2d (G's in all / backbone_3d / backbone_2d + dense_head
# layers; hybrids from analysis/item59_bn_split.py make). Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item59a_bnsplit_20261127.sh SEED
# Each run logs its exit status; the next runs regardless; exit 1 if any failed. Do not edit while a job runs it.
# ANALYSIS: G's gamma / beta were fit with KITTI train labels; target labels also score.
set -u
SEED=$1
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
H="Item 59a (T05 §5 item 59, design fixed 2026-11-27 08:15 JST), labelled ANALYSIS, rescore only"
SRC_CFG=cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml
EVAL_CFG=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml
declare -A WHAT=([rs1]="P_K's own gamma / beta (repeats item 57's RS-1)" \
                 [aff_all]="G's gamma / beta in all 34 BN layers (item 58 G checkpoint, statistics replaced)" \
                 [aff_3d]="G's gamma / beta in the 21 backbone_3d BN layers, P_K's elsewhere" \
                 [aff_2d]="G's gamma / beta in the 13 backbone_2d + dense_head BN layers, P_K's elsewhere")
echo "=== item59a hybrids, seed $SEED ($(date '+%F %T')) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item59_bn_split.py make "$SEED" || {
  echo "item59a: hybrids failed, nothing run"; exit 2; }
runs=$(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item59_bn_split.py runs "$SEED") || {
  echo "item59a: no runs"; exit 2; }
C0=$(echo "$runs" | awk '$1=="rs1"{print $2}')
echo "=== item59a smoke (P_K seed $SEED) ($(date '+%F %T')) ==="
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
  python3 analysis/rs_adabn_smoke.py --ckpt "$C0" < /dev/null || { echo "item59a: smoke failed, nothing else run"; exit 1; }
F=0
while read -r arm ck tag; do
  [ -n "$tag" ] || continue
  echo "=== item59a $tag (arm $arm, seed $SEED) ($(date '+%F %T')) ==="
  [ -f "$ck" ] || { echo "exit 2 (no checkpoint $ck)"; F=$((F+1)); continue; }
  echo "checkpoint $ck"
  WANDB_NOTES="$H: seed $SEED, ${WHAT[$arm]}; global AdaBN from 1000 strided KITTI train frames (no labels), full KITTI val. Readings fixed in T05 item 59." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/rs_adabn_eval.py --cfg_file "$EVAL_CFG" --src_cfg "$SRC_CFG" \
        --ckpt "$ck" --knots global --extra_tag "$tag" --eval_tag "$tag" --run_name "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$runs"
echo "=== item59a seed $SEED done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
