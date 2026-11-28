#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 70 (design fixed 2026-11-28 20:10 JST, before any number): item 57's range-stratified AdaBN
# on the real pair nuScenes -> KITTI. Item 41a's control row of one seed (epoch 20) on KITTI val; source statistics from
# nuScenes train (the control's DATA_CONFIG), target statistics from KITTI train; arms global (= AdaBN), item57, item57_3d,
# item57_2d. Rows: analysis/item70_rs_adabn_real.py runs SEED. Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item70_rsadabn_real_20261128.sh <seed>
# A GPU smoke on that seed's checkpoint runs first; if it fails nothing else runs. Each arm logs its exit status; the next
# runs regardless; exit 1 if any failed. Do not edit while a job runs it. ANALYSIS: target labels score only.
set -u
cd /home/koyama/code/ST3D/tools
S=${1:?usage: _item70_rsadabn_real_20261128.sh <seed>}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml
H="Item 70 (T05 §5 item 70, design fixed 2026-11-28 20:10 JST), labelled ANALYSIS, test-time only"
declare -A WHAT=([global]="global AdaBN through the RS-AdaBN code (one knot; must lie within 0.3 of 41a's B1)" \
                 [item57]="range-stratified AdaBN, knots 5/15/25/35/55 m in all 34 BN layers" \
                 [item57_3d]="range-stratified AdaBN in the 21 backbone_3d layers only, global AdaBN elsewhere" \
                 [item57_2d]="range-stratified AdaBN in the 13 backbone_2d + dense_head layers only, global AdaBN elsewhere")
runs=$(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item70_rs_adabn_real.py runs "$S") \
  || { echo "item70: no runs"; exit 2; }
J=$(echo "$runs" | head -1 | cut -d' ' -f2)
C=$(bash analysis/ckpt_of_job.sh "$J" 20) || { echo "item70: no checkpoint for $J"; exit 2; }
echo "=== item70 smoke (41a $J, seed $S, $C) ($(date '+%F %T')) ==="
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
  python3 analysis/rs_adabn_smoke.py --ckpt "$C" --cfg_file "$CFG" --src_cfg "$CFG" --src_block DATA_CONFIG < /dev/null \
  || { echo "item70: smoke failed, nothing else run"; exit 1; }
F=0
while read -r seed job arm tag; do
  [ -n "$tag" ] || continue
  echo "=== item70 $tag (41a $job, seed $seed, arm $arm) ($(date '+%F %T')) ==="
  WANDB_NOTES="$H: 41a control $job (seed $seed, epoch 20) on KITTI val, ${WHAT[$arm]}; source statistics from 1000 strided nuScenes train frames, target statistics from 1000 strided KITTI train frames, no labels. Readings fixed in T05 item 70." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/rs_adabn_eval.py --cfg_file "$CFG" --src_cfg "$CFG" --src_block DATA_CONFIG \
        --ckpt "$C" --knots "$arm" --extra_tag "$tag" --eval_tag "$tag" --run_name "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$runs"
echo "=== item70 seed $S done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
