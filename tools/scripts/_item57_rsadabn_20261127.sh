#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 57 (design fixed 2026-11-27 00:10 JST, before any number): range-stratified AdaBN on item 48's
# forward pair. P_K (32283 / 32284, epoch 152) on full KITTI val, source statistics from S' train (P_K's training cut), target
# statistics from KITTI train, arms global (= AdaBN) and item57 (5 knots). Rows: analysis/item57_rs_adabn.py runs. Usage (from
# ST3D/tools, inside a 1-GPU allocation): bash scripts/_item57_rsadabn_20261127.sh. Each run logs its exit status; the next runs
# regardless; exit 1 if any failed. Do not edit while a job runs it. ANALYSIS: target labels score only.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
H="Item 57 (T05 §5 item 57, design fixed 2026-11-27 00:10 JST), labelled ANALYSIS, test-time only"
declare -A WHAT=([global]="global AdaBN through the RS-AdaBN code (one knot; must reproduce item 48's AdaBN row within 0.3)" \
                 [item57]="range-stratified AdaBN, knots 5/15/25/35/55 m, per-knot source offsets from S' train")
echo "=== item57 smoke (P_K 32283) ($(date '+%F %T')) ==="
C0=$(bash analysis/ckpt_of_job.sh 32283 152) || { echo "smoke: no checkpoint"; exit 2; }
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
  python3 analysis/rs_adabn_smoke.py --ckpt "$C0" < /dev/null || { echo "item57: smoke failed, nothing else run"; exit 1; }
F=0
runs=$(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item57_rs_adabn.py runs) || { echo "item57: no runs"; exit 2; }
while read -r seed job arm tag; do
  [ -n "$tag" ] || continue
  echo "=== item57 $tag (train job $job, seed $seed, arm $arm) ($(date '+%F %T')) ==="
  C=$(bash analysis/ckpt_of_job.sh "$job" 152) || { echo "exit 2 (no checkpoint)"; F=$((F+1)); continue; }
  echo "checkpoint $C"
  WANDB_NOTES="$H: P_K $job (seed $seed, epoch 152) on full KITTI val, ${WHAT[$arm]}; target statistics from 1000 strided KITTI train frames, no labels. Readings fixed in T05 item 57." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/rs_adabn_eval.py --cfg_file cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml \
        --src_cfg cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml \
        --ckpt "$C" --knots "$arm" --extra_tag "$tag" --eval_tag "$tag" --run_name "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$runs"
echo "=== item57 done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
