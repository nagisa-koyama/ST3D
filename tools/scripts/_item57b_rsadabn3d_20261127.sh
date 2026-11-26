#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 57b (design fixed 2026-11-27 01:02 JST, before any number of the variant): RS-AdaBN in the
# 3D backbone only (5 knots on the 21 backbone_3d BN layers, global AdaBN on the rest), P_K 32283 / 32284 on full KITTI val,
# source S' train, target KITTI train. Item 57's smoke runs first again (the code changed: per-layer knots) and must pass.
# Usage (from ST3D/tools, inside a 1-GPU allocation): bash scripts/_item57b_rsadabn3d_20261127.sh. Do not edit while a job runs
# it. ANALYSIS: target labels score only (chosen after item 57's results).
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
H="Item 57b (T05 §5 item 57b, design fixed 2026-11-27 01:02 JST), labelled ANALYSIS, test-time only"
echo "=== item57b smoke (P_K 32283) ($(date '+%F %T')) ==="
C0=$(bash analysis/ckpt_of_job.sh 32283 152) || { echo "smoke: no checkpoint"; exit 2; }
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
  python3 analysis/rs_adabn_smoke.py --ckpt "$C0" < /dev/null || { echo "item57b: smoke failed, nothing else run"; exit 1; }
F=0
runs=$(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/item57_rs_adabn.py runs item57_3d) || { echo "item57b: no runs"; exit 2; }
while read -r seed job arm tag; do
  [ -n "$tag" ] || continue
  echo "=== item57b $tag (train job $job, seed $seed, arm $arm) ($(date '+%F %T')) ==="
  C=$(bash analysis/ckpt_of_job.sh "$job" 152) || { echo "exit 2 (no checkpoint)"; F=$((F+1)); continue; }
  echo "checkpoint $C"
  WANDB_NOTES="$H: P_K $job (seed $seed, epoch 152) on full KITTI val, RS-AdaBN with 5 knots (5/15/25/35/55 m) in backbone_3d only, global AdaBN elsewhere; source offsets from S' train, target statistics from 1000 strided KITTI train frames, no labels. Readings fixed in T05 item 57b." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/rs_adabn_eval.py --cfg_file cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml \
        --src_cfg cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml \
        --ckpt "$C" --knots "$arm" --extra_tag "$tag" --eval_tag "$tag" --run_name "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done <<< "$runs"
echo "=== item57b done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
