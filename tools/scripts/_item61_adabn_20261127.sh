#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 61 (design fixed 2026-11-27 12:53 JST, before any number): global-AdaBN rescoring of
# trained checkpoints, exactly as item 59a (rs_adabn_eval --knots global, 1,000 strided KITTI train clouds, P_K's eval
# config, full KITTI val). Usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item61_adabn_20261127.sh TAG JOB EPOCH [TAG JOB EPOCH ...]
# Each run logs its exit status; the next runs regardless; exit 1 if any failed. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
SRC_CFG=cfgs/da-ieee-access-analysis/item47/centerpoint-item47-kitti-p-on-nuscenes-lattice.yaml
EVAL_CFG=cfgs/da-ieee-access-analysis/factorswap-s2/centerpoint-factorswap-s2-p-pattern.yaml
F=0
while [ $# -ge 3 ]; do
  tag=$1; job=$2; ep=$3; shift 3
  echo "=== item61 $tag (job $job epoch $ep) ($(date '+%F %T')) ==="
  ck=$(bash analysis/ckpt_of_job.sh "$job" "$ep") || { echo "exit 2 (no checkpoint for job $job epoch $ep)"; F=$((F+1)); continue; }
  echo "checkpoint $ck"
  WANDB_NOTES="Item 61 (T05 §5, design 2026-11-27 12:53 JST), labelled ANALYSIS, rescore only: $tag = job $job epoch $ep under global AdaBN from 1000 strided KITTI train frames (no labels), full KITTI val. Readings fixed in T05 item 61." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/rs_adabn_eval.py --cfg_file "$EVAL_CFG" --src_cfg "$SRC_CFG" \
        --ckpt "$ck" --knots global --extra_tag "$tag" --eval_tag "$tag" --run_name "$tag" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
echo "=== item61 done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
