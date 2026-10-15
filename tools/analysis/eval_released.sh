#!/usr/bin/env bash
# Score a sanitized released checkpoint on KITTI val (experiments_md 20261015_03). Same container call as
# eval_checkpoint.sh; an optional 4th argument s != 1 runs DALI's test-time scale (analysis/released_test_scaled.py).
#   bash analysis/eval_released.sh <cfg> <sanitized.pth (HOST path)> <eval_tag> [scale]
set -u
CFG=${1:?cfg}; CKPT=${2:?ckpt}; TAG=${3:?tag}; SCALE=${4:-1}
cd /home/koyama/code/ST3D/tools
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }
[ -f "$CKPT" ] || { echo "no such checkpoint: $CKPT" >&2; exit 2; }
case "$CKPT" in *_sanitized.pth) ;; *) echo "refusing a checkpoint that was not sanitized: $CKPT" >&2; exit 2;; esac
if [ "$SCALE" = "1" ]; then ENTRY=test.py; ENVV=; else ENTRY=analysis/released_test_scaled.py; ENVV="EVAL_POINT_SCALE=$SCALE"; fi
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  env $ENVV python3 $ENTRY --cfg_file "$CFG" --ckpt "$CKPT" --batch_size 6 \
    --extra_tag released --eval_tag "$TAG"
