#!/usr/bin/env bash
# Test-time BatchNorm re-estimation (AdaBN) + evaluation of one checkpoint, on ONE GPU, no training.
#
#   bash analysis/adabn_eval.sh <cfg> <ckpt.pth> <none|target|source> [mix] [run_name] [extra args...]
#
# See analysis/adabn_eval.py for what each statistics mode means. Single GPU and --launcher none for
# the reason eval_checkpoint.sh gives (forked workers never re-import the live checkout). Do not edit
# this file while a job runs it: bash reads scripts incrementally.
set -u
CFG=${1:?usage: adabn_eval.sh <cfg> <ckpt> <none|target|source> [mix] [run_name] [extra...]}
CKPT=${2:?ckpt}
STATS=${3:?stats mode}
MIX=${4:-1.0}
RUN=${5:-adabn_${STATS}_$(basename "$CFG" .yaml)}
shift $(( $# < 5 ? $# : 5 ))
cd /home/koyama/code/ST3D/tools
[ -f "$CFG" ] || { echo "no such config: $CFG" >&2; exit 2; }
[ -f "${CKPT/\/storage/\/home/koyama/data}" ] || { echo "no such checkpoint: $CKPT" >&2; exit 2; }

singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D \
  /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif \
  python3 analysis/adabn_eval.py --cfg_file "$CFG" --ckpt "$CKPT" --stats "$STATS" --mix "$MIX" \
    --run_name "$RUN" "$@"
