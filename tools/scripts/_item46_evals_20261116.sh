#!/usr/bin/env bash
# Item 46 (experiments_md T05 §5): the two eval arms of one checkpoint for the three-way split of the
# pseudo-labelling gain. `none` scores the checkpoint as saved (the procedure check against its value on
# record); `source_all` re-estimates every BatchNorm on 1,000 SOURCE train frames, the per-platform blocks in
# proportion to size (per-domain BN: the TARGET set the model is scored with), i.e. the checkpoint under source
# statistics.
#
# usage (from ST3D/tools, inside a 1-GPU allocation):
#   bash scripts/_item46_evals_20261116.sh <cfg> <host ckpt path | job:<jobid>:<epoch>> <tag>
# job:<jobid>:<epoch> resolves the checkpoint from the training row's log when this runs (afterany chains).
# Do not edit while a job runs it (bash reads scripts incrementally).
set -eu
cd /home/koyama/code/ST3D/tools
CFG=${1:?cfg}
SPEC=${2:?ckpt or job:<id>:<epoch>}
TAG=${3:?tag}
case "$SPEC" in
  job:*) J=$(echo "$SPEC" | cut -d: -f2); E=$(echo "$SPEC" | cut -d: -f3); CKPT=$(bash analysis/ckpt_of_job.sh "$J" "$E") ;;
  *) CKPT=$SPEC ;;
esac
echo "item46 eval: $TAG on $CKPT ($CFG)"
rc=0
bash analysis/adabn_eval.sh "$CFG" "$CKPT" none 1.0 "${TAG}_none" --extra_tag "${TAG}_none" --eval_tag "${TAG}_none" || rc=1
bash analysis/adabn_eval.sh "$CFG" "$CKPT" source_all 1.0 "${TAG}_src" --extra_tag "${TAG}_src" --eval_tag "${TAG}_src" || rc=1
echo "item46 eval: $TAG done (rc $rc)"
exit $rc
