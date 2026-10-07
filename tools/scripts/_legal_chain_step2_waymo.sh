#!/usr/bin/env bash
# Legal chain STEP 2 with WAYMO as the target (experiments_md/20261007_03; rules pre-declared in 20261004_01 §5e):
# the teacher's pseudo-labels on WAYMO TRAIN (through the pseudo-labelling config's own target block, so the same
# SAMPLED_INTERVAL frames the run will use), score floor 0.0001, no W&B, then rule A (NEG_THRESH). Rule B (SCORE_THRESH)
# reads the labelled SOURCE val and does not depend on the target; for teacher 27471 it is 27608's output.
# Usage (inside an sbatch allocation, 1 GPU):
#   bash scripts/_legal_chain_step2_waymo.sh <ckpt> <st_cfg> <out dir under /storage>
# Do not edit while a job runs it.
set -u
CKPT=${1:?ckpt}; ST_CFG=${2:?st_cfg}; OUT=${3:?out_dir}
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
P=analysis/pseudo_label_threshold
singularity exec --nv --bind /home/koyama/data/:/storage "$SIF" \
  python $P/generate_pseudo_labels.py --cfg_file "$ST_CFG" --teacher_ckpt "$CKPT" --out_dir "$OUT" --thresh 0.0001
echo "gen $OUT exit $?"
echo "=== rule A: NEG_THRESH (background plateau, Waymo train) ==="
singularity exec --bind /home/koyama/data/:/storage "$SIF" python $P/plateau_neg_thresh.py "$OUT/ps_label_e0.pkl"
echo "=== done ==="
