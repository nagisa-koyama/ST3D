#!/usr/bin/env bash
# ENV E1 second seed (experiments_md 20261024_02 §9; design fixed in T05 §5, sub-entry "User 10-24 (received between 03:22
# and 03:25)", experiments_md 8d96563): E1 rerun on the seed-667 quartet, then the two-seed readings.
# CPU-only Slurm job from ST3D/tools, chained afterok on both successors (scripts/_env_seed667_after_row.sh), which leave
# /home/koyama/data/env_splits/seed667/{bb,ss}.env with the prediction paths and the logged values.
#
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --dependency=afterok:<bbx>:<ssx> --kill-on-invalid-dep=yes \
#     --partition=a100,a6000_ada --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=envE1s667 \
#     --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt analysis/env_e1_seed667_job.sh
#
# The design runs the E1 scripts at ST3D 935dcb1 UNCHANGED: the job refuses to start if scene_error_decomposition.py,
# kitti_frame_decomposition.py or kitti_eval_cpu.py differ from 935dcb1 in the live tree, and logs whether the nuScenes
# dataset or the KITTI evaluator changed since (the sanity gate then guards the AP itself).
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/env_splits
C=cfgs/da-ieee-access
echo "=== ENV E1 seed 667 at live ST3D $(git -C /home/koyama/code/ST3D log -1 --format=%h) on $(hostname), $(date '+%Y-%m-%d %H:%M %Z') ==="
PINNED="tools/analysis/scene_error_decomposition.py tools/analysis/kitti_frame_decomposition.py tools/analysis/kitti_eval_cpu.py"
if ! git -C /home/koyama/code/ST3D diff --quiet 935dcb1 -- $PINNED; then
  echo "the E1 scripts differ from 935dcb1 in the live tree; refusing (design: scripts at 935dcb1, unchanged)" >&2
  git -C /home/koyama/code/ST3D diff --stat 935dcb1 -- $PINNED >&2
  exit 2
fi
echo "E1 scripts identical to 935dcb1: $PINNED"
echo "eval-path changes since 935dcb1 (beside; the gate guards the AP):"
git -C /home/koyama/code/ST3D diff --stat 935dcb1 -- pcdet/datasets/nuscenes pcdet/datasets/kitti/kitti_object_eval_python pcdet/datasets/dataset.py || true
for w in bb ss; do
  [ -f "$OUT/seed667/$w.env" ] || { echo "missing $OUT/seed667/$w.env" >&2; exit 2; }
  echo "--- $w.env"; cat "$OUT/seed667/$w.env"
done
# shellcheck disable=SC1091
BB_IN_PRED=$(. "$OUT/seed667/bb.env"; echo "$IN_PRED"); BB_IN_LOGGED=$(. "$OUT/seed667/bb.env"; echo "$IN_LOGGED")
BB_X_PRED=$(. "$OUT/seed667/bb.env"; echo "$X_PRED");   BB_X_LOGGED=$(. "$OUT/seed667/bb.env"; echo "$X_LOGGED")
SS_IN_PRED=$(. "$OUT/seed667/ss.env"; echo "$IN_PRED"); SS_IN_LOGGED=$(. "$OUT/seed667/ss.env"; echo "$IN_LOGGED")
SS_X_PRED=$(. "$OUT/seed667/ss.env"; echo "$X_PRED");   SS_X_LOGGED=$(. "$OUT/seed667/ss.env"; echo "$X_LOGGED")
run() { singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 "$@"; }
# the same four models and pairs as env_splits_job.sh e1, seed 667 (B->S = the B->B 667 model on Singapore val, S->B = the
# S->S 667 model on Boston val)
run analysis/scene_error_decomposition.py \
  --model "B->B=$C/centerpoint-sourceonly-nuscenes-boston2boston.yaml=$BB_IN_PRED=$BB_IN_LOGGED" \
  --model "S->B=$C/centerpoint-sourceonly-nuscenes-singapore2boston.yaml=$SS_X_PRED=$SS_X_LOGGED" \
  --model "S->S=$C/centerpoint-sourceonly-nuscenes-singapore2singapore.yaml=$SS_IN_PRED=$SS_IN_LOGGED" \
  --model "B->S=$C/centerpoint-sourceonly-nuscenes-boston2singapore.yaml=$BB_X_PRED=$BB_X_LOGGED" \
  --pair "S->B:B->B" --pair "B->S:S->S" --out "$OUT/e1_scene_error_decomposition_seed667.json"
run analysis/env_e1_two_seed.py --seed666 "$OUT/e1_scene_error_decomposition.json" \
  --seed667 "$OUT/e1_scene_error_decomposition_seed667.json" --out "$OUT/e1_two_seed.json"
echo "=== ENV E1 seed 667 exit 0 at $(date '+%Y-%m-%d %H:%M %Z') ==="
