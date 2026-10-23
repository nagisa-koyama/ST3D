#!/usr/bin/env bash
# ENV E1 / E2 (experiments_md 20261024_02; design fixed in T05 §5 "User 10-24 ~02:40" item 4, commit 3cddbb7).
# CPU-only Slurm job, from ST3D/tools (a bash script, not a dash --wrap):
#
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=32G \
#     --time=03:00:00 --job-name=envE1 --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     analysis/env_splits_job.sh e1        (or e2)
#
# Saved predictions (epoch 20 of 30188 B->B and 30189 S->S; the cross-city readings rescored by 30303 / 30304 because
# the result.pkl of the 30190 / 30191 `none` arms was overwritten by later AdaBN runs in the same output dir; the
# in-domain oracle 25743's eval 25768). Logged Car BEV AP_R40 @0.7 for the sanity gate, read from the logs:
#   B->B 44.2594 (error_30188_sceneBB.txt, epoch 20), S->S 40.5096 (error_30189_sceneSS.txt, epoch 20),
#   B->S 39.5360 (error_30190_sceneBBx.txt, none arm), S->B 34.7584 (error_30191_sceneSSx.txt, none arm),
#   oracle 44.9825 (error_25768_recover_eval.txt).
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/env_splits
C=cfgs/da-ieee-access
BB=/home/koyama/data/wandb/run-20261022_225702-ulio5sgf/files/eval/eval_with_train/epoch_20/val/result.pkl
SS=/home/koyama/data/wandb/run-20261022_225659-6iu7h6p5/files/eval/eval_with_train/epoch_20/val/result.pkl
BS=/home/koyama/code/ST3D/output/da-ieee-access/centerpoint-sourceonly-nuscenes-boston2singapore/20261024_env/eval/epoch_20/val/env_e1/result.pkl
SB=/home/koyama/code/ST3D/output/da-ieee-access/centerpoint-sourceonly-nuscenes-singapore2boston/20261024_env/eval/epoch_20/val/env_e1/result.pkl
OR=/home/koyama/code/ST3D/output/da-ieee-access/centerpoint-sourceonly-nuscenes/20260922_sourceonly/eval/epoch_20/val/recover_ep20/result.pkl
mkdir -p "$OUT"
echo "=== ENV $1 at ST3D $(git -C /home/koyama/code/ST3D log -1 --format=%h) on $(hostname), $(date '+%Y-%m-%d %H:%M %Z') ==="
run() { singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 "$@"; }
case "${1:?usage: env_splits_job.sh e1|e2}" in
  e1)
    run analysis/scene_error_decomposition.py \
      --model "B->B=$C/centerpoint-sourceonly-nuscenes-boston2boston.yaml=$BB=44.2594" \
      --model "S->B=$C/centerpoint-sourceonly-nuscenes-singapore2boston.yaml=$SB=34.7584" \
      --model "S->S=$C/centerpoint-sourceonly-nuscenes-singapore2singapore.yaml=$SS=40.5096" \
      --model "B->S=$C/centerpoint-sourceonly-nuscenes-boston2singapore.yaml=$BS=39.5360" \
      --pair "S->B:B->B" --pair "B->S:S->S" --out "$OUT/e1_scene_error_decomposition.json"
    ;;
  e2)
    run analysis/nuscenes_condition_ap.py \
      --model "B->B=$C/centerpoint-sourceonly-nuscenes-boston2boston.yaml=$BB=44.2594=Boston" \
      --model "S->S=$C/centerpoint-sourceonly-nuscenes-singapore2singapore.yaml=$SS=40.5096=Singapore" \
      --model "oracle=$C/centerpoint-sourceonly-nuscenes.yaml=$OR=44.9825=Boston,Singapore" \
      --out "$OUT/e2_condition_ap.json"
    ;;
  *) echo "unknown part $1" >&2; exit 2 ;;
esac
echo "=== ENV $1 exit 0 at $(date '+%Y-%m-%d %H:%M %Z') ==="
