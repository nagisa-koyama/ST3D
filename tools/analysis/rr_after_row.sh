#!/usr/bin/env bash
# Chained CPU step after the relocate-and-render training row (experiments_md 20261018_02 §3.3). Submitted with
# --dependency=afterany:<row> --kill-on-invalid-dep=yes, so it runs whatever the row's final state (a 2-GPU Waymo-target
# row can read FAILED and still leave a score). Run from ST3D/tools inside a Slurm CPU job:
#
#   bash analysis/rr_after_row.sh <row jobid> <cfg> [rescored]
#
# 1. Finds the row's epoch-20 Waymo-val result.pkl: the in-run evaluation's (W&B run dir, eval_with_train/epoch_20),
#    else the launcher's auto-recovery (output/.../eval/epoch_20/val/autorecover_<row>), else a fallback rescore's
#    (eval/epoch_20/val/rr_ep20).
# 2. None found and not yet rescored: submits ONE GPU rescore of checkpoint_epoch_20 (eval_checkpoint.sh) and this step
#    again after it (afterok), then exits. None found after a rescore: says so and exits 2.
# 3. Found: prints the row's AP lines and runs R2 (analysis, no selection) against 27596: waymo_target_residual.py (per
#    range ring and stored-points bin) and waymo_ranking_loss_fp_taxonomy.py --official (recall / ranking split, FP
#    taxonomy, background FPs above the TP median). Outputs under /home/koyama/data/relocate_render_20261018/r2_<row>/.
#    RR_EXTRA_RUNS (optional, space-separated <label>=<result.pkl>) adds rows to both analyses after the reference, e.g.
#    RR_EXTRA_RUNS=rr7a_29690=<29690's result.pkl> for v8 (experiments_md 20261018_02 §13.4).
set -u
JOB=${1:?usage: rr_after_row.sh <row jobid> <cfg> [rescored]}
CFG=${2:?usage: rr_after_row.sh <row jobid> <cfg> [rescored]}
RESCORED=${3:-}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/relocate_render_20261018/r2_${JOB}
REF=/home/koyama/code/ST3D/output/da-ieee-access/centerpoint-gblobs-sourceonly-nuscenes2waymo/20260922_sourceonly/eval/epoch_20/val/gblobs_accum_legaldepth_on_waymo/result.pkl
TAG=$(basename "$CFG" .yaml)
EVD=/home/koyama/code/ST3D/output/da-ieee-access/${TAG}/20260922_sourceonly/eval/epoch_20/val
mkdir -p "$OUT"
LOG=$(ls logs/error_"${JOB}"_*.txt 2>/dev/null | head -n 1)
[ -n "$LOG" ] || { echo "rr_after_row: no logs/error_${JOB}_*.txt" >&2; exit 2; }
RUN=$(tr '\r' '\n' < "$LOG" | /usr/bin/grep -m 1 -o 'Run data is saved locally in /storage/wandb/run-[0-9_]*-[a-z0-9]*' \
      | sed 's#^Run data is saved locally in /storage#/home/koyama/data#')
RES=""
OUTD=/home/koyama/code/ST3D/output/da-ieee-access/${TAG}
for c in "$RUN/files/eval/eval_with_train/epoch_20/val/result.pkl" "$EVD/autorecover_${JOB}/result.pkl" "$EVD/rr_ep20/result.pkl" \
         "$(find "$OUTD" -path '*epoch_20*' -path "*autorecover_${JOB}*" -name result.pkl 2>/dev/null | head -n 1)" \
         "$(find "$OUTD" -path '*epoch_20*' -path '*rr_ep20*' -name result.pkl 2>/dev/null | head -n 1)"; do
    if [ -n "$c" ] && [ -f "$c" ]; then RES=$c; break; fi
done
echo "row $JOB: W&B run dir ${RUN:-<none>}; epoch-20 result.pkl: ${RES:-<none>}"
if [ -z "$RES" ]; then
    if [ -n "$RESCORED" ]; then
        echo "rr_after_row: no epoch-20 result.pkl even after the rescore - needs a human" >&2; exit 2
    fi
    JID() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
    E=$(WANDB_NOTES="Rescore RR row $JOB (relocate-and-render, experiments_md 20261018_02) epoch 20 on Waymo val: fallback, the row left no epoch-20 result.pkl" \
        scripts/submit.sh --partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=02:00:00 \
        --job-name=ev$JOB --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
        --wrap="set -e; C=\$(bash analysis/ckpt_of_job.sh $JOB 20); bash analysis/eval_checkpoint.sh $CFG \$C rr_ep20" | JID)
    [ -n "$E" ] || { echo "rr_after_row: rescore submission failed" >&2; exit 2; }
    WANDB_NOTES="CPU only, no W&B: R2 analysis of RR row $JOB after its fallback rescore $E (experiments_md 20261018_02 §3.2)" \
        scripts/submit.sh --dependency=afterok:$E --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=8 \
        --mem=48G --time=03:00:00 --job-name=r2$JOB --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
        --wrap="bash analysis/rr_after_row.sh $JOB $CFG rescored"
    echo "submitted rescore $E and R2 after it"; exit 0
fi
echo "== row $JOB AP lines (Car, Waymo val) =="
tr '\r' '\n' < "$LOG" | /usr/bin/grep -A 3 "Car AP_R40@0.70, 0.70, 0.70\|Car AP_R40@0.70, 0.50, 0.50" | head -40
ln -sf "$RES" "$OUT/result_row.pkl"
OMP_NUM_THREADS=1 singularity exec --bind /home/koyama/data/:/storage "$SIF" \
    python3 analysis/waymo_target_residual.py ref27596="$REF" ${RR_EXTRA_RUNS:-} rr="$RES" > "$OUT/residual.txt" 2>&1
echo "residual rc=$? -> $OUT/residual.txt"
OMP_NUM_THREADS=1 singularity exec --bind /home/koyama/data/:/storage "$SIF" \
    python3 analysis/waymo_ranking_loss_fp_taxonomy.py --out "$OUT/taxonomy" --workers 8 --official \
    ref27596="$REF" ${RR_EXTRA_RUNS:-} rr="$RES" > "$OUT/taxonomy.txt" 2>&1
echo "taxonomy rc=$? -> $OUT/taxonomy.txt"
