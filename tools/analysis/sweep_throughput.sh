#!/usr/bin/env bash
# Runs profile_throughput.py once per (batch_size, workers) combo, each as its OWN
# `singularity exec` -> fresh Python process -> fresh CUDA context and DataLoader workers.
# See profile_throughput.py's header for why: a single process looping over combos accumulates
# /dev/shm pressure across torn-down persistent workers and segfaults (reproduced 2026-09-22,
# job 25679, at the third combo). One process per combo tears everything down cleanly on exit.
#
# Usage (inside an existing srun/sbatch GPU allocation, from ST3D/tools):
#   analysis/sweep_throughput.sh <cfg_file> <results_dir> "<batch_sizes>" "<workers_list>"
#
# Example:
#   analysis/sweep_throughput.sh cfgs/da-ieee-access/centerpoint-sourceonly-lyft.yaml \
#       analysis/throughput_results "6 12 18 24 32 48" "4 8 12"
#
# Appends one RESULT line per successful combo to <results_dir>/<cfg-basename>.tsv and continues
# past a failed (e.g. OOM'd) combo instead of aborting the sweep.
#
# COMBO_TIMEOUT guards against a hang, not just an OOM. Observed 2026-09-22: Lyft's
# bs=24/workers=4 combo ran 13+ minutes (vs 1-4 min for every other combo, including Lyft's own
# neighbours) while node-level CPU load stayed near zero - a stall, not slow compute - most likely
# in the per-object augmentation's O(boxes^2) collision check (scale_pre_object ->
# boxes_bev_iou_cpu), which Lyft is disproportionately exposed to since it has the highest box
# density of any source here (~18-25 cars/frame). Concurrent Waymo/nuScenes combos on the SAME
# node kept progressing normally throughout, ruling out node-wide memory/CPU contention as the
# cause. Without a timeout this silently consumes an entire job slot for its whole --time budget.

set -u  # deliberately NOT -e: one combo's failure must not stop the rest of the sweep

CFG_FILE=$1
RESULTS_DIR=$2
BATCH_SIZES=$3
WORKERS_LIST=$4
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
COMBO_TIMEOUT=${COMBO_TIMEOUT:-360}  # seconds; well above the ~250s slowest legitimate combo seen

mkdir -p "$RESULTS_DIR"
OUT="$RESULTS_DIR/$(basename "$CFG_FILE" .yaml).tsv"
echo "=== sweeping $CFG_FILE -> $OUT ===" 1>&2

for workers in $WORKERS_LIST; do
    for bs in $BATCH_SIZES; do
        echo "--- bs=$bs workers=$workers ---" 1>&2
        LINE=$(timeout --signal=KILL "$COMBO_TIMEOUT" singularity exec --nv \
            --bind /home/koyama/data/:/storage \
            --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
            python analysis/profile_throughput.py --cfg_file "$CFG_FILE" \
            --batch_size "$bs" --workers "$workers" --warmup 10 --measure 40 \
            2>&1 | tee /dev/stderr | grep '^RESULT ')
        if [ -n "$LINE" ]; then
            echo "$LINE" >> "$OUT"
        else
            echo "FAILED bs=$bs workers=$workers cfg=$(basename "$CFG_FILE")" >> "$OUT"
        fi
        # Let the previous combo's worker processes and shm segments fully exit before the next
        # `singularity exec` forks new ones - the same margin the repo gives real jobs between
        # epoch boundaries.
        sleep 5
    done
done

echo "=== sweep complete: $OUT ===" 1>&2
cat "$OUT" 1>&2
