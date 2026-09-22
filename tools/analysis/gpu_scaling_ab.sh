#!/usr/bin/env bash
# Same-node 1-GPU vs 2-GPU A/B, plus the sub-6 batch question. PandaSet, workers=8 throughout.
#
# WHY THIS EXISTS: every DDP speedup in 20260922_06 section 2b is confounded. PandaSet's
# single-GPU profile ran on rtx8000/node61 (job 25692) and its 2-GPU run on a6000/node12 (job
# 25697), so "1.93x" conflates a second GPU with a faster GPU; rescaling by the one controlled
# hardware pair available suggests 1.40x instead. No source in that report has both measurements on
# one GPU type. This script takes ONE allocation of 2 GPUs on ONE node and measures every arm
# inside it, so the hardware is controlled by construction rather than by a correction factor.
#
# Arms:
#   1 GPU  bs=3   - is per-sample cost still flat below 6? The saturation result only covers bs>=6,
#                   and 3 is what 2-GPU DDP at a global batch of 6 would put on each rank.
#   1 GPU  bs=6   - the configured recipe, and the denominator for everything else
#   1 GPU  bs=12  - global batch 12 on one GPU, the honest comparison for 2-GPU at 6/rank
#   2 GPU  3/rank - global batch 6: IDENTICAL recipe to the configured run (same steps, same LR)
#   2 GPU  6/rank - global batch 12: what was measured before, now on the same node as its baseline
#
# Each single-GPU arm is its own `singularity exec` (one process per combo, same
# no-shm-accumulation principle as sweep_throughput.sh) and is pinned to GPU 0 so the idle second
# GPU cannot flatter it.
#
# Usage: gpu_scaling_ab.sh [cfg_file] [workers]   (defaults: PandaSet, 8)
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=${1:-cfgs/da-ieee-access/centerpoint-sourceonly-pandaset.yaml}
W=${2:-8}
NAME=$(basename "$CFG" .yaml)
OUT=analysis/throughput_results/gpu_scaling_ab_${NAME}_w${W}.tsv
: > "$OUT"
echo "node=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-unset}" | tee -a "$OUT" 1>&2

for bs in 3 6 12; do
  echo "=== 1 GPU, bs=$bs, w$W ===" 1>&2
  LINE=$(SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 600 singularity exec --nv \
      --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python analysis/profile_throughput.py --cfg_file "$CFG" \
      --batch_size "$bs" --workers "$W" --warmup 10 --measure 40 \
      2>&1 | tee /dev/stderr | grep '^RESULT ')
  echo "${LINE:-FAILED 1gpu bs=$bs}" >> "$OUT"
  sleep 5
done

# DDP arms. ddp_smoke_test.sh takes <cfg> <ngpus> <TOTAL batch> <seconds> <workers>; TOTAL is
# divided across ranks by train.py, so 6 -> 3/rank and 12 -> 6/rank.
for total in 6 12; do
  echo "=== 2 GPU, total_bs=$total ($((total/2))/rank), w$W ===" 1>&2
  bash analysis/ddp_smoke_test.sh "$CFG" 2 "$total" 300 "$W" > /dev/null 2>&1
  LOG=analysis/throughput_results/ddp_${NAME}_ngpu2_bs${total}_w${W}.log
  # Read the RAW log, not the smoke script's summary - its regex stops at "]" and never captures
  # the per-iteration counter (the second harness bug recorded in 20260922_06 section 6).
  RATE=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE '[0-9.]+(it/s|s/it)' | tail -1)
  ITERS=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE 'total_it=[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  echo "DDP total_bs=$total per_rank=$((total/2)) workers=$W rate=${RATE:-none} max_it=${ITERS:-0}" >> "$OUT"
  sleep 5
done

echo "=== RESULTS ===" 1>&2
cat "$OUT" 1>&2
