#!/usr/bin/env bash
# Multi-GPU DDP smoke test: measures the SAME (batch, LR, iterations) recipe as the single-GPU
# baseline, distributed across NGPUS on one node via torch.distributed.launch, and reports the
# tqdm iteration rate so it is directly comparable to the profile_throughput.py single-GPU
# numbers.
#
# --batch_size TOTAL is passed explicitly (not left to default to BATCH_SIZE_PER_GPU) so
# train.py divides it across GPUs (train.py:87-91) rather than multiplying it - that is what
# keeps the effective batch size, LR schedule and iteration count IDENTICAL to the single-GPU
# run, isolating "more GPUs" as the only variable. TOTAL must be evenly divisible by NGPUS.
#
# Runs for a short, fixed WALL-CLOCK duration (not a full epoch) and is killed after: this is a
# rate measurement, not a training run, and CenterPoint-sourceonly budgets are 92k-158k
# iterations, far more than needed to get a stable it/s reading.
#
# --workers is PER PROCESS (i.e. per GPU), matching profile_throughput.py's convention, so a
# PandaSet run should pass 8 here - at the default 4 it would just reproduce the known 46.5%
# dataloader-bound result and confound the GPU-count/batch-size measurement with an already-solved
# problem.
#
# Usage (inside an existing srun/sbatch GPU allocation, from ST3D/tools):
#   analysis/ddp_smoke_test.sh <cfg_file> <ngpus> <total_batch_size> <run_seconds> [workers]
set -u

CFG_FILE=$1
NGPUS=$2
TOTAL_BS=$3
RUN_SECONDS=${4:-150}
WORKERS=${5:-4}
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
LOG=analysis/throughput_results/ddp_$(basename "$CFG_FILE" .yaml)_ngpu${NGPUS}_bs${TOTAL_BS}_w${WORKERS}.log

echo "=== DDP smoke: $CFG_FILE ngpus=$NGPUS total_bs=$TOTAL_BS workers=$WORKERS for ${RUN_SECONDS}s ===" 1>&2

PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))

# --use-env: without it, torch>=2.x's launcher passes --local-rank (hyphen) to the child, which
# train.py's argparse does not recognize under any spelling - see the --local_rank comment in
# tools/train.py (fixed 2026-09-22) for the full story and why --use-env alone would NOT have been
# enough without that fix (every process would default to local_rank=0 and collide).
singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
    python -m torch.distributed.launch --use-env --nproc_per_node="$NGPUS" --rdzv_endpoint=localhost:"$PORT" \
    train.py --launcher pytorch --tcp_port "$PORT" \
    --cfg_file "$CFG_FILE" --batch_size "$TOTAL_BS" --workers "$WORKERS" \
    --epochs 1 --num_epochs_to_eval 0 --run_name "ddp_smoke_ngpu${NGPUS}" \
    --extra_tag 20260922_ddp_smoke \
    > "$LOG" 2>&1 &
PID=$!

sleep "$RUN_SECONDS"
kill -TERM "$PID" 2>/dev/null
sleep 5
kill -KILL "$PID" 2>/dev/null
pkill -TERM -P "$PID" 2>/dev/null

echo "=== last tqdm rate lines (tr \\r -> \\n first, per the tqdm-log-parsing convention) ===" 1>&2
tr '\r' '\n' < "$LOG" | grep -oE "train: *[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+, *[0-9.]+(it/s|s/it)\]" \
    | tail -5 | tee /dev/stderr
echo "=== full log at $LOG ===" 1>&2
