#!/usr/bin/env bash
# Per-GPU memory for the committed 2-GPU / global-batch-6 recipe, measured at DEVICE level.
#
# WHY device level and not torch's counters: profile_throughput.py reports
# torch.cuda.max_memory_allocated/reserved, which exclude the CUDA context and NCCL's own buffers -
# exactly the parts DDP adds. 20260922_06 section 3 already showed DDP does NOT inherit the
# single-GPU memory profile (PandaSet bs=12 fits solo at 47.0/48 GB but HANGS under DDP), so the
# single-GPU bs=3 figure is a lower bound, not an answer.
#
# Sampling nvidia-smi is sound here because SLURM allocates whole GPUs: --gres=gpu:2 means those
# two devices are ours alone, so memory.used on them is entirely this job's.
#
# PandaSet is the subject: it is the memory-heaviest source (19.85 GB allocated at bs=12, with
# reserved touching 47.07 of 48 GB).
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access/centerpoint-sourceonly-pandaset.yaml
OUT=analysis/throughput_results/ddp_memory_probe.tsv
: > "$OUT"
echo "node=$(hostname)" >> "$OUT"

sample_peak () {   # $1 = label, samples until the pid in $2 exits
  local label=$1 pid=$2 peak0=0 peak1=0
  while kill -0 "$pid" 2>/dev/null; do
    read -r m0 m1 <<< "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    [ -n "${m0:-}" ] && [ "$m0" -gt "$peak0" ] && peak0=$m0
    [ -n "${m1:-}" ] && [ "$m1" -gt "$peak1" ] && peak1=$m1
    sleep 2
  done
  echo "$label peak_gpu0_MiB=$peak0 peak_gpu1_MiB=$peak1" >> "$OUT"
  echo "$label peak_gpu0_MiB=$peak0 peak_gpu1_MiB=$peak1" 1>&2
}

# --- arm 1: single GPU, bs=6 (the reference recipe) ---
echo "=== 1 GPU bs=6 ===" 1>&2
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 timeout --signal=KILL 480 singularity exec --nv \
    --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
    python analysis/profile_throughput.py --cfg_file "$CFG" --batch_size 6 --workers 8 \
    --warmup 5 --measure 20 > analysis/throughput_results/mem_1gpu_bs6.log 2>&1 &
sample_peak "1GPU_bs6_global6" $!
wait; sleep 10

# --- arm 2: 2 GPUs, 3/rank = global batch 6 (THE COMMITTED RECIPE) ---
echo "=== 2 GPU 3/rank (global 6) ===" 1>&2
bash analysis/ddp_smoke_test.sh "$CFG" 2 6 200 8 > /dev/null 2>&1 &
sample_peak "2GPU_3perRank_global6" $!
wait; sleep 10

# --- arm 3: 2 GPUs, 6/rank = global batch 12 (the rejected alternative) ---
echo "=== 2 GPU 6/rank (global 12) ===" 1>&2
bash analysis/ddp_smoke_test.sh "$CFG" 2 12 200 8 > /dev/null 2>&1 &
sample_peak "2GPU_6perRank_global12" $!
wait

echo "=== RESULTS ===" 1>&2
cat "$OUT" 1>&2
