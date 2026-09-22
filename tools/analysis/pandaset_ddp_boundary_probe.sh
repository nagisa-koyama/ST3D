#!/usr/bin/env bash
# One-off probe: find the max per-GPU batch size PandaSet can hold under 2-GPU DDP.
# Tests each candidate as its own singularity exec (one process per DDP launch, same
# no-accumulation-across-combos principle as sweep_throughput.sh). RUN_SECONDS=240: measured
# 2026-09-22 that PandaSet+workers=8 DDP startup alone can take 150s+ before the FIRST training
# iteration even appears (two prior attempts at 45s and 150s both got killed mid-startup with
# zero iterations completed, landing as neither a real success nor an OOM - a useless result). A
# real OOM shows up as an exception within the first iteration or two once training starts, so
# 240s leaves ~90s of margin past the observed worst-case startup for that to surface.
#
# bs_per_gpu=12 (2026-09-22): three consecutive attempts (45s, 150s, 240s windows) all ended the
# same way - the dataloader/training-bar startup completed, then ZERO iterations finished before
# the timeout killed it. Not an OOM exception, not slow startup - a genuine stall during the
# first forward/backward/optimizer step, most likely CUDA allocator thrashing or a blocked NCCL
# collective right at PandaSet's single-GPU memory ceiling (bs=12 alone uses 47.0/48 GB). DDP does
# NOT simply inherit the single-GPU memory profile: a batch that barely fits solo can destabilize
# once DDP's own overhead (NCCL buffers, a second live process) is added. Backed off to search the
# gap between the confirmed-safe bs=6 (8.76 samples/s, clean) and the unstable bs=12 instead.
set -u
cd /home/koyama/code/ST3D/tools
OUT=analysis/throughput_results/pandaset_ddp_boundary.tsv
: > "$OUT"
for bs_per_gpu in 8 10; do
  total=$((bs_per_gpu * 2))
  echo "=== per-GPU bs=$bs_per_gpu (total=$total) ===" 1>&2
  SUMMARY=/home/koyama/code/ST3D/tools/analysis/throughput_results/probe_${bs_per_gpu}.log
  # The FULL raw training log, same path ddp_smoke_test.sh computes internally - NOT SUMMARY.
  # SUMMARY is ddp_smoke_test.sh's own last-5-lines report, whose regex
  # (train: N%|...|N/M [..<.., X it/s]) stops at the closing "]" and never captures the
  # ", total_it=N" suffix tqdm appends after it - so total_it= never appears in SUMMARY at all,
  # regardless of how many iterations actually ran. Read the raw log instead.
  LOG=/home/koyama/code/ST3D/tools/analysis/throughput_results/ddp_centerpoint-sourceonly-pandaset_ngpu2_bs${total}_w8.log
  bash analysis/ddp_smoke_test.sh cfgs/da-ieee-access/centerpoint-sourceonly-pandaset.yaml 2 "$total" 240 8 \
    > "$SUMMARY" 2>&1
  # FIXED 2026-09-22 (first bug): the earlier version's success check was `grep train: *[0-9]+%`
  # against SUMMARY, which trivially matches "train:   0%|...| 0/204" - the progress bar's INITIAL
  # state, printed the instant tqdm is constructed, before any real iteration runs. That produced
  # a false "OK" for bs_per_gpu=12, which direct log inspection showed was actually stuck at 0/N
  # for the entire 240s window and killed by SIGTERM.
  # FIXED 2026-09-22 (second bug, found immediately after the first fix): switched to checking
  # SUMMARY's total_it=N, which is NEVER present in SUMMARY at all (see LOG/SUMMARY split above) -
  # this produced a false "STALLED" for bs_per_gpu=7 even though direct inspection showed 11 real
  # completed iterations. Now reads the real per-iteration counter from the full raw LOG.
  max_it=$(tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE 'total_it=[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  max_it=${max_it:-0}
  if grep -qa "CUDA out of memory\|OutOfMemoryError" "$LOG"; then
    echo "bs_per_gpu=$bs_per_gpu total=$total OOM" >> "$OUT"
  elif [ "$max_it" -gt 0 ]; then
    echo "bs_per_gpu=$bs_per_gpu total=$total OK max_it=$max_it" >> "$OUT"
  elif grep -qa "ChildFailedError" "$LOG"; then
    echo "bs_per_gpu=$bs_per_gpu total=$total FAILED(no_OOM_string,see_log)" >> "$OUT"
  else
    echo "bs_per_gpu=$bs_per_gpu total=$total STALLED(0_iters_in_window,check $LOG)" >> "$OUT"
  fi
  sleep 5
done
cat "$OUT" 1>&2
