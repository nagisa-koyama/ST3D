#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 64 (design fixed 2026-11-27 13:02 JST, before any number): the three evaluations of one
# count-normalized row's epoch-152 checkpoint: src (S' val, own statistics), none (KITTI val, own statistics), ada (KITTI
# val after global AdaBN on 1,000 strided KITTI train clouds, no labels). Outputs: analysis/item64_si.py dir MODE SEED.
# Usage (from ST3D/tools, inside a 1-GPU allocation): bash scripts/_item64_evals_20261127.sh JOB SEED
# Each evaluation logs its exit status; the next runs regardless; exit 1 if any failed. Do not edit while a job runs it.
set -u
JOB=$1; SEED=$2
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
CFG=cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si.yaml
CFG_SRC=cfgs/da-ieee-access-analysis/item64/centerpoint-item64-pk-si-on-sprime.yaml
C=$(bash analysis/ckpt_of_job.sh "$JOB" 152) || { echo "item64: no epoch-152 checkpoint for job $JOB"; exit 2; }
echo "=== item64 evaluations, job $JOB seed $SEED, checkpoint $C ($(date '+%F %T')) ==="
F=0
for mode in src none ada; do
  case $mode in src) cfg=$CFG_SRC; adapt=none ;; none) cfg=$CFG; adapt=none ;; ada) cfg=$CFG; adapt=ada ;; esac
  out=$(python3 analysis/item64_si.py dir "$mode" "$SEED")
  echo "=== item64 $mode (seed $SEED) -> $out ($(date '+%F %T')) ==="
  WANDB_NOTES="Item 64 (T05 §5, design 2026-11-27 13:02 JST), labelled ANALYSIS: P_K + count-normalized sparse convolution, job $JOB seed $SEED, eval $mode (src: S' val; none: KITTI val; ada: KITTI val after global AdaBN on 1000 strided KITTI train clouds, no labels). Readings fixed in T05 item 64." \
    singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
      python3 analysis/item64_si_eval.py --cfg_file "$cfg" --ckpt "$C" --adapt "$adapt" --out_dir "$out" \
        --run_name "i64_${mode}_s${SEED}" < /dev/null
  s=$?; echo "exit $s"; [ $s -eq 0 ] || F=$((F+1))
done
echo "=== item64 job $JOB seed $SEED done: failures $F ($(date '+%F %T')) ==="
[ $F -eq 0 ]
