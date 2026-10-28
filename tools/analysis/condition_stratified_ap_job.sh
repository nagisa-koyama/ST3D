#!/usr/bin/env bash
# P2-A, the condition-stratified reading of saved predictions (ANALYSIS; experiments_md 20261028_02; design FIXED in
# T05 §5 "PHASE 2, design FIXED 12:04 JST", experiments_md af7680c). CPU-only Slurm job, from ST3D/tools:
#
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=8 --mem=64G \
#     --time=06:00:00 --job-name=condP2A --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     analysis/condition_stratified_ap_job.sh all
#
# Parts: `all` = capture every P2-A row (5 in parallel), then read Waymo and nuScenes; `capture` / `read` alone.
# Each row is captured from its own evaluation log (cfg, result dir, logged Car BEV / 3D AP_R40 @0.7 of the epoch
# below); the logged values are also written here and must agree with the log to 1e-4. Gate: the recomputed full-val
# Car BEV must equal the logged one within 0.05, else that capture exits 3 and the read refuses to run.
#
# Adding a row later (PL-GT 30904 once its rescore 30905 has scored epoch 20; a P2-B condition-oracle arm):
#   singularity exec --bind /home/koyama/data/:/storage $SIF python3 analysis/condition_stratified_ap.py capture \
#     --target waymo --name 30904 --from_log logs/error_30905_<name>.txt --epoch 20
#   then re-run `read` for that target with one more --row (e.g. 'PL-GT 30904=30904') and, for a P2-B pair,
#   --pair '<cond arm>:<base arm>'. The bootstrap draws depend only on --seed and the cells, so every earlier number
#   reproduces exactly. Use a new --out directory so the P2-A outputs stay as they are.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/cond_p2a_20261028
P=${P:-5}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "$OUT/logs" "$OUT/rows"
echo "=== P2-A $1 at ST3D $(git -C /home/koyama/code/ST3D log -1 --format=%h) on $(hostname), $(date '+%Y-%m-%d %H:%M %Z') ==="

# target name log epoch logged_bev logged_3d
# Waymo val (7,998 frames at test interval 5). 27704 is scored by its rescore 27803, the teacher 27471 by 27596, the
# control (25743, single sweep) by 27178; 28563 in-run.
ROWS=(
  "waymo 27179 logs/error_27179_evWa2Wa.txt 7 65.9189 51.7702"
  "waymo 27704 logs/error_27803_evWymPL.txt 20 55.6625 32.8152"
  "waymo 28563 logs/error_28563_wymPL667.txt 20 55.7085 32.8011"
  "waymo 27471 logs/error_27596_gbLegalWaymo.txt 20 51.5805 28.3536"
  "waymo 27178 logs/error_27178_evNu2Wa.txt 20 2.2001 0.8374"
  # nuScenes val (6,019 keyframes). The oracle 25743 is scored by 25768; every other row in-run at its final epoch.
  "nuscenes 25743 logs/error_25768_recover_eval.txt 20 44.9825 26.6326"
  "nuscenes 27594 logs/error_27594_stDsbnFfix.txt 30 33.3943 15.4344"
  "nuscenes 28304 logs/error_28304_s1M667.txt 30 33.2134 15.1765"
  "nuscenes 27592 logs/error_27592_stCBfix.txt 30 23.0251 8.3445"
  "nuscenes 28394 logs/error_28394_s1C667.txt 30 23.8670 9.0213"
  "nuscenes 27807 logs/error_27807_ringW2N.txt 7 36.1179 16.7306"
  "nuscenes 27840 logs/error_27840_s2RingW2N.txt 7 36.2983 16.5661"
  "nuscenes 27850 logs/error_27850_c1W2N.txt 7 31.1395 14.3550"
  "nuscenes 28278 logs/error_28278_c1W2Ns2.txt 7 30.6949 13.6495"
  "nuscenes 28034 logs/error_28034_rpPANDA.txt 115 33.9326 15.4780"
  "nuscenes 28279 logs/error_28279_rpPANDAs2.txt 115 34.2755 16.1122"
  "nuscenes 28041 logs/error_28041_c1PANDA.txt 115 28.5720 14.3119"
  "nuscenes 28280 logs/error_28280_c1PANDAs2.txt 115 28.2357 13.9235"
)

capture_one() {   # target name log epoch logged_bev logged_3d
  rm -f "$OUT/rows/$1/$2.pkl"
  if singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/condition_stratified_ap.py capture \
      --target "$1" --name "$2" --from_log "$3" --epoch "$4" --logged "$5" --logged_3d "$6" --rows_dir "$OUT/rows" \
      > "$OUT/logs/capture_$1_$2.log" 2>&1; then
    echo "  capture $1 $2: ok ($(date '+%H:%M'))"; tail -n 12 "$OUT/logs/capture_$1_$2.log" | sed 's/^/    /'
  else
    echo "  capture $1 $2: FAILED, exit $? ($(date '+%H:%M'))"; tail -n 30 "$OUT/logs/capture_$1_$2.log" | sed 's/^/    /'
    return 1
  fi
}
export -f capture_one
export SIF OUT

do_capture() {
  printf '%s\n' "${ROWS[@]}" | /usr/bin/grep -v '^ *#' | xargs -P "$P" -L 1 bash -c 'capture_one "$@"' _
}

do_read() {
  local run=(singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 analysis/condition_stratified_ap.py read
             --rows_dir "$OUT/rows" --n_boot 1000 --seed 20261028 --workers "${SLURM_CPUS_PER_TASK:-8}")
  "${run[@]}" --target waymo --oracle 27179 \
    --row 'PL 27704/28563=27704,28563' --row 'teacher 27471=27471' --row 'control 25743 (27178)=27178' \
    --out "$OUT/waymo"
  "${run[@]}" --target nuscenes --oracle 25743 \
    --row 'S1 27594/28304=27594,28304' --row 'S1 matched baseline 27592/28394=27592,28394' \
    --row 'W->N P 27807/27840=27807,27840' --row 'W->N C1 27850/28278=27850,28278' \
    --row 'P->N P 28034/28279=28034,28279' --row 'P->N C1-analog 28041/28280=28041,28280' \
    --out "$OUT/nuscenes"
}

case "${1:?usage: condition_stratified_ap_job.sh all|capture|read}" in
  all) do_capture; do_read ;;
  capture) do_capture ;;
  read) do_read ;;
  *) echo "unknown part $1" >&2; exit 2 ;;
esac
echo "=== P2-A $1 exit 0 at $(date '+%Y-%m-%d %H:%M %Z') ==="
