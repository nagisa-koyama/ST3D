#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 43b, report 20261109_01 amendments A7-A8 (2026-11-09 07:33 JST): the CPU checks of L'_K
# rerun after the first run (job 32416, scripts/_item43b_checks_20261109.sh) found
#   - the gate DIFFER on n008 (0-5 m x1.28): the thinning step's default guard left the 0-1.5 m bin uncorrected; L'_K now
#     carries MIN_HIST_BIN_FRACTION 0.0 (A7);
#   - identity "DIFFERS" on 1 of 96 frames by gt_boxes only, and the infos check "DIFFERS" by float noise; both checks now
#     classify what differs (A8).
# The single-sweep measurements do not depend on the arm and are reused from the first run (/home/koyama/data/item43/).
# Outputs: /home/koyama/data/item43/rerun_a7/.
#   bash scripts/_item43b_checks_rerun_20261109.sh            # inside a CPU job (scripts/submit.sh --wrap=...)
set -u
cd /home/koyama/code/ST3D/tools
FIRST=/home/koyama/data/item43
OUT=$FIRST/rerun_a7
mkdir -p "$OUT"
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
PY() { singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 "$@"; }
D=cfgs/da-ieee-access-analysis/item43
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
{
  echo "=== item 43b checks rerun $(date '+%F %T') on $(hostname), ST3D $(git -C .. rev-parse --short HEAD) ==="
  PY analysis/make_item43_configs.py --check
  PY analysis/item43_config_check.py
} > "$OUT/config.log" 2>&1
PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43b-lprime-k-gate-n008.yaml 300 "$OUT/gate_lprime_n008.json" \
  > "$OUT/gate_lprime_n008.log" 2>&1 &
PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43b-lprime-k-gate-n015.yaml 300 "$OUT/gate_lprime_n015.json" \
  > "$OUT/gate_lprime_n015.log" 2>&1 &
( PY analysis/item43b_checks.py identity 48 "$OUT/identity.json"
  PY analysis/item43b_checks.py inert 24 "$OUT/inert.json"
  PY analysis/item43b_checks.py infos "$OUT/infos.json" ) > "$OUT/checks.log" 2>&1 &
PY analysis/item43b_checks.py emptied n008 300 "$OUT/emptied_n008.json" > "$OUT/emptied_n008.log" 2>&1 &
PY analysis/item43b_checks.py emptied n015 300 "$OUT/emptied_n015.json" > "$OUT/emptied_n015.log" 2>&1 &
wait
{
  echo "# item 43b gate after A7 (experiments_md 20261109_01), $(date '+%F %T')"
  PY analysis/arm_point_count_bins.py report "$OUT/gate_lprime_n008.json" "$FIRST/gate_single_n008.json" \
    "n008 Boston: L'_K (lattice column) vs single sweep (density column)"
  PY analysis/arm_point_count_bins.py report "$OUT/gate_lprime_n015.json" "$FIRST/gate_single_n015.json" \
    "n015 Singapore: L'_K (lattice column) vs single sweep (density column)"
  echo; echo "## pooled by platform TRAIN frames"
  PY analysis/item43b_checks.py pooled "$OUT/gate_lprime_n008.json" "$OUT/gate_lprime_n015.json" "$FIRST/gate_single_pooled.json"
} > "$OUT/gate_report.md" 2>&1
echo "=== done $(date '+%F %T') ===" >> "$OUT/config.log"
/usr/bin/grep -h "MATCHED\|DIFFER\|EXACT\|INERT\|SAME\|recount\|calibration\|PASSED\|FAILED\|DIFFERS\|Traceback" "$OUT"/*.log "$OUT/gate_report.md"
