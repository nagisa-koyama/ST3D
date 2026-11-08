#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 43b (design fixed 2026-11-09 01:18 JST; report experiments_md 20261109_01): every CPU check
# of L'_K before any GPU row (bug_prevention rules 1, 4, 8, 9), in ONE Slurm CPU job, the independent parts in parallel.
#   gate   analysis/arm_point_count_bins.py measure, 300 frames, in-grid, 5-m bins: L'_K per platform block (gate views, the
#          calibration installed as train.py installs it) and the single-sweep nuScenes source per platform (the reference
#          configs) and pooled (the oracle config); report per platform, and pooled by platform TRAIN frames
#   checks analysis/item43b_checks.py identity 48 (rate 1 exact), inert 24 (eval view = the oracle's), infos (200- vs
#          10-sweep val infos), emptied per platform 300 (what the thinning empties, what the recount drops; the rates)
#   config analysis/item43_config_check.py (against the jobs' logged configs) and make_item43_configs.py --check
# Outputs: /home/koyama/data/item43/ (JSON per part, gate_report.md, checks.log).
#   bash scripts/_item43b_checks_20261109.sh            # inside a CPU job (scripts/submit.sh --wrap=...)
set -u
cd /home/koyama/code/ST3D/tools
OUT=/home/koyama/data/item43
mkdir -p "$OUT"
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
PY() { singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 "$@"; }
D=cfgs/da-ieee-access-analysis/item43
{
  echo "=== item 43b checks $(date '+%F %T') on $(hostname), ST3D $(git -C .. rev-parse --short HEAD) ==="
  PY analysis/make_item43_configs.py --check
  PY analysis/item43_config_check.py
} > "$OUT/config.log" 2>&1
PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43b-lprime-k-gate-n008.yaml 300 "$OUT/gate_lprime_n008.json" \
  > "$OUT/gate_lprime_n008.log" 2>&1 &
PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43b-lprime-k-gate-n015.yaml 300 "$OUT/gate_lprime_n015.json" \
  > "$OUT/gate_lprime_n015.log" 2>&1 &
( PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43-ref-single-n008.yaml 300 "$OUT/gate_single_n008.json"
  PY analysis/arm_point_count_bins.py measure $D/centerpoint-i43-ref-single-n015.yaml 300 "$OUT/gate_single_n015.json"
  PY analysis/arm_point_count_bins.py measure cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml 300 \
    "$OUT/gate_single_pooled.json" ) > "$OUT/gate_single.log" 2>&1 &
( PY analysis/item43b_checks.py identity 48 "$OUT/identity.json"
  PY analysis/item43b_checks.py inert 24 "$OUT/inert.json"
  PY analysis/item43b_checks.py infos "$OUT/infos.json" ) > "$OUT/checks.log" 2>&1 &
PY analysis/item43b_checks.py emptied n008 300 "$OUT/emptied_n008.json" > "$OUT/emptied_n008.log" 2>&1 &
PY analysis/item43b_checks.py emptied n015 300 "$OUT/emptied_n015.json" > "$OUT/emptied_n015.log" 2>&1 &
wait
{
  echo "# item 43b gate (experiments_md 20261109_01), $(date '+%F %T')"
  PY analysis/arm_point_count_bins.py report "$OUT/gate_lprime_n008.json" "$OUT/gate_single_n008.json" \
    "n008 Boston: L'_K (lattice column) vs single sweep (density column)"
  PY analysis/arm_point_count_bins.py report "$OUT/gate_lprime_n015.json" "$OUT/gate_single_n015.json" \
    "n015 Singapore: L'_K (lattice column) vs single sweep (density column)"
  echo; echo "## pooled by platform TRAIN frames"
  PY analysis/item43b_checks.py pooled "$OUT/gate_lprime_n008.json" "$OUT/gate_lprime_n015.json" "$OUT/gate_single_pooled.json"
} > "$OUT/gate_report.md" 2>&1
echo "=== done $(date '+%F %T') ===" >> "$OUT/config.log"
/usr/bin/grep -h "MATCHED\|DIFFER\|EXACT\|INERT\|SAME\|recount\|PASSED\|FAILED\|DIFFERS\|Traceback" "$OUT"/*.log "$OUT/gate_report.md"
