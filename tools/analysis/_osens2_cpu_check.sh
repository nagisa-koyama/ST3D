#!/usr/bin/env bash
# CPU job for OSENS-2 (experiments_md 20261025_03), before any GPU evaluation: the unit tests, the twins' --check,
# preflight_eval on every oracle and twin (one config per process), then the one-frame-per-arm dump and per-arm
# statistics (analysis/osens2_dump.py). Runs inside the container from ST3D/tools. Do not edit while a job runs it.
#   usage: bash analysis/_osens2_cpu_check.sh <out dir under /storage/osens2> [frames]
set -u
OUT=${1:?out dir}
FRAMES=${2:-60}
cd /root/ST3D && python3 -m pytest tests/test_osens2.py -q 2>&1 | tail -3
cd /root/ST3D/tools
python3 analysis/make_osens2_configs.py --check | grep -c ': OK'
for c in cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml \
         cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml cfgs/da-ieee-access-analysis/osens2/*.yaml; do
  if python3 analysis/preflight_eval.py "$c" > "$OUT/pf_$$.log" 2>&1; then echo "PREFLIGHT OK $c"; else echo "PREFLIGHT FAIL $c"; tail -5 "$OUT/pf_$$.log"; fi
done
rm -f "$OUT/pf_$$.log"
python3 analysis/osens2_dump.py --out "$OUT" --frames "$FRAMES"
echo "ALL DONE rc=$?"
