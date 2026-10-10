#!/usr/bin/env bash
# CPU job for experiments_md 20261011_02: kitti_cut_check.py, then preflight_eval.py on the oracle config and each of its
# 21 eval twins, one config per process. Runs inside the container from ST3D/tools. Do not edit while a job runs it.
set -u
python3 analysis/kitti_cut_check.py --out /home/koyama/data/kitti_sensitivity/cut_check_20261011.json
for c in cfgs/da-ieee-access/centerpoint-sourceonly-kitti2kitti-noros-zshift.yaml \
         cfgs/da-ieee-access-analysis/sensitivity/centerpoint-sourceonly-kitti2kitti-noros-zshift-*.yaml; do
  if python3 analysis/preflight_eval.py "$c" > /tmp/pf_$$.log 2>&1; then echo "PREFLIGHT OK $c"; else echo "PREFLIGHT FAIL $c"; tail -5 /tmp/pf_$$.log; fi
done
rm -f /tmp/pf_$$.log
echo "ALL DONE"
