#!/usr/bin/env bash
# One Slurm CPU job = P sim workers on the shared queue, WITH the label-free ray-drop (experiments_md 20261017_01 §4). Output:
# /home/koyama/data/lit/data/sim_nuscenes_rd (a NEW directory; sim_nuscenes is untouched), stage `simrd`.
P=${1:-8}
ROOT=/home/koyama/data/lit/data
cd /home/koyama/code/ST3D/tools
python3 lit_port/lit_reclaim.py $ROOT
pids=()
for i in $(seq 0 $((P-1))); do
  lit_port/lit_exec.sh python -I lit_port/lit_worker.py --role sim --root $ROOT --plan $ROOT/plan.json --tag $i \
      --engine fast --ray_drop $ROOT/ray_drop_v1.json --out_name sim_nuscenes_rd > $ROOT/logs/simrd_${SLURM_JOB_ID}_$i.log 2>&1 &
  pids+=($!)
  sleep 5
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
exit $rc
