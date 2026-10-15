#!/usr/bin/env bash
# One Slurm CPU job = P sim workers on the shared queue (experiments_md 20261015_02 §7). Engine from $ENGINE (default fast).
#   sbatch ... --cpus-per-task=C --wrap="lit_port/run_sim_shard.sh P"
P=${1:-8}
ROOT=/home/koyama/data/lit/data
cd /home/koyama/code/ST3D/tools
python3 lit_port/lit_reclaim.py $ROOT
pids=()
for i in $(seq 0 $((P-1))); do
  lit_port/lit_exec.sh python -I lit_port/lit_worker.py --role sim --root $ROOT --plan $ROOT/plan.json --tag $i \
      --engine ${ENGINE:-fast} > $ROOT/logs/sim_${SLURM_JOB_ID}_$i.log 2>&1 &
  pids+=($!)
  sleep 5
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
exit $rc
