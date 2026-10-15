#!/usr/bin/env bash
# One Slurm job = N_GPU gpu workers (one per GPU) on the shared queue (experiments_md 20261015_02 §7). Resume-safe.
#   sbatch ... --gres=gpu:2 --wrap="lit_port/run_gpu_shard.sh 2"
N=${1:-1}
ROOT=/home/koyama/data/lit/data
cd /home/koyama/code/ST3D/tools
python3 lit_port/lit_reclaim.py $ROOT
nvidia-smi --query-gpu=name,driver_version --format=csv
pids=()
for i in $(seq 0 $((N-1))); do
  CUDA_VISIBLE_DEVICES=$i lit_port/lit_exec.sh --nv python -I lit_port/lit_worker.py --role gpu --root $ROOT \
      --plan $ROOT/plan.json --tag $i --scene_tmp $ROOT/scene_tmp/${SLURM_JOB_ID}_$i > $ROOT/logs/gpu_${SLURM_JOB_ID}_$i.log 2>&1 &
  pids+=($!)
  sleep 30
done
rc=0
for p in "${pids[@]}"; do wait $p || rc=1; done
exit $rc
