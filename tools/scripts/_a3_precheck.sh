#!/usr/bin/env bash
# A3 pre-checks on CPU, before any GPU pass (experiments_md 20261017_06 §3, §5):
#   1. the eval-mode generation set of the rule-B flat config serves teacher 27807 its training input
#      (analysis/a3_ringpattern_eval_view_check.py, 60 strided Waymo val frames);
#   2. the Waymo ring-pattern CPU tests (tests/test_waymo_ring_pattern.py) at the live checkout;
#   3. the A3 config resolved here against what 27807 and 27812 actually trained with (their W&B config.yaml).
# No GPU. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN="singularity exec --bind /home/koyama/data/:/storage $SIF"
echo "=== 1. eval-view check ==="
$RUN python analysis/a3_ringpattern_eval_view_check.py --frames 60 2>&1 | /usr/bin/grep -v " is loaded"
echo "exit $?"
echo "=== 2. ring-pattern CPU tests ==="
$RUN bash -c "cd /home/koyama/code/ST3D && python -m pytest -q tests/test_waymo_ring_pattern.py 2>&1 | tail -5"
echo "=== 3a. A3 config vs 27807's resolved config (W&B 87h61vfd) ==="
$RUN python analysis/resolved_cfg_vs_wandb.py cfgs/da-ieee-access/centerpoint-ringpattern-st3d-waymo2nuscenes.yaml \
  /home/koyama/data/wandb/run-20261008_212958-87h61vfd/files/config.yaml 2>&1 | /usr/bin/grep -v " is loaded"
echo "=== 3b. A3 config vs 27812's resolved config (W&B pi8c395l) ==="
$RUN python analysis/resolved_cfg_vs_wandb.py cfgs/da-ieee-access/centerpoint-ringpattern-st3d-waymo2nuscenes.yaml \
  /home/koyama/data/wandb/run-20261008_213837-pi8c395l/files/config.yaml 2>&1 | /usr/bin/grep -v " is loaded"
echo "=== done ==="
