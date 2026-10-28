#!/usr/bin/env bash
# P2-B condition oracles (ANALYSIS; experiments_md 20261028_03, design T05 §5 af7680c): one CPU job that builds the
# four arms' train frame sets (condoracle_build.py), then runs the pre-launch checks of each arm config
# (condoracle_check.py) and a 30-frame assigned-target dump (target_dump.py). No GPU, no W&B.
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=32G \
#     --time=01:00:00 --job-name=condOrChk --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt analysis/condoracle_job.sh
set -uo pipefail
cd /home/koyama/code/ST3D/tools
echo "=== code at ST3D $(git -C .. rev-parse --short HEAD) on $(hostname) ==="
git -C .. status --short -- tools/analysis/condoracle_build.py tools/analysis/condoracle_check.py tools/cfgs/da-ieee-access/ | /usr/bin/grep condoracle || echo "=== condoracle files clean at HEAD ==="
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN=(singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF")
"${RUN[@]}" python3 analysis/condoracle_build.py || { echo "=== BUILD FAILED ==="; exit 1; }
WL=/home/koyama/data/wandb/run-20260928_082639-y6p61n0x/files/log_train_20260928-082644.txt   # 26398, Waymo oracle training
WE=logs/error_27179_evWa2Wa.txt                                                                 # 27179, its Waymo-val scoring
NL=/home/koyama/data/wandb/run-20260922_155434-wfdcq75s/files/log_train_20260922-155437.txt   # 25743, nuScenes oracle
rc=0
while read -r name key log elog; do
  cfg=cfgs/da-ieee-access/centerpoint-condoracle-$name.yaml
  "${RUN[@]}" python3 analysis/condoracle_check.py "$cfg" "$key" "$log" $elog || rc=1
  echo "=== target dump: $cfg ==="
  "${RUN[@]}" python3 analysis/target_dump.py "$cfg" DATA_CONFIG 30 || rc=1
done <<LIST
waymo-night waymo_night $WL $WE
waymo-day waymo_day $WL $WE
nuscenes-bostonrain nuscenes_boston_rain $NL
nuscenes-bostonclear nuscenes_boston_clear $NL
LIST
echo "=== condoracle_job rc=$rc ==="
exit $rc
