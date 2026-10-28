#!/usr/bin/env bash
# O_f, the frame-matched Waymo oracle twin (ANALYSIS; experiments_md 20261028_04, design T05 §5 item 23, e9bb63b): one CPU
# job with the pre-launch checks (oraclef_check.py) and a 30-frame assigned-target dump (target_dump.py). No GPU, no W&B.
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G \
#     --time=01:00:00 --job-name=oraclefChk --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt analysis/oraclef_job.sh
set -uo pipefail
cd /home/koyama/code/ST3D/tools
echo "=== code at ST3D $(git -C .. rev-parse --short HEAD) on $(hostname) ==="
git -C .. status --short -- tools/analysis/oraclef_check.py tools/analysis/condoracle_check.py tools/analysis/target_dump.py \
    tools/cfgs/da-ieee-access/centerpoint-oraclef-waymo-interval10.yaml || true
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN=(singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF")
CFG=cfgs/da-ieee-access/centerpoint-oraclef-waymo-interval10.yaml
rc=0
"${RUN[@]}" python3 analysis/oraclef_check.py || rc=1
echo "=== target dump: $CFG ==="
"${RUN[@]}" python3 analysis/target_dump.py "$CFG" DATA_CONFIG 30 || rc=1
echo "=== oraclef_job rc=$rc ==="
exit $rc
