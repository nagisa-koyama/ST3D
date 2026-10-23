#!/usr/bin/env bash
# Pre-launch smoke of the Lyft 40-beam rows P / C1-analog (experiments_md 20261024_01 §5). CPU only, no W&B.
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=8 --mem=64G \
#     --time=03:00:00 --job-name=lyft40smoke --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     scripts/_lyft40_smoke.sh
set -euo pipefail
OUT=/home/koyama/data/lyft_rings/v1_40beam_smoke
P=cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml
C=cfgs/da-ieee-access/centerpoint-global-recount-lyft40-lyft2nuscenes.yaml
mkdir -p "$OUT"
cd /home/koyama/code/ST3D/tools
SING=(singularity exec --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif)
echo "=== $(date '+%Y-%m-%d %H:%M %Z') on $(hostname), ST3D $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) ==="
echo "=== 1. pattern / density / boxes check ==="
"${SING[@]}" python3 analysis/lyft_ring_pattern_check.py "$P" "$C" 300 "$OUT/check.json"
echo "=== 2. target dump, P ==="
"${SING[@]}" python3 analysis/target_dump.py "$P" LYFT_40BEAM 200
echo "=== 3. target dump, C1-analog (calibration installed, 1000 frames) ==="
TARGET_DUMP_CALIB_FRAMES=1000 "${SING[@]}" python3 analysis/target_dump.py "$C" LYFT_40BEAM 200
echo "=== 4. emptied_boxes.py, P (processor stage) ==="
"${SING[@]}" python3 analysis/emptied_boxes.py "$P" LYFT_40BEAM 200
echo "=== done $(date '+%Y-%m-%d %H:%M %Z') ==="
