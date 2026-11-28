#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 68 (design fixed 2026-11-28 16:42 JST): KITTI val evaluation of one LN / GN training row
# at epoch 20, with the row's KITTI twin config (centerpoint-sourceonly-nuscenes2kitti.yaml + the same NORM_SWAP). No
# adaptation exists for these layers. Run as the row's 1-GPU successor (afterany), from ST3D/tools.
#   bash scripts/_item68_evals_20261128.sh <train jobid> <ln|gn> <seed>
set -u
cd /home/koyama/code/ST3D/tools
J=${1:?usage: _item68_evals_20261128.sh <train jobid> <ln|gn> <seed>}
V=${2:?ln or gn}
S=${3:?seed}
CFG=cfgs/da-ieee-access-analysis/item68/centerpoint-item68-nuscenes2kitti-$V.yaml
echo "=== item68 KITTI val, $V seed $S, row $J, epoch 20 ($(date '+%F %T')) ==="
C=$(bash analysis/ckpt_of_job.sh $J 20) || { echo "item68: no epoch-20 checkpoint of $J"; exit 2; }
bash analysis/eval_checkpoint.sh $CFG $C i68_kitti_${V}_s$S 20261128_i68
echo "exit $? ($(date '+%F %T'))"
