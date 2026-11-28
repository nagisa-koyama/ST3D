#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 69 (design fixed 2026-11-28 17:51 JST): the 1-GPU evaluation successor of one G_r row.
# G_r with its own statistics (eval_checkpoint.sh, the control's KITTI config), then the G_r bitwise check and 59a's hybrids
# (item69_real_bn.py make), then AdaBN (adabn_eval.sh target 1.0, as 41a's B1) for rs1 (41a's own checkpoint) and the hybrids.
#   bash scripts/_item69_evals_20261128.sh <G_r jobid> <seed>
set -u
cd /home/koyama/code/ST3D/tools
J=${1:?usage: _item69_evals_20261128.sh <G_r jobid> <seed>}
S=${2:?seed}
CTRL=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes2kitti.yaml
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
py() { singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" python3 "$@" < /dev/null; }
step() { echo "=== item69 $* ($(date '+%F %T')) ==="; }
mkdir -p /home/koyama/data/item69
CG=$(bash analysis/ckpt_of_job.sh $J 20) || { echo "item69: no epoch-20 checkpoint of $J"; exit 2; }
step "G_r own statistics, seed $S, $CG"
bash analysis/eval_checkpoint.sh $CTRL $CG i69_G_r_s$S 20261128_i69
echo "exit $?"
step "G_r check and hybrids, seed $S"
py analysis/item69_real_bn.py make $S $J || { echo "item69: check or hybrids failed; no AdaBN arm run"; exit 3; }
for A in rs1 aff_all aff_3d aff_2d; do
  C=$(py analysis/item69_real_bn.py ckpt $A $S) || { echo "item69: no checkpoint for $A"; continue; }
  step "AdaBN $A, seed $S, $C"
  bash analysis/adabn_eval.sh $CTRL $C target 1.0 i69_${A}_s$S --extra_tag i69_${A}_s$S --eval_tag i69_${A}_s$S
  echo "exit $?"
done
step "done"
