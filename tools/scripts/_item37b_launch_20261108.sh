#!/bin/bash
# experiments_md 20261003_05 §5 item 37b (design fixed 2026-11-08 01:40 JST, before any number): the Lyft-40 count-matched
# control L_L40, seeds 666 / 667. ANALYSIS (read on Lyft-40 val labels; trains on Lyft source labels only).
#   gate: arm_point_count_bins.py measure of the lattice arm P (ringpattern-lyft40) and of L_L40, 300 frames each; the
#         report must print "points: ... -> MATCHED", else the training rows never start (afterok);
#   rows: L_L40 = the C1-analog 30313 recipe (30 epochs, 1 GPU x 6) thinned to P's in-grid per-bin counts;
#   eval: each row's epoch 30 on Lyft-40 val through centerpoint-factorswap-lyft40-evalonly-l-on-lyft40.yaml (afterany),
#         as P and C1-analog were scored (item 34 E2).
# Reading (T05 item 37): structure beyond count = AP(L) - AP(P) (P 30312 / 30363 on Lyft-40 val 44.81 / 37.94), count part
# = AP(C1) - AP(L) (C1 30313 / 30364: 64.02 / 64.82), paired by seed; read at two seeds if the mean is >= 2, same sign.
# Run once: bash scripts/_item37b_launch_20261108.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
APC=/home/koyama/data/arm_point_counts
P_CFG=cfgs/da-ieee-access/centerpoint-ringpattern-lyft40-lyft2nuscenes.yaml
L_CFG=cfgs/da-ieee-access/centerpoint-global-recount-latcount-lyft40-lyft2nuscenes.yaml
TWIN=cfgs/da-ieee-access-analysis/factorswap-lyft40/centerpoint-factorswap-lyft40-evalonly-l-on-lyft40.yaml
ET=20261108_fs37
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
for f in $P_CFG $L_CFG $TWIN; do [ -f $f ] || { echo "missing $f" >&2; exit 2; }; done
Q=$(squeue -u koyama -h | wc -l); [ $((Q + 8)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need 8" >&2; exit 2; }
H="Item 37b (T05 §5, design fixed 2026-11-08 01:40 JST), labelled ANALYSIS"
SI="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG"
G1=$(WANDB_NOTES="CPU only, no W&B: $H gate: Lyft-40 lattice arm P (ringpattern-lyft40) points inside the voxel grid per 5-m bin, 300 frames" \
  scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=apc-l40-lat \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="$SI python3 analysis/arm_point_count_bins.py measure $P_CFG 300 $APC/l40-lattice.json" < /dev/null | jid)
G2=$(WANDB_NOTES="CPU only, no W&B: $H gate: Lyft-40 count-matched control L_L40, points inside the voxel grid per 5-m bin, 300 frames" \
  scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=apc-l40-latcnt \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="$SI python3 analysis/arm_point_count_bins.py measure $L_CFG 300 $APC/l40-latcount.json" < /dev/null | jid)
[ -n "$G1" ] && [ -n "$G2" ] || { echo "gate submission failed" >&2; exit 3; }
R=$APC/report_l40_latcount.md
GC=$(WANDB_NOTES="CPU only, no W&B: $H gate check: report l40-lattice vs l40-latcount; fails unless the decisive 'points' line reads MATCHED, so the L_L40 rows (afterok) never start on DIFFER" \
  scripts/submit.sh --dependency=afterok:$G1:$G2 --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=1 \
  --mem=4G --time=00:15:00 --job-name=apc-l40-check --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="set -e; $SI python3 analysis/arm_point_count_bins.py report $APC/l40-lattice.json $APC/l40-latcount.json 'Lyft-40 -> nuScenes, count-matched L_L40 (item 37b)' > $R; cat $R; /usr/bin/grep -q '^points: .*-> MATCHED\$' $R" \
  < /dev/null | jid)
[ -n "$GC" ] || { echo "gate check submission failed" >&2; exit 3; }
train() {  # name seed tag run_name
  local extra=""; [ "$2" = 666 ] || extra="--seed $2"
  EST_H=6 WANDB_NOTES="$H: Lyft-40 count-matched control L_L40 seed $2 (C1-analog 30313 recipe, per-bin thinning matched to the lattice arm P's in-grid counts; gate $GC). Structure beyond count = L - P, count part = C1 - L, paired by seed, read on Lyft-40 val by eval twin." \
    scripts/submit.sh --dependency=afterok:$GC --kill-on-invalid-dep=yes --partition=a100,a6000_ada,a6000,rtx8000 \
    --gres=gpu:1 --cpus-per-task=6 --mem=96G --time=0-22:00:00 --exclude=node03 --job-name=$1 \
    scripts/run_sourceonly_2gpu.sh $L_CFG $3 $4 $extra < /dev/null | jid
}
T1=$(train l40L 666 20261108_fs37_l40latcount global-recount-latcount-lyft40-lyft2nuscenes)
T2=$(train l40L667 667 20261108_fs37_l40latcount_s667 global-recount-latcount-lyft40-lyft2nuscenes-seed667)
[ -n "$T1" ] && [ -n "$T2" ] || { echo "training submission failed (gate $G1 $G2 $GC)" >&2; exit 3; }
ev() {  # train_job seed
  WANDB_NOTES="$H, eval-only: L_L40 seed $2 ($1) epoch 30 on Lyft-40 val (eval twin evalonly-l-on-lyft40)." \
    scripts/submit.sh --dependency=afterany:$1 --kill-on-invalid-dep=yes --partition=a100,a6000_ada,a6000,rtx8000 \
    --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:30:00 --exclude=node03 --job-name=i37bE$2 \
    --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
    --wrap="set -e; C=\$(bash analysis/ckpt_of_job.sh $1 30); bash analysis/eval_checkpoint.sh $TWIN \$C fs37_LL40_s$2 $ET" \
    < /dev/null | jid
}
E1=$(ev $T1 666); E2=$(ev $T2 667)
echo "item37b: gate $G1 $G2 check $GC | L_L40 666 $T1 eval $E1 | 667 $T2 eval $E2"
