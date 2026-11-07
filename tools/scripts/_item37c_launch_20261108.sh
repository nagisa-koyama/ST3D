#!/bin/bash
# experiments_md 20261003_05 §5 item 37c (design fixed 2026-11-08 01:40 JST, before any number): the KITTI pattern row P_K
# and its count-matched control L_K, seeds 666 / 667, on the oracle 26843's recipe (1 GPU x 6, 152 epochs, in-run eval on
# unperturbed KITTI val). ANALYSIS (KITTI TRAIN labels).
#   P_K: TRAIN_FACTOR_SWAP RING_THIN MODE pattern (1.33 deg lattice, 0.332 deg step); its checks passed in CPU job 32279.
#   L_K: RANGE_KEEP at P_K's in-grid per-bin share (kitti_pattern_count_rates.py, job 32278). Before its rows: the item-37c
#        check again with L_K (box rule, eval inertness) and the gate arm_point_count_bins.py P_K vs L_K, 300 frames,
#        which must print "points: ... -> MATCHED" (afterok).
# Reading (T05 item 37): structure beyond count = L_K - P_K, count part = R3 - L_K, pattern = R3 - P_K, paired by seed.
# Run once: bash scripts/_item37c_launch_20261108.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
APC=/home/koyama/data/arm_point_counts
S2=cfgs/da-ieee-access-analysis/factorswap-s2
PK=$S2/centerpoint-factorswap-s2-p-pattern.yaml
LK=$S2/centerpoint-factorswap-s2-l-latcount.yaml
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
for f in $PK $LK; do [ -f $f ] || { echo "missing $f" >&2; exit 2; }; done
Q=$(squeue -u koyama -h | wc -l); [ $((Q + 8)) -le 20 ] || { echo "only $((20 - Q)) submissions left, need 8" >&2; exit 2; }
H="Item 37c (T05 §5, design fixed 2026-11-08 01:40 JST), labelled ANALYSIS (KITTI TRAIN labels)"
SI="singularity exec --bind /home/koyama/data/:/storage $IMG"
train() {  # name seed cfg tag run_name dep note
  local extra="" dep=""; [ "$2" = 666 ] || extra="--seed $2"; [ -z "$6" ] || dep="--dependency=afterok:$6 --kill-on-invalid-dep=yes"
  EST_H=13 WANDB_NOTES="$H: $7 seed $2." \
    scripts/submit.sh $dep --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=6 --mem=96G \
    --time=1-06:00:00 --exclude=node03 --job-name=$1 scripts/run_sourceonly_2gpu.sh $3 $4 $5 $extra < /dev/null | jid
}
NP="P_K, KITTI training clouds thinned to nuScenes' scan pattern (rings nearest the 1.33 deg lattice, 0.332 deg azimuth step; kitti_rings MODE pattern, frame-seeded); pattern = R3 - P_K, structure beyond count = L_K - P_K"
P1=$(train fsPK 666 $PK fsS2_p-pattern factorswap-s2-p-pattern "" "$NP")
P2=$(train fsPK667 667 $PK fsS2_p-pattern-seed667 factorswap-s2-p-pattern-seed667 "" "$NP")
[ -n "$P1" ] && [ -n "$P2" ] || { echo "P_K submission failed" >&2; exit 3; }
C=$(WANDB_NOTES="CPU only, no W&B: $H checks with L_K: identity at RATE 1, P_K and L_K box rule, eval inertness (item37c_kitti_pattern_check.py, 48 frames)" \
  scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=32G --time=02:00:00 --job-name=i37cChkL \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="$SI python3 analysis/item37c_kitti_pattern_check.py 48" < /dev/null | jid)
G1=$(WANDB_NOTES="CPU only, no W&B: $H gate: P_K points inside the voxel grid per 5-m bin, 300 frames" \
  scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=apc-k-lat \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="$SI python3 analysis/arm_point_count_bins.py measure $PK 300 $APC/k-lattice.json" < /dev/null | jid)
G2=$(WANDB_NOTES="CPU only, no W&B: $H gate: L_K points inside the voxel grid per 5-m bin, 300 frames" \
  scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G --time=04:00:00 --job-name=apc-k-latcnt \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="$SI python3 analysis/arm_point_count_bins.py measure $LK 300 $APC/k-latcount.json" < /dev/null | jid)
R=$APC/report_k_latcount.md
GC=$(WANDB_NOTES="CPU only, no W&B: $H gate check: report k-lattice vs k-latcount; fails unless 'points' reads MATCHED, so the L_K rows (afterok) never start on DIFFER" \
  scripts/submit.sh --dependency=afterok:$C:$G1:$G2 --kill-on-invalid-dep=yes --partition=a100,a6000_ada --cpus-per-task=1 \
  --mem=4G --time=00:15:00 --job-name=apc-k-check --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="set -e; $SI python3 analysis/arm_point_count_bins.py report $APC/k-lattice.json $APC/k-latcount.json 'KITTI, count-matched L_K vs P_K (item 37c)' > $R; cat $R; /usr/bin/grep -q '^points: .*-> MATCHED\$' $R" \
  < /dev/null | jid)
[ -n "$C" ] && [ -n "$G1" ] && [ -n "$G2" ] && [ -n "$GC" ] || { echo "check / gate submission failed (P_K $P1 $P2 in)" >&2; exit 3; }
NL="L_K, KITTI training clouds thinned at random per 1.5-m bin to P_K's in-grid count (RANGE_KEEP, frame-seeded; gate $GC); structure beyond count = L_K - P_K, count part = R3 - L_K"
L1=$(train fsLK 666 $LK fsS2_l-latcount factorswap-s2-l-latcount $GC "$NL")
L2=$(train fsLK667 667 $LK fsS2_l-latcount-seed667 factorswap-s2-l-latcount-seed667 $GC "$NL")
[ -n "$L1" ] && [ -n "$L2" ] || { echo "L_K submission failed (P_K $P1 $P2, gate $GC in)" >&2; exit 3; }
echo "item37c: P_K $P1 $P2 | check $C gate $G1 $G2 -> $GC | L_K $L1 $L2"
