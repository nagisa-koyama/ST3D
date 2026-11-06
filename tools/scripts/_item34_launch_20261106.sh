#!/bin/bash
# experiments_md 20261003_05 §5 item 34 (design fixed 2026-11-06 12:27 JST; build ST3D cf8ef19, report 20261106_01):
# first wave of the factor swaps on Pandar64 / Lyft-64 / Lyft-40. ANALYSIS (each target's TRAIN labels train the rows;
# val labels only score). Submits, once:
#   - six training rows (the 12-job cap with item 33's six running): O_L64, Lyft-64 D, O_L40, PandaSet R5 / R6 / R7;
#   - a CPU successor (afterany on R5 / R6 / R7) that rescores their in-run epoch-115 results under rule A;
#   - E1, one 1-GPU eval-only job (exempt): O_P 26389 on spin val (reproduces 81.67), PandaSet P 28034 / 28279 and
#     C1-analog 28041 / 28280 on spin val, the nuScenes oracle 25743 on spin val (the control), O_P on background-1/2
#     spin val (R9's matched cut); and its CPU rule-A successor (afterok);
#   - E2, one 1-GPU eval-only job (exempt): Lyft-40 P 30312 / 30363 and C1-analog 30313 / 30364 on Lyft-40 val, the
#     nuScenes oracle 25743 on Lyft-64 val (SHIFT_COOR 1.69, rule G) and Lyft-40 val (1.64, rule G).
# The rest of item 34 (Lyft-40 R6 / R7, Lyft-64 R6 / R7, then the three R9 rows with their background-1/2 twins) goes in
# as slots free; the queue is recorded in T05 item 34d.
#
# Run once from anywhere: bash scripts/_item34_launch_20261106.sh
set -euo pipefail
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
A=cfgs/da-ieee-access-analysis
FP=$A/factorswap-pandaset
F64=$A/factorswap-lyft64
F40=$A/factorswap-lyft40
ET=20261106_fs34
RS=/home/koyama/data/factorswap_pandaset/rescore
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/code/ST3D/output
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }
need() { for f in "$@"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done; }
mkdir -p "$RS"

OP=$W/run-20260928_054827-u2i837pw/files/ckpt/checkpoint_epoch_115.pth
P1=$W/run-20261010_193142-lf9c36yn/files/ckpt/checkpoint_epoch_115.pth
P2=$W/run-20261013_020122-ip52dd1a/files/ckpt/checkpoint_epoch_115.pth
C1=$W/run-20261010_200333-673dtnza/files/ckpt/checkpoint_epoch_115.pth
C2=$W/run-20261013_020220-zqumnro8/files/ckpt/checkpoint_epoch_115.pth
LP1=$W/run-20261024_032320-wvzuo8eu/files/ckpt/checkpoint_epoch_30.pth
LP2=$W/run-20261024_132413-h4p4rgmz/files/ckpt/checkpoint_epoch_30.pth
LC1=$W/run-20261024_032300-ttnfpw95/files/ckpt/checkpoint_epoch_30.pth
LC2=$W/run-20261024_132414-h5k0kery/files/ckpt/checkpoint_epoch_30.pth
NUS=$W/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth
need $OP $P1 $P2 $C1 $C2 $LP1 $LP2 $LC1 $LC2 $NUS
need $F64/centerpoint-factorswap-lyft64-oracle.yaml $F64/centerpoint-factorswap-lyft64-d-density.yaml \
     $F40/centerpoint-factorswap-lyft40-oracle.yaml $FP/centerpoint-factorswap-pandaset-r5-pitch.yaml \
     $FP/centerpoint-factorswap-pandaset-r6-labelsize.yaml $FP/centerpoint-factorswap-pandaset-r7-labelrange.yaml \
     cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml $FP/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml \
     $FP/centerpoint-factorswap-pandaset-evalonly-c1-on-spin.yaml \
     $FP/centerpoint-factorswap-pandaset-evalonly-nuscenes-oracle-on-pandaset.yaml \
     $FP/centerpoint-factorswap-pandaset-evalbg050.yaml $F40/centerpoint-factorswap-lyft40-evalonly-p-on-lyft40.yaml \
     $F40/centerpoint-factorswap-lyft40-evalonly-c1-on-lyft40.yaml \
     $F64/centerpoint-factorswap-lyft64-evalonly-nuscenes-oracle-on-lyft64.yaml \
     $F40/centerpoint-factorswap-lyft40-evalonly-nuscenes-oracle-on-lyft40.yaml

H="Item 34 (T05 §5, design fixed 2026-11-06 12:27; code ST3D cf8ef19; report 20261106_01), labelled ANALYSIS"
train() {  # name est_h gres cpus mem time cfg tag note
  local id
  id=$(EST_H=$2 WANDB_NOTES="$H: $9" scripts/submit.sh --gres=gpu:$3 --cpus-per-task=$4 --mem=$5 --time=$6 \
       --exclude=node03 --job-name=$1 scripts/run_sourceonly_2gpu.sh $7 $8 < /dev/null | tee /dev/stderr | jid)
  [ -n "$id" ] || { echo "submission of $1 failed" >&2; exit 3; }
  echo "$id"
}
L64O=$(train fsL64o 6 1 6 96G 12:00:00 $F64/centerpoint-factorswap-lyft64-oracle.yaml 20261106_fs_lyft64_oracle \
  "(Lyft TRAIN labels) O_L64: the Lyft oracle recipe 25769 on the 64-beam platform (LIDAR_CONFIG 64, SHIFT_COOR 2.04, 30 epochs = 17,010 steps), train and eval on Lyft-64; every Lyft-64 term is read against it.")
L64D=$(train fsL64d 6 1 6 96G 12:00:00 $F64/centerpoint-factorswap-lyft64-d-density.yaml 20261106_fs_lyft64_d \
  "(Lyft TRAIN labels) Lyft-64 D: O_L64 + the Lyft-40 C1-analog's operators (per-range-bin density to nuScenes TRAIN from fixed histograms, CPU job 31957, kept share 0.232, + recount). Density term = O_L64 - D; prediction for G64: '+'; read if >= 3 AP.")
L40O=$(train fsL40o 16 1 6 96G 1-02:00:00 $F40/centerpoint-factorswap-lyft40-oracle.yaml 20261106_fs_lyft40_oracle \
  "(Lyft TRAIN labels) O_L40: the Lyft-40 C1-analog 30313 without its density sampler (kept-frame list, recount, 30 epochs), eval on Lyft-40 val; density = O_L40 - C1 and pattern = C1 - P from 30312 / 30313 / 30363 / 30364 rescored on Lyft-40 val.")
PR5=$(train fsPr5 25 2 20 150G 24:00:00 $FP/centerpoint-factorswap-pandaset-r5-pitch.yaml 20261106_fs_pandaset_r5 \
  "(PandaSet TRAIN labels) Pandar64 R5 pitch: points and boxes pitched +2.02 / +1.08 deg (nuScenes Boston / Singapore forward slope minus PandaSet's +0.27) at p 0.5579 about [0, 0, 1.88], spin val in-run, rule A by a CPU successor. Term = O_P 26389 (81.67, rule A) - R5; prediction for G64: n.d.; read if >= 3 AP.")
PR6=$(train fsPr6 25 2 20 150G 24:00:00 $FP/centerpoint-factorswap-pandaset-r6-labelsize.yaml 20261106_fs_pandaset_r6 \
  "(PandaSet TRAIN labels) Pandar64 R6 label size: Car x1.0135, Pedestrian x0.8833 (nuScenes / PandaSet TRAIN median isotropic scale), bottom kept. Term = O_P - R6, rule A; no prediction (size is MIXED between KITTI and Waymo); read if >= 3 AP.")
PR7=$(train fsPr7 25 2 20 150G 24:00:00 $FP/centerpoint-factorswap-pandaset-r7-labelrange.yaml 20261106_fs_pandaset_r7 \
  "(PandaSet TRAIN labels) Pandar64 R7 label range: Car boxes set to IGNORE per 10 m bin to nuScenes' supervised range histogram (kept share 0.448). Term = O_P - R7, rule A; no prediction (range is MIXED); read if >= 3 AP.")

PRS=$(WANDB_NOTES="CPU only, no W&B: item 34a (T05 §5) rule-A / as-scored rescoring of the Pandar64 factor-swap rows R5 $PR5, R6 $PR6, R7 $PR7 (in-run epoch-115 results on spin val, 360 deg)" \
  scripts/submit.sh --dependency=afterany:$PR5:$PR6:$PR7 --kill-on-invalid-dep=yes \
  --partition=a100,a6000_ada --cpus-per-task=4 --mem=24G --time=02:00:00 --job-name=i34Prs \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="bash analysis/pandaset_rule_a_of_jobs.sh $RS/rescore_34a_r5r6r7.txt 'R5 pitch=$PR5:115:0:false' 'R6 label size=$PR6:115:0:false' 'R7 label range=$PR7:115:0:false'" \
  < /dev/null | tee /dev/stderr | jid)
[ -n "$PRS" ] || { echo "submission of i34Prs failed" >&2; exit 3; }

# eval-only jobs: "cfg ckpt tag" triples, one after another on one GPU
evaljob() {  # name time note triples...
  local name=$1 t=$2 note=$3; shift 3
  local body="F=0" id
  while [ $# -gt 0 ]; do body="$body; bash analysis/eval_checkpoint.sh $1 $2 $3 $ET || F=\$((F+1))"; shift 3; done
  body="$body; echo item34 eval failures \$F; [ \$F -eq 0 ]"
  id=$(WANDB_NOTES="$H. Eval-only, one config after another: $note" \
    scripts/submit.sh --partition=a100,a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=$t \
    --job-name=$name --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt --wrap="$body" \
    < /dev/null | tee /dev/stderr | jid)
  [ -n "$id" ] || { echo "submission of $name failed" >&2; exit 3; }
  echo "$id"
}
E1=$(evaljob i34E1 05:00:00 "PandaSet spin val: O_P 26389 (reproduces 81.67), P 28034 / 28279 and C1-analog 28041 / 28280 at epoch 115 (density = O_P - C1, pattern = C1 - P), the nuScenes oracle 25743 epoch 20 at SHIFT_COOR 0.30 (the control), O_P on background-1/2 val (R9's matched cut); rule A by the CPU successor." \
  cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml $OP fs34_OP \
  $FP/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml $P1 fs34_P_28034 \
  $FP/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml $P2 fs34_P_28279 \
  $FP/centerpoint-factorswap-pandaset-evalonly-c1-on-spin.yaml $C1 fs34_C1_28041 \
  $FP/centerpoint-factorswap-pandaset-evalonly-c1-on-spin.yaml $C2 fs34_C1_28280 \
  $FP/centerpoint-factorswap-pandaset-evalonly-nuscenes-oracle-on-pandaset.yaml $NUS fs34_ctl_25743 \
  $FP/centerpoint-factorswap-pandaset-evalbg050.yaml $OP fs34_OP_bg050)

row() {  # label cfg epoch tag
  local s d; s=$(basename "$2" .yaml); d=$(dirname "${2#cfgs/}")
  printf '["%s", "%s/%s/%s/%s/eval/epoch_%s/val/%s/result.pkl", 0, false]' "$1" "$OUT" "$d" "$s" "$ET" "$3" "$4"
}
{
  echo "["
  row "O_P 26389 (E1 $E1)" cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml 115 fs34_OP; echo ","
  row "P 28034 on spin" $FP/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml 115 fs34_P_28034; echo ","
  row "P 28279 on spin" $FP/centerpoint-factorswap-pandaset-evalonly-p-on-spin.yaml 115 fs34_P_28279; echo ","
  row "C1 28041 on spin" $FP/centerpoint-factorswap-pandaset-evalonly-c1-on-spin.yaml 115 fs34_C1_28041; echo ","
  row "C1 28280 on spin" $FP/centerpoint-factorswap-pandaset-evalonly-c1-on-spin.yaml 115 fs34_C1_28280; echo ","
  row "control 25743 on spin" $FP/centerpoint-factorswap-pandaset-evalonly-nuscenes-oracle-on-pandaset.yaml 20 fs34_ctl_25743; echo ","
  row "O_P 26389 on bg-1/2 spin" $FP/centerpoint-factorswap-pandaset-evalbg050.yaml 115 fs34_OP_bg050; echo
  echo "]"
} > $RS/rows_34_E1.json
python3 -c "import json; r = json.load(open('$RS/rows_34_E1.json')); print(len(r), 'rows')"
E1R=$(WANDB_NOTES="CPU only, no W&B: item 34a (T05 §5) rule-A / as-scored rescoring of eval job $E1 (O_P, P, C1, the control and O_P on background-1/2, spin val)" \
  scripts/submit.sh --dependency=afterok:$E1 --kill-on-invalid-dep=yes \
  --partition=a100,a6000_ada --cpus-per-task=4 --mem=24G --time=02:00:00 --job-name=i34E1rs \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG python3 analysis/pandaset_eval_protocols.py --rows_json $RS/rows_34_E1.json --protocols as_scored rule_A > $RS/rescore_34_E1.txt" \
  < /dev/null | tee /dev/stderr | jid)

E2=$(evaljob i34E2 04:00:00 "Lyft: P 30312 / 30363 and C1-analog 30313 / 30364 on Lyft-40 val at epoch 30 (density = O_L40 - C1, pattern = C1 - P), the nuScenes oracle 25743 epoch 20 on Lyft-64 val (SHIFT_COOR 1.69) and Lyft-40 val (1.64), rule G label-free heights (the controls)." \
  $F40/centerpoint-factorswap-lyft40-evalonly-p-on-lyft40.yaml $LP1 fs34_P_30312 \
  $F40/centerpoint-factorswap-lyft40-evalonly-p-on-lyft40.yaml $LP2 fs34_P_30363 \
  $F40/centerpoint-factorswap-lyft40-evalonly-c1-on-lyft40.yaml $LC1 fs34_C1_30313 \
  $F40/centerpoint-factorswap-lyft40-evalonly-c1-on-lyft40.yaml $LC2 fs34_C1_30364 \
  $F64/centerpoint-factorswap-lyft64-evalonly-nuscenes-oracle-on-lyft64.yaml $NUS fs34_ctl_25743 \
  $F40/centerpoint-factorswap-lyft40-evalonly-nuscenes-oracle-on-lyft40.yaml $NUS fs34_ctl_25743)

echo "item34 wave 1: O_L64 $L64O D $L64D O_L40 $L40O PS R5 $PR5 R6 $PR6 R7 $PR7 rescore $PRS | E1 $E1 rescore $E1R | E2 $E2"
