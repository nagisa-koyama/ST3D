#!/bin/bash
# experiments_md 20261003_05 §5 item 30 (design fixed 2026-11-03 15:50 JST, before any number): RANDOM-CUT REDRAWS of
# the contribution-1 D = AP(lines halved) - AP(count-matched random cut) on the four CenterPoint target oracles. The
# lines cut is deterministic; only the random arm is a draw. Draw 0 is the number on record; this script SUBMITS draws
# 1 and 2 of each oracle's random arm as eight 1-GPU eval-only jobs (analysis/eval_checkpoint.sh), each with the
# original random arm's checkpoint, eval script, extra_tag and config, plus only DRAW (cfgs/.../redraw/*-draw<k>.yaml,
# which resolve to the original + DATA_CONFIG_TAR.<block>.DRAW). PandaSet is read under rule A, so a CPU rescoring
# (pandaset_eval_protocols.py, as job 28110 did) follows its two evaluations as a Slurm dependency.
#
#   oracle (job)        original random arm (job, W&B)          rows arm W&B   checkpoint                       eval ~min
#   Waymo 26398         rowcut/...-toprandom (27953, pyq4bjea)  7i382qrf       run-20260928_082639-y6p61n0x ep 7     12-55
#   nuScenes 25743      sensitivity/...-ringrandom (28024, izqfu3pl) n2ats91a  run-20260922_155434-wfdcq75s ep 20     4
#   KITTI 26843         sensitivity/...-ringrandom2 (28094, 0vis22l4) m0o4weoh run-20261001_005323-06iwywhs ep 152    4
#   PandaSet spin 26389 sensitivity-pandaset/...-ringrandom (28095, e1nkih3c) 83on89ls run-20260928_054827-u2i837pw ep 115 3
#
# ANALYSIS (each oracle's own val labels only score). Run from anywhere when GPUs are free: bash scripts/_item30_...sh
# It refuses to submit if any of its eight outputs already exists (FORCE=1 overrides).
set -euo pipefail
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
R=cfgs/da-ieee-access-analysis/redraw
OUT=/home/koyama/code/ST3D/output/da-ieee-access-analysis/redraw
GPU=(--partition=a6000_ada,a6000,rtx8000 --gres=gpu:1 --cpus-per-task=8 --mem=64G)
jid() { sed -n 's/^submitted job \([0-9]*\):.*/\1/p'; }

#     key  config stem                                             checkpoint                                                    extra_tag               epoch time      original
ARMS="W    centerpoint-sourceonly-waymo2waymo-toprandom              $W/run-20260928_082639-y6p61n0x/files/ckpt/checkpoint_epoch_7.pth   20260922_sourceonly     7   02:00:00 Waymo_oracle_26398:_job_27953_W&B_pyq4bjea,_rows_arm_7i382qrf
N    centerpoint-sourceonly-nuscenes-ringrandom                $W/run-20260922_155434-wfdcq75s/files/ckpt/checkpoint_epoch_20.pth  20260922_sourceonly     20  00:45:00 nuScenes_oracle_25743:_job_28024_W&B_izqfu3pl,_rows_arm_n2ats91a
K    centerpoint-sourceonly-kitti2kitti-noros-zshift-ringrandom2 $W/run-20261001_005323-06iwywhs/files/ckpt/checkpoint_epoch_152.pth 20261011_kitti_sens     152 00:45:00 KITTI_oracle_26843:_job_28094_W&B_0vis22l4,_rows_arm_m0o4weoh
P    centerpoint-pandaset-spin2spin-ringrandom                 $W/run-20260928_054827-u2i837pw/files/ckpt/checkpoint_epoch_115.pth 20261011_pandaset_sens  115 00:45:00 PandaSet_Pandar64_spin_oracle_26389:_job_28095_W&B_e1nkih3c,_rows_arm_83on89ls_(rule_A)"

# pre-flight: every config and checkpoint on disk (HOST paths), no output overwritten
while read -r key stem ckpt etag ep t orig; do
  [ -f "$ckpt" ] || { echo "missing checkpoint $ckpt" >&2; exit 2; }
  for k in 1 2; do
    [ -f "$R/$stem-draw$k.yaml" ] || { echo "missing config $R/$stem-draw$k.yaml" >&2; exit 2; }
    res=$OUT/$stem-draw$k/$etag/eval/epoch_$ep/val/item30_${key}_draw$k/result.pkl
    if [ -e "$res" ] && [ "${FORCE:-0}" != 1 ]; then echo "exists, not overwriting: $res" >&2; exit 2; fi
  done
done <<< "$ARMS"

declare -A JOBS
while read -r key stem ckpt etag ep t orig; do
  for k in 1 2; do
    note="Item 30 (20261003_05 §5) random-cut redraw: draw $k of the lines-halved count-matched random arm, ${orig//_/ }; draw 0 is that original job; eval-only, same checkpoint, eval script and config except DRAW $k; analysis."
    j=$(WANDB_NOTES="$note" scripts/submit.sh "${GPU[@]}" --time=$t --job-name=i30${key}$k \
          --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
          --wrap="bash analysis/eval_checkpoint.sh $R/$stem-draw$k.yaml $ckpt item30_${key}_draw$k $etag" \
          < /dev/null | tee /dev/stderr | jid)
    [ -n "$j" ] || { echo "submission of $key draw $k failed" >&2; exit 3; }
    JOBS[$key$k]=$j
  done
done <<< "$ARMS"

# PandaSet: rule-A (and as-scored) rescoring of rows, draw 0 (both on record: 74.52 / 75.10 rule A BEV, must reproduce)
# and draws 1, 2 on CPU, once both PandaSet evaluations have succeeded.
PS=/home/koyama/data/pandaset_sens
SENS=/home/koyama/code/ST3D/output/da-ieee-access-analysis/sensitivity-pandaset
P=centerpoint-pandaset-spin2spin
cat > $PS/rows_item30_redraw.json <<EOF
[["psSens_spin_ringrows (rows arm, on record)", "$SENS/$P-ringrows/20261011_pandaset_sens/eval/epoch_115/val/psSens_spin_ringrows/result.pkl", 0, false],
 ["psSens_spin_ringrandom (draw 0, on record)", "$SENS/$P-ringrandom/20261011_pandaset_sens/eval/epoch_115/val/psSens_spin_ringrandom/result.pkl", 0, false],
 ["item30_P_draw1 (job ${JOBS[P1]})", "$OUT/$P-ringrandom-draw1/20261011_pandaset_sens/eval/epoch_115/val/item30_P_draw1/result.pkl", 0, false],
 ["item30_P_draw2 (job ${JOBS[P2]})", "$OUT/$P-ringrandom-draw2/20261011_pandaset_sens/eval/epoch_115/val/item30_P_draw2/result.pkl", 0, false]]
EOF
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
WANDB_NOTES="CPU only, no W&B: item 30 (20261003_05 §5) rule-A / as-scored rescoring of the PandaSet spin oracle's lines-random redraws (jobs ${JOBS[P1]}, ${JOBS[P2]}) beside the rows arm and draw 0 on record (28095, rule A as 28110)" \
  scripts/submit.sh --dependency=afterok:${JOBS[P1]}:${JOBS[P2]} --kill-on-invalid-dep=yes \
  --partition=a100,a6000_ada --cpus-per-task=4 --mem=24G --time=01:00:00 --job-name=i30Prs \
  --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
  --wrap="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG python3 analysis/pandaset_eval_protocols.py --rows_json $PS/rows_item30_redraw.json --protocols as_scored rule_A > $PS/rescore_item30_redraw.txt"

echo "eval jobs: W ${JOBS[W1]} ${JOBS[W2]}  N ${JOBS[N1]} ${JOBS[N2]}  K ${JOBS[K1]} ${JOBS[K2]}  P ${JOBS[P1]} ${JOBS[P2]}"
echo "readout: Car BEV AP_R40 @0.7 from each eval log (KITTI: moderate); PandaSet rule A from $PS/rescore_item30_redraw.txt"
