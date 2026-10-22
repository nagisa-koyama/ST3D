#!/usr/bin/env bash
# Check PL-TGTONLY (experiments_md 20261022_08): CPU checks before the four target-only rows T1-T4 reach a GPU.
#   1. pytest: tests/test_pl_tgtonly.py (configs = twin + 4 keys, schedule arithmetic, loop never reads the source)
#      and the existing self-training loop / epoch-length tests.
#   2. resolved config of each row vs the RESOLVED config its twin trained with (W&B config.yaml).
#   3. target dump (bug-prevention rule 4): one batch per frame of each row's target block through the real dataset,
#      collate and CenterHead.assign_targets, with the twin's own epoch-0 pseudo-labels (same frozen teacher, same cuts).
# CPU only, no W&B. Run as a Slurm CPU job from ST3D/tools (never on the master):
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=48G \
#     --time=01:30:00 --job-name=tgtonlyChk --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt \
#     scripts/_pl_tgtonly_checks.sh
set -u
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) $(date '+%F %T') ==="
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN=(singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF")
W=/home/koyama/data/wandb
D=cfgs/da-ieee-access
T1=$D/centerpoint-global-st3d-tgtonly-waymo2nuscenes.yaml
T2=$D/centerpoint-ringpattern-st3d-tgtonly-waymo2nuscenes.yaml
T3=$D/centerpoint-gblobs-accum-legaldepth-st3d-tgtonly-nuscenes2waymo.yaml
T4=$D/centerpoint-st3d-cb-tgtonly-lyft2nuscenes.yaml
# twins: 27812 pi8c395l, 29334 zy379iy1, 27704 5hzcs4e4, 27592 yfe7t7hz
R1=$W/run-20261008_213837-pi8c395l/files
R2=$W/run-20261017_212145-zy379iy1/files
R3=$W/run-20261008_030315-5hzcs4e4/files
R4=$W/run-20261007_063627-yfe7t7hz/files

echo "=== 1. pytest ==="
(cd .. && set -o pipefail && "${RUN[@]}" python3 -m pytest -q -rf tests/test_pl_tgtonly.py tests/test_genuine_st3d.py \
    tests/test_self_train_epoch_length.py tests/test_self_training_schedule_sizing.py 2>&1 | /usr/bin/grep -v "is loaded$" | tail -40)
echo "pytest exit $?"

echo "=== 2. resolved config vs the twin's W&B config.yaml ==="
"${RUN[@]}" python3 analysis/resolved_cfg_vs_wandb.py "$T1" "$R1/config.yaml" "$T2" "$R2/config.yaml" \
    "$T3" "$R3/config.yaml" "$T4" "$R4/config.yaml"
echo "resolved_cfg_vs_wandb exit $?"

echo "=== 3. target dumps (twin's ps_label_e0.pkl) ==="
for pair in "$T1:$R1" "$T4:$R4" "$T2:$R2" "$T3:$R3"; do
    CFG=${pair%%:*}; RD=${pair#*:}
    "${RUN[@]}" python3 analysis/target_dump.py "$CFG" TAR 12 "$RD/ps_label/ps_label_e0.pkl" 2>&1 \
        | /usr/bin/grep -vE "is loaded$|INFO" | tail -12
    echo "target_dump $(basename "$CFG") exit ${PIPESTATUS[0]}"
done
echo "=== done $(date '+%F %T') ==="
