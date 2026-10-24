#!/usr/bin/env bash
# SCENE, pre-launch smoke of UADA3D at GRL 1.0 on S->B (experiments_md 20261022_06 §10, row 3). One GPU, ~5 min.
#   (1) CPU unit tests of the conditional GRL and the DANN paths
#   (2) analysis/uada3d_grad_smoke.py on the real DACenterPoint: discriminator losses / accuracy, both upstream gradient norms,
#       GRL sign (cosine -1 against the bypassed gradient), discriminator not reversed, target batch unlabelled and unassigned
#   (3) analysis/target_dump.py on the row's SOURCE (DATA_CONFIG, 20 frames): the detection targets (no pseudo-labels in this
#       row; the target is UNSUPERVISED and assigned nothing, checked in (2))
# adaptive_train.py has no --use_subset; the real loop (the one 26390 / 27174 ran) is checked by the row's first epoch.
# A bash script job, never a --wrap. Runs from the LIVE checkout. Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
echo "=== live HEAD $(git -C /home/koyama/code/ST3D rev-parse --short HEAD) on $(hostname) at $(date '+%F %T') ==="
CFG=cfgs/da-ieee-access/centerpoint-uada3d-g1-nuscenes-singapore2boston.yaml
CK=/storage/wandb/run-20261022_225659-6iu7h6p5/files/ckpt/checkpoint_epoch_20.pth   # S->S 30189 epoch 20 (common init)
X="singularity exec --nv --bind /home/koyama/data/:/storage /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif"
RC=0
echo "##### (1) unit tests"
(cd /home/koyama/code/ST3D && $X python3 -m pytest -q -p no:warnings tests/test_conditional_grl.py tests/test_dann_grl_coeff.py tests/test_centerpoint_dann.py)
R=$?; echo "unit tests exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (2) gradient smoke on the real DACenterPoint"
$X python3 analysis/uada3d_grad_smoke.py "$CFG" "$CK" 3
R=$?; echo "grad smoke exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "##### (3) target dump of the source (DATA_CONFIG, 20 frames)"
$X python3 analysis/target_dump.py "$CFG" DATA_CONFIG 20
R=$?; echo "target dump exit $R"; [ "$R" -eq 0 ] || RC=$R
echo "=== done $(date '+%F %T') rc=$RC ==="
exit "$RC"
