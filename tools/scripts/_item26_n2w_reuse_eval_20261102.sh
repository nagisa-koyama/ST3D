#!/bin/bash
# experiments_md 20261003_05 §5 item 26 (design fixed 2026-11-02 before any number): FACTOR-SWAP N->W, the reused rows.
# The Waymo -> nuScenes training rows P (scan pattern, 27807 / 27840) and C1 (density twin, 27850 / 28278) train the
# Waymo ORACLE recipe (26398; resolved configs differ only in the data operators) on Waymo TRAIN with a nuScenes-side
# factor applied. Scored on UNPERTURBED Waymo val they are factor-swap oracles in the nuScenes -> Waymo direction:
#   C1  TOP only + nuScenes per-range-bin density + recount     P  TOP only + nuScenes scan pattern + recount
# The oracle checkpoint is rescored first as the reproduction check (65.92 as scored by 27179; within +-0.3 BEV).
# The eval config is the oracle's (its DATA_CONFIG_TAR is full Waymo val, all five lidars). ANALYSIS (Waymo val labels
# only score; the training rows are legal source-only rows).
cd /home/koyama/code/ST3D/tools
W=/home/koyama/data/wandb
O_CFG=cfgs/da-ieee-access/centerpoint-sourceonly-waymo2waymo.yaml
BASE_NOTE="Item 26 (20261003_05 §5), FACTOR-SWAP N->W reused rows: epoch-7 checkpoint scored on unperturbed Waymo val with the oracle's eval block; analysis."
ev() {  # label ckpt tag
  echo "##### $1"
  WANDB_NOTES="$BASE_NOTE $1." bash analysis/eval_checkpoint.sh $O_CFG "$2" "$3" 20260922_sourceonly
  echo "eval $3 exit $?"
}
ev "Waymo oracle 26398 (reproduction check)" $W/run-20260928_082639-y6p61n0x/files/ckpt/checkpoint_epoch_7.pth item26_O_26398
ev "C1 density twin seed 666 (27850)" $W/run-20261009_075900-pybgphs0/files/ckpt/checkpoint_epoch_7.pth item26_C1_27850
ev "C1 density twin seed 667 (28278)" $W/run-20261013_020040-n6acyx3p/files/ckpt/checkpoint_epoch_7.pth item26_C1_28278
ev "P scan pattern seed 666 (27807)" $W/run-20261008_212958-87h61vfd/files/ckpt/checkpoint_epoch_7.pth item26_P_27807
ev "P scan pattern seed 667 (27840)" $W/run-20261009_074254-bpertm50/files/ckpt/checkpoint_epoch_7.pth item26_P_27840
echo "##### all done"
