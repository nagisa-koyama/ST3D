#!/usr/bin/env bash
# Item 53 D2 (experiments_md 20261003_05 §5; ANALYSIS): IA-SSD A1 (b2, camera-FOV-cropped KITTI target, epoch 20) with BN
# statistics re-estimated on target TRAIN clouds (1000 frames, mix 1, as E1c 29318) for six layer groups, plus full AdaBN
# again as the procedure check. New extra tags i53_s{1,2}, one --layers_name per group, so nothing of b2_* is overwritten.
# Then analysis/item53_d2_read.py applies the fixed reading. Usage (1 GPU): bash scripts/_item53_d2_evals.sh
# Do not edit while a job runs it.
set -u
cd /home/koyama/code/ST3D/tools
CFG=cfgs/da-ieee-access-pointrcnn/iassd-sourceonly-nuscenes2kitti-fovtar.yaml
declare -A CKPT=([1]=/home/koyama/data/wandb/run-20261006_125447-z7im8onl/files/ckpt/checkpoint_epoch_20.pth
                 [2]=/home/koyama/data/wandb/run-20261006_200105-pq6z15my/files/ckpt/checkpoint_epoch_20.pth)
GROUPS=(bb head nobox conf noconf novote)
declare -A RE=([bb]='^backbone_3d\.' [head]='^point_head\.' [nobox]='^(?!point_head\.box_center_layers)'
               [conf]='confidence_layers' [noconf]='^(?!.*confidence_layers)' [novote]='^(?!backbone_3d\.SA_modules\.4\.)')
for s in 1 2; do
  bash analysis/adabn_eval.sh $CFG ${CKPT[$s]} target 1.0 i53_D2_s${s}_full --extra_tag i53_s$s
  echo "i53 s$s full exit $?"
  for g in "${GROUPS[@]}"; do
    bash analysis/adabn_eval.sh $CFG ${CKPT[$s]} target 1.0 i53_D2_s${s}_$g --layers "${RE[$g]}" --layers_name $g --extra_tag i53_s$s
    echo "i53 s$s $g exit $?"
  done
done
singularity exec /home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif python3 analysis/item53_d2_read.py \
  --json /home/koyama/data/item53/d2_read.json
