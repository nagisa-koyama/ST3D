#!/usr/bin/env bash
# End-to-end DDP validation: the gap left open by 20260922_06 §7.
#
# Every DDP measurement in that report stopped at the training loop - the longest ran 267
# iterations / ~5 minutes and none reached an epoch boundary. So these remain unverified under DDP:
#   * epoch completion and LR-schedule wrap-up
#   * checkpoint save through the DistributedDataParallel `.module` wrapper
#   * the EVAL path, which calls build_dataloader(..., dist=dist_train) and therefore takes the
#     DistributedSampler branch that has never executed to completion
#   * KITTI-metric AP computation on the nuScenes target under DDP
#
# KITTI is chosen as the smallest source (3712 frames ~ 4 min/epoch at 2 GPUs), so this is the
# cheapest run that exercises all of the above.
#
# --batch_size 12 is the TOTAL across both GPUs (train.py:87-91 divides it), reproducing the
# configs' BATCH_SIZE_PER_GPU: 6 exactly. Omitting it would give 6 PER GPU instead.
#
# NOTE this is a VALIDATION run, not a training run: --epochs 1 makes OneCycle compute
# total_steps for a single epoch, so the resulting AP is meaningless. Only "did it complete
# without crashing" is being tested.
#
# The --extra_tag MUST be fresh on every re-run. repeat_eval_ckpt arbitrates through
# output/<cfg>/<extra_tag>/eval/eval_with_train/eval_list_val.txt, and a tag reused from a
# previous attempt still lists that attempt's claimed epochs - so the run would find nothing
# to evaluate and "pass" without ever entering the eval path this script exists to test.
# Defaults to a timestamp; pass one explicitly to override.
set -u
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
TAG=${1:-$(date +%Y%m%d_%H%M%S)_ddp_e2e}
LOG=analysis/throughput_results/ddp_e2e_validation_kitti_${TAG}.log

PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
echo "=== DDP end-to-end validation: KITTI, 2 GPUs, total bs=12 (6/GPU), 1 epoch + eval ===" 1>&2
echo "=== port $PORT, extra_tag $TAG, log $LOG ===" 1>&2

singularity exec --nv --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D "$SIF" \
    python -m torch.distributed.launch --use-env --nproc_per_node=2 --rdzv_endpoint=localhost:"$PORT" \
    train.py --launcher pytorch --tcp_port "$PORT" \
    --cfg_file cfgs/da-ieee-access/centerpoint-sourceonly-kitti.yaml \
    --batch_size 12 --workers 4 \
    --epochs 1 --num_epochs_to_eval 1 --ckpt_save_interval 1 \
    --run_name "ddp_e2e_validation_kitti_${TAG}" --extra_tag "$TAG" \
    > "$LOG" 2>&1
RC=$?

echo "=== train.py exit code: $RC ===" 1>&2
echo "=== checkpoint(s) written ===" 1>&2
find /home/koyama/data/wandb -name 'checkpoint_epoch_*.pth' -newermt '-2 hours' 2>/dev/null | tail -5 1>&2
echo "=== eval / AP lines ===" 1>&2
tr '\r' '\n' < "$LOG" | grep -aE "Car_3d|Car_bev|recall|Average predicted|\*+Evaluation|Result is save|error|Error|Traceback" | tail -40 1>&2
echo "=== last 15 lines ===" 1>&2
tr '\r' '\n' < "$LOG" | tail -15 1>&2
