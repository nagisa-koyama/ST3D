#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}
MASTER_ADDR=4000
MASTER_PORT=4000
WORLD_SIZE=2
RANK=0
LOCAL_RANK=0
NODE_RANK=0

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}

