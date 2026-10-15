#!/usr/bin/env bash
# Run a command in the LiT translation container (experiments_md 20261015_02):  lit_port/lit_exec.sh [--nv] python -I <script> ...
# LD_LIBRARY_PATH: conda's lib (libEGL etc.) plus the libusb prefix on NFS (open3d); TORCH_EXTENSIONS_DIR: where LiT's
# lit_ext JIT-builds. The scripts under lit_port/ add LiT and the pcdet shim to sys.path themselves (python -I ignores
# PYTHONPATH). LiT's clone is bound read-write only because LiT writes nothing into it with the paths this row sets.
NV=""
if [ "${1:-}" = "--nv" ]; then NV="--nv"; shift; fi
mkdir -p /home/koyama/data/lit/torch_ext
exec singularity exec $NV --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/home/koyama/code/ST3D \
    --bind /home/koyama/code/lit:/home/koyama/code/lit \
    --env LD_LIBRARY_PATH=/opt/conda/lib:/home/koyama/data/lit/extra_libs/lib \
    --env TORCH_EXTENSIONS_DIR=/home/koyama/data/lit/torch_ext \
    /home/koyama/code/singularity/lit_cuda128.sif "$@"
