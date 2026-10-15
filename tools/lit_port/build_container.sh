#!/usr/bin/env bash
# Build the LiT translation container lit_cuda128.sif as a Slurm CPU job (never on the master):
#
#   WANDB_NOTES="CPU only, no W&B: build ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=16 \
#       --mem=48G --time=04:00:00 --job-name=litbuild --output=/home/koyama/data/lit/build/build_%j.txt \
#       --error=/home/koyama/data/lit/build/build_%j.txt lit_port/build_container.sh
#
# For the legal LiT row (experiments_md 20261015_02). It runs LiT's scene reconstruction (NKSR background, DeepSDF
# foreground) and its CPU ray caster (Open3D / Embree) on Waymo TRAIN. The detector trains later, in the usual
# st3d_cuda12_ubuntu2404.sif, on the translated frames.
#
# UNPRIVILEGED on purpose: compute nodes have no subuid mapping for koyama, so `singularity build --fakeroot` from a
# .def fails there ("no valid mapping entry", job 28566). Instead the official PyTorch devel image is unpacked into a
# user-owned sandbox, packages go in with pip / conda as the user, and the sandbox is packed into a SIF.
#
# Departures from LiT's pins (torch 2.0.0 + cu118, NKSR wheels from nksr.huangjh.tech):
#   - that wheel host no longer resolves (checked 2026-10-15); NKSR's maintainers moved to source builds tested on
#     torch 2.7.0 + CUDA 12.8 (NKSR README, 2025-09-08). So torch 2.7.0 + cu128, NKSR from source at a pinned commit.
#   - Python 3.11 (the image's), not 3.10.
#   - not installed, because the translation path does not use them: TensorFlow / waymo-open-dataset (scenes are built
#     from our processed Waymo data), OptiX / otk-pyoptix / cupy (CPU ray caster), LiT's spconv 1.2.1 (detector only).
# LiT itself is NOT baked in: the clone /home/koyama/code/lit (b1516de) is used in place, its ops built in-tree.
set -euo pipefail
OUT=${OUT:-/home/koyama/code/singularity/lit_cuda128.sif}
BASE=docker://pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
NKSR_COMMIT=e403368
SCRATCH=/local_cache/${SLURM_JOB_ID:-manual}
mkdir -p "$SCRATCH" 2>/dev/null || true
if ! { [ -d "$SCRATCH" ] && [ -w "$SCRATCH" ]; }; then SCRATCH=$(mktemp -d /tmp/litbuild.XXXXXX); fi
export SINGULARITY_TMPDIR=$SCRATCH/tmp SINGULARITY_CACHEDIR=$SCRATCH/cache
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"
JOBS=${SLURM_CPUS_PER_TASK:-8}
echo "host $(hostname)  scratch $SCRATCH  out $OUT  jobs $JOBS  $(date)"
[ -e "$OUT" ] && { echo "refusing to overwrite $OUT" >&2; exit 2; }
trap 'chmod -R u+w "$SCRATCH" 2>/dev/null; rm -rf "$SCRATCH"' EXIT

SB=$SCRATCH/lit_sb
singularity build --sandbox "$SB" "$BASE"
# --writable cannot create a bind destination (job 28567: "destination /scratch doesn't exist in container"); the sandbox is
# user-owned, so create it here.
mkdir -p "$SB/scratch"
singularity exec --writable --no-home --bind "$SCRATCH:/scratch" --env MAX_JOBS="$JOBS" --env TORCH_CUDA_ARCH_LIST="7.5;8.6;8.9" \
    --env PIP_CACHE_DIR=/scratch/pipcache --env HOME=/scratch/home "$SB" bash -euxc "
    mkdir -p /scratch/home
    python -c 'import torch, sys; print(sys.version); print(torch.__version__, torch.version.cuda)'
    which nvcc gcc g++ ninja || true
    conda install -y -q -c conda-forge eigen git
    pip install --no-cache-dir torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
    # numpy < 2: LiT and its dependencies predate numpy 2
    pip install --no-cache-dir 'numpy==1.26.4' 'python-pycg>=1.0.1' matplotlib ninja GitPython pyntcloud plyfile \
        open3d camtools trimesh scikit-image python-igraph numba easydict pyyaml tqdm ipdb tensorboardX scipy
    pip install --no-cache-dir 'numpy==1.26.4'
    # no git in the image: fetch the pinned commit as a tarball (job 28639: 'git: command not found')
    mkdir -p /opt/NKSR && python - <<'PY'
import io, tarfile, urllib.request
url = 'https://github.com/nv-tlabs/NKSR/archive/$NKSR_COMMIT.tar.gz'
data = urllib.request.urlopen(url, timeout=120).read()
t = tarfile.open(fileobj=io.BytesIO(data))
top = t.getnames()[0].split('/')[0]
t.extractall('/opt/NKSR_x')
import os, shutil
shutil.rmtree('/opt/NKSR'); os.rename('/opt/NKSR_x/' + top, '/opt/NKSR')
print('NKSR sources', top)
PY
    cd /opt/NKSR
    CPATH=/opt/conda/include/eigen3:\${CPATH:-} pip install --no-cache-dir --no-build-isolation /opt/NKSR/package/
    pip freeze > /opt/lit_container_pip_freeze.txt
    python -c 'import torch, torch_scatter, camtools, nksr; print(\"core imports ok\", torch.__version__, nksr.__file__)'
    python -c 'import open3d; print(\"open3d\", open3d.__version__)' || echo 'WARNING: open3d import failed'
    python -c 'import pycg.vis; print(\"pycg.vis ok\")' || echo 'WARNING: pycg.vis import failed'
"
singularity build "$SCRATCH/lit_cuda128.sif" "$SB"
cp "$SCRATCH/lit_cuda128.sif" "$OUT"
sha256sum "$OUT"
echo "done $(date)"
