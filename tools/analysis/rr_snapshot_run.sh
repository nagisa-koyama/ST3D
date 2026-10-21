#!/usr/bin/env bash
# Run a CPU analysis command from a COMMITTED snapshot of ST3D (experiments_md 20261018_02 §13.3), so a Slurm CPU job never
# reads the shared checkout's uncommitted edits: `git archive <sha>` into the job's node-local scratch, plus the compiled
# extensions (*.so, gitignored) and the data/ symlinks copied from the live checkout, bound over the repo path in the container.
#
#   bash analysis/rr_snapshot_run.sh <sha> <command ...>        (from ST3D/tools, inside a Slurm job; the command runs in tools/)
set -euo pipefail
SHA=${1:?usage: rr_snapshot_run.sh <sha> <command ...>}; shift
REPO=/home/koyama/code/ST3D
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local_cache/${SLURM_JOB_ID}" ] && [ -w "/local_cache/${SLURM_JOB_ID}" ]; then
    SCR=/local_cache/${SLURM_JOB_ID}
else
    SCR=/tmp/st3d_snap_${SLURM_JOB_ID:-$$}
    trap 'rm -rf "$SCR"' EXIT
fi
SNAP=$SCR/ST3D
mkdir -p "$SNAP"
git -C "$REPO" archive "$SHA" | tar -x -C "$SNAP"
(cd "$REPO" && /usr/bin/find pcdet -name '*.so' -print0 | xargs -0 -I{} cp --parents {} "$SNAP"/)
cp -a "$REPO/data/." "$SNAP/data/"
NSO=$(/usr/bin/find "$SNAP/pcdet" -name '*.so' | wc -l)
[ "$NSO" -gt 0 ] || { echo "rr_snapshot_run: no compiled extensions in the snapshot" >&2; exit 2; }
echo "=== snapshot of $(git -C "$REPO" rev-parse --short "$SHA") -> $SNAP ($NSO .so) ==="
singularity exec --bind /home/koyama/data/:/storage --bind "$SNAP":/home/koyama/code/ST3D \
    --pwd /home/koyama/code/ST3D/tools "$SIF" "$@"
