#!/usr/bin/env bash
# Re-run the whole pseudo-label threshold analysis on a pseudo-label file.
#
#   rerun_all.sh <ps_label_e0.pkl (container path, /storage/...)> <work dir> <old work dir with kitti_oos.pkl> [FULL_RANGE=1]
#
# Stage A (parallel, CPU, ~15 min): audit, per-box score/points, temporal persistence, temporal gaps.
# Stage B: every table script, then the figures into <work dir>/figs. Logs in <work dir>/logs.
set -uo pipefail
PS=$1; W=$2; OLD=$3; export FULL_RANGE=${4:-1}; export PS_LABEL=$PS
D=/home/koyama/code/ST3D/tools/analysis/pseudo_label_threshold
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
X="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/st3d --bind /home/koyama/code:/home/koyama/code --bind /home/koyama/data:/home/koyama/data --bind $W:$W --bind $OLD:$OLD $SIF"
mkdir -p "$W/logs" "$W/figs"; cp "$OLD/kitti_oos.pkl" "$W/"
cd /home/koyama/code/ST3D/tools
CFG=cfgs/da-ieee-access/centerpoint-foreground-lyft2nuscenes.yaml

if [ -z "${SKIP_A:-}" ]; then
echo "stage A start $(date +%T)"
$X python analysis/pseudo_label_foreground_audit.py --cfg_file $CFG --ps_label "$PS" --out "$W/audit_full.pkl" > "$W/logs/audit.txt" 2>&1 &
$X python $D/score_vs_points.py $CFG "$PS" "$W/score_pts.pkl" > "$W/logs/score_vs_points.txt" 2>&1 &
$X python $D/persistence.py "$W" > "$W/logs/persistence.txt" 2>&1 &
$X python $D/gaps.py "$W" 0.3 2.0 > "$W/logs/gaps.txt" 2>&1 &
wait
echo "stage A done $(date +%T)"; ls -la "$W"/*.pkl
fi

echo "stage B start $(date +%T)"
$X python $D/rates.py "$W/audit_full.pkl" > "$W/logs/rates.txt" 2>&1 || echo "FAILED rates"
$X python $D/corr.py "$W/score_pts.pkl" > "$W/logs/corr.txt" 2>&1 || echo "FAILED corr"
for s in perclass overlap thr thr2 abs gtmed side side2 rangemix closest cum changepoint an1 harness split boot i6 i7 ringcb i8 otsu gmm_split bic; do
  $X python $D/$s.py "$W" > "$W/logs/$s.txt" 2>&1 || echo "FAILED $s"
done
$X python $D/an_gaps.py "$W" > "$W/logs/an_gaps.txt" 2>&1 || echo "FAILED an_gaps"
$X python $D/figures.py "$W" "$W/figs" > "$W/logs/figures.txt" 2>&1 || echo "FAILED figures"
echo "stage B done $(date +%T)"
