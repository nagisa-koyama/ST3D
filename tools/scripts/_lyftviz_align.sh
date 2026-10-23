#!/bin/bash
# Lyft ring visual check, 64-beam drawn the same way as 40-beam (experiments_md 20261022_05 §6; user 2026-10-23 ~23:30:
# 「60ビームの可視化方法を40ビームと合わせてくれないか」). CPU only, no W&B. Submit from ST3D/tools:
#   WANDB_NOTES="CPU only, no W&B: ..." scripts/submit.sh --partition=a100,a6000_ada --cpus-per-task=4 --mem=32G \
#     --time=01:00:00 --job-name=lyftviz --output=logs/output_%j_%x.txt --error=logs/error_%j_%x.txt scripts/_lyftviz_align.sh
# 0. back up the published directory; 1. replay the gate's frames again into a separate directory and require the frames
# JSON to equal the published one (G1-G3 and the kept-ring histograms are printed by the frames stage); 2. redraw with
# --align_platforms in place; every file that existed before must come out byte-identical, else the backup copy is put
# back and the job says so; 3. the per-frame tables.
set -euo pipefail
cd /home/koyama/code/ST3D/tools
SIF=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
OUT=/home/koyama/data/lyft_rings/viz_20261022
BAK=/home/koyama/data/lyft_rings/viz_20261022_before_20261023
CHK=/home/koyama/data/lyft_rings/viz_20261023_replay_check
run() { singularity exec --bind /home/koyama/data/:/storage "$SIF" python3 "$@"; }
echo "=== code at $(git -C /home/koyama/code/ST3D log -1 --format=%h) (live checkout), $(date '+%F %T %Z'), $(hostname)"

if [ ! -d "$BAK" ]; then cp -a "$OUT" "$BAK"; echo "backed up $OUT -> $BAK"; fi

mkdir -p "$CHK"
run analysis/lyft_ring_viz.py frames --out_dir "$CHK"
run - "$BAK/frames_20261022.json" "$CHK/frames_20261022.json" <<'EOF'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
a.pop('seconds'); b.pop('seconds')
print('FRAMES_JSON_EQUAL', a == b, flush=True)
sys.exit(0 if a == b else 1)
EOF

run analysis/lyft_ring_viz.py plots --align_platforms --out_dir "$OUT"
n_same=0; n_diff=0
for f in "$BAK"/*.png; do
  b=$(basename "$f")
  if cmp -s "$f" "$OUT/$b"; then n_same=$((n_same + 1)); echo "IDENTICAL $b"
  else n_diff=$((n_diff + 1)); cp -p "$f" "$OUT/$b"; echo "DIFFERED $b: published copy restored"; fi
done
for f in "$OUT"/*.png; do
  b=$(basename "$f"); [ -e "$BAK/$b" ] || echo "NEW $b $(stat -c %s "$f") bytes"
done
echo "pre-existing figures: $n_same identical, $n_diff differed (restored)"

run analysis/lyft_ring_viz.py table --out_dir "$OUT"
echo "=== done $(date '+%F %T %Z')"
