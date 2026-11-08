#!/usr/bin/env bash
# experiments_md 20261003_05 §5 item 43a (design fixed 2026-11-09 01:18 JST; report experiments_md 20261109_01): eval-only
# scoring of the accumulated nuScenes checkpoints (epoch 20) on nuScenes val, one row after another on ONE GPU. ANALYSIS.
#   s1_<job>:  unperturbed single-sweep nuScenes val, the nuScenes oracle config itself (its MODEL / CLASS_NAMES / ONTOLOGY
#              equal every row's training config as logged, analysis/item43_config_check.py);
#   own_<job>: the row's OWN input, nuScenes val accumulated per platform to its training depth with compensation
#              (cfgs/da-ieee-access-analysis/item43/centerpoint-i43a-ownval-*.yaml, analysis/make_item43_configs.py);
#   s1_25743 / s1_31818: the oracle O_N seeds rescored with today's code (banked 44.98 / 44.91).
# Each row prints "=== ROW <tag> ..." / "=== END <tag> ..." to stderr; analysis/item43a_collect.py reads the log.
#   bash scripts/_item43a_evals_20261109.sh [row ...]      # default: every row, in the order below
set -u
cd /home/koyama/code/ST3D/tools
O=cfgs/da-ieee-access/centerpoint-sourceonly-nuscenes.yaml
D=cfgs/da-ieee-access-analysis/item43
declare -A JOB=([s1_27468]=27468 [s1_27804]=27804 [s1_27185]=27185 [s1_29354]=29354 [s1_25743]=25743 [s1_31818]=31818
                [own_27468]=27468 [own_27804]=27804 [own_27185]=27185 [own_29354]=29354)
declare -A CFG=([s1_27468]=$O [s1_27804]=$O [s1_27185]=$O [s1_29354]=$O [s1_25743]=$O [s1_31818]=$O
                [own_27468]=$D/centerpoint-i43a-ownval-n15-10.yaml [own_27804]=$D/centerpoint-i43a-ownval-n15-10.yaml
                [own_27185]=$D/centerpoint-i43a-ownval-n30-20.yaml [own_29354]=$D/centerpoint-i43a-ownval-n22-17.yaml)
ORDER=(s1_27468 s1_27804 s1_27185 s1_29354 s1_25743 s1_31818 own_27468 own_27804 own_27185 own_29354)
ROWS=("$@")
[ ${#ROWS[@]} -gt 0 ] || ROWS=("${ORDER[@]}")
for r in "${ROWS[@]}"; do
  [ -n "${JOB[$r]:-}" ] || { echo "=== ROW $r UNKNOWN ===" >&2; continue; }
  C=$(bash analysis/ckpt_of_job.sh "${JOB[$r]}" 20) || { echo "=== ROW $r FAILED: no checkpoint ===" >&2; continue; }
  B=6; case "$r" in own_*) B=4 ;; esac
  echo "=== ROW $r job ${JOB[$r]} cfg ${CFG[$r]} ckpt $C batch $B $(date '+%F %T') ===" >&2
  bash analysis/eval_checkpoint.sh "${CFG[$r]}" "$C" "i43a_$r" 20261109_i43a --batch_size $B
  echo "=== END $r exit $? $(date '+%F %T') ===" >&2
done
