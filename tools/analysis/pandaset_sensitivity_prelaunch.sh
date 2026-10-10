#!/usr/bin/env bash
# Pre-launch checks of the PandaSet oracle-degradation twins (experiments_md 20261011_03), as one Slurm CPU job:
# the manipulated-cloud check over every spin twin, preflight_eval one config per process, and the config tests.
# From ST3D/tools:  bash analysis/pandaset_sensitivity_prelaunch.sh <frames>
set -u
N=${1:-30}
IMG=/home/koyama/code/singularity/st3d_cuda12_ubuntu2404.sif
RUN="singularity exec --bind /home/koyama/data/:/storage --bind /home/koyama/code/ST3D:/root/ST3D $IMG"
D=cfgs/da-ieee-access-analysis/sensitivity-pandaset
echo "=== manipulated-cloud check, $N val frames ==="
$RUN python3 analysis/pandaset_eval_thin_check.py "$N" $D/centerpoint-pandaset-spin2spin-*.yaml 2>&1 \
    | /usr/bin/grep -vE "is loaded|alloc_conf|INFO"
echo "=== preflight_eval, one config per process ==="
for c in cfgs/da-ieee-access/centerpoint-pandaset-spin2spin.yaml cfgs/da-ieee-access/centerpoint-pandaset-flash2flash.yaml $D/*.yaml; do
    $RUN python3 analysis/preflight_eval.py "$c" 2>&1 | /usr/bin/grep -E "^(OK|FAIL|ERR)|Traceback|Error" | head -3
done
echo "=== config tests ==="
cd .. && $RUN python3 -m pytest tests/test_config_real_configs.py tests/test_pandaset_rings.py -q 2>&1 | tail -2
echo "=== done ==="
