"""A tools/ script must import the REPO's pcdet, never the stale copy baked into the container image (/code/ST3D/pcdet), which
predates fixes such as the `_BASE_CONFIG_` merge (`e107939`).

Why (experiments_md 20261003_05 §5 item 59, 2026-11-27): item 59's tool, run as `python3 analysis/item59_bn_split.py
base_config`, imported pcdet before `_init_path`. Python puts the script's own directory first on sys.path, not tools/ or the
repo root, so the image's copy was imported. Its merge let the base config overwrite the child, and the generated config lost
item 58's z-shift augmentation. Caught before any run.

1. Static, every tracked tools/ file: a pcdet import must come after an `import _init_path` earlier in the file. ALLOWED lists
   the files that reach the repo's pcdet some other way. Each was judged once (2026-11-27, `scripts/check_pcdet_origin.py`,
   job 37208, output /home/koyama/data/item59/pcdet_probe.txt). The list may only shrink.
2. The probe tells the image's copy from the repo's.
3. End to end: item 59's tool, run exactly as a job runs it, writes the committed base config.
Run from the repo root inside the container.
"""
import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / 'tools'
sys.path.insert(0, str(TOOLS / 'scripts'))
import check_pcdet_origin as probe_mod  # noqa: E402

_REPO = 'repo: puts the repo root on sys.path itself before its module-level pcdet import (probe 37208: repo)'
_ENTRY = ('entry: pcdet is imported only inside functions, which run after main / cmd_* / the worker initializer has imported '
          '_init_path (checked by hand 2026-11-27)')
_LIB = 'library: imported only by entry points (train.py, test.py) that import _init_path first; never run as a script'
ALLOWED = {
    'tools/analysis/build_reference_objects.py': _REPO,
    'tools/analysis/condition_stratified_ap.py': _REPO + ', via kitti_frame_decomposition',
    'tools/analysis/foreground_correction_figures.py': _REPO,
    'tools/analysis/iassd_padding_check.py': _REPO,
    'tools/analysis/intensity_groups_eval.py': _REPO,
    'tools/analysis/iou_threshold_sensitivity.py': _REPO,
    'tools/analysis/kitti_cut_check.py': _REPO,
    'tools/analysis/lyft_deskew_rings.py': _REPO,
    'tools/analysis/lyft_ring_cache.py': _REPO,
    'tools/analysis/lyft_ring_viz.py': _REPO,
    'tools/analysis/oracle_box_error_kitti.py': _REPO,
    'tools/analysis/oraclef_check.py': _REPO,
    'tools/analysis/pandaset_ring_cache.py': _REPO,
    'tools/analysis/pl_teacher_near_miss_cuts.py': _REPO,
    'tools/analysis/preflight_eval.py': _REPO,
    'tools/analysis/profile_throughput.py': _REPO,
    'tools/analysis/ptsn_search.py': _REPO,
    'tools/analysis/reflection_r11_target_dump.py': _REPO,
    'tools/analysis/reflection_raydrop_m1.py': _REPO,
    'tools/analysis/released_test_scaled.py': _REPO,
    'tools/analysis/s2_box_residuals.py': _REPO,
    'tools/analysis/viz/dump_frames.py': _REPO,
    'tools/analysis/viz/select_frames.py': _REPO,
    'tools/lit_port/validate_translation.py': _REPO,
    'tools/analysis/iassd_pb_budget_census.py': _ENTRY,
    'tools/analysis/item47_thin_check.py': _ENTRY,
    'tools/analysis/item50_waymo_rings.py': _ENTRY,
    'tools/analysis/item51_d0_check.py': _ENTRY,
    'tools/analysis/item51_dts_views_check.py': _ENTRY,
    'tools/analysis/item54_lines.py': _ENTRY,
    'tools/analysis/item56_halves.py': _ENTRY,
    'tools/scripts/size_job.py': 'entry: inserts the repo root right before its function-level pcdet import',
    'tools/analysis/pseudo_label_threshold/kitti_oos.py': ('2026-09-26 one-off: inserts /st3d (that era\'s container bind of '
                                                           'the repo) and uses only the roiaware CPU op, no config'),
    'tools/lit_port/check_imports.py': 'LiT port check: imports pcdet shims from tools/lit_port/shims on purpose',
    'tools/eval_utils/eval_utils.py': _LIB,
    'tools/train_utils/train_st_utils.py': _LIB,
    'tools/train_utils/train_utils.py': _LIB,
}


def first_lines(src):
    """(line of the first pcdet import or None, line of the first `import _init_path` or None), at any depth."""
    pc, ip = [], []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name == 'pcdet' or a.name.startswith('pcdet.'):
                    pc.append(n.lineno)
                if a.name == '_init_path':
                    ip.append(n.lineno)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and (
                n.module == 'pcdet' or n.module.startswith('pcdet.')):
            pc.append(n.lineno)
    return (min(pc) if pc else None), (min(ip) if ip else None)


def violates(src):
    pc, ip = first_lines(src)
    return pc is not None and (ip is None or ip > pc)


def tracked_tools_py():
    out = subprocess.check_output(['git', 'ls-files', '--', 'tools/*.py', 'tools/**/*.py'], cwd=str(REPO), text=True)
    return sorted(set(out.split()))


def violators():
    bad = []
    for f in tracked_tools_py():
        try:
            src = (REPO / f).read_text()
            if violates(src):
                bad.append(f)
        except (SyntaxError, FileNotFoundError):
            continue
    return bad


def test_pre_fix_item59_tool_is_flagged_and_the_fixed_one_is_not():
    fixed = (TOOLS / 'analysis' / 'item59_bn_split.py').read_text()
    line = 'import _init_path  # noqa: F401,E402\n'
    assert fixed.count(line) == 1 and not violates(fixed)
    assert violates(fixed.replace(line, ''))                    # the source as it was before the fix


def test_every_new_tools_script_imports_init_path_before_pcdet():
    new = sorted(set(violators()) - set(ALLOWED))
    assert not new, ('these tools/ files import pcdet without `import _init_path` before it, so run as a script they get the '
                     'image\'s stale pcdet (tests/test_pcdet_import_order.py docstring): %s' % new)


def test_the_allowlist_only_shrinks():
    gone = sorted(set(ALLOWED) - set(violators()))
    assert not gone, 'fixed or removed, so drop them from ALLOWED: %s' % gone


def _script(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body)
    return p


def test_probe_tells_the_image_copy_from_the_repo(tmp_path):
    plain = _script(tmp_path, 'plain.py', 'x = 1\n')
    fixed = _script(tmp_path, 'fixed.py', 'import sys\nsys.path.insert(0, %r)\n' % str(REPO))
    assert probe_mod.probe(plain)['verdict'] == 'stale'
    assert probe_mod.probe(fixed)['verdict'] == 'repo'


def test_item59_tool_run_as_a_job_runs_it_writes_the_committed_base_config(tmp_path):
    """Pre-fix, this produced a config without random_world_translation (the image's merge)."""
    out = tmp_path / 'base.yaml'
    p = subprocess.run([sys.executable, 'analysis/item59_bn_split.py', 'base_config', '--out', str(out)], cwd=str(TOOLS),
                       capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    committed = (TOOLS / 'cfgs/da-ieee-access-analysis/item59/centerpoint-item59-b-base.yaml').read_text()
    assert out.read_text() == committed
    assert 'random_world_translation' in committed

