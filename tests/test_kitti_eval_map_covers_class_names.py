"""
Every dataset's `kitti_eval` must be able to map the DA class set it will be evaluated with.

Bug (found 2026-09-23, job 25769): the Lyft -> Lyft source-only row trained all 30 epochs in
10h47m and then died at its first evaluation with `KeyError: 'Cyclist'`, thrown by

    kitti_class_names = [map_name_to_kitti[x] for x in class_names]

in `lyft_dataset.py::kitti_eval`. `class_names` is the MODEL's `CLASS_NAMES`, which across the
whole `da-ieee-access` family is `['Car', 'Pedestrian', 'Cyclist']` - already in KITTI vocabulary,
because the ontology remap happens upstream in the dataset. Lyft's map carried passthrough entries
for 'Car' and 'Pedestrian' but not 'Cyclist', so two of three classes resolved and the third
raised. `nuscenes_dataset.py` had had exactly this passthrough since the UADA3D migration
("passthrough if already remapped upstream"); the same line was simply never mirrored to Lyft or
to PandaSet.

Blast radius when found: Lyft (job 25769, dead) and PandaSet (jobs 25782 and 25790, which would
have died the same way ~38 h and ~101 h in). Nothing needed retraining in any case, because the
crash lands in the post-training evaluation and every checkpoint is already on disk - but a run
that produces no AP has still cost its full wall-clock.

Why this test is STATIC. `map_name_to_kitti` is a local variable inside each `kitti_eval`, not a
module constant, and reaching it for real needs that dataset's infos on disk plus a CUDA-capable
node (`kitti_object_eval_python.eval` compiles a numba.cuda kernel at import). Parsing the literal
out of the source is what can run on a CPU node in milliseconds, and it pins the property that
actually failed: the map's KEY SET must cover the class names it will be indexed with.
"""
import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# The class set every `da-ieee-access` config declares, and the one the four sensor-pair rows and
# the five source-only rows are all evaluated with.
DA_CLASS_NAMES = ['Car', 'Pedestrian', 'Cyclist']

# Each NON-KITTI dataset whose `evaluation()` routes to the KITTI-metric path, and the file
# holding its map. KITTI itself is deliberately absent: its GT is already in KITTI vocabulary, so
# `kitti_dataset.py::evaluation` carries no `map_name_to_kitti` at all - it uses the config's
# optional `CLASS_MAPPING` to normalise a SOURCE-vocabulary model's names instead, which is a
# different mechanism with its own regression test (test_kitti_eval_class_mapping.py).
DATASET_FILES = {
    'lyft': 'pcdet/datasets/lyft/lyft_dataset.py',
    'nuscenes': 'pcdet/datasets/nuscenes/nuscenes_dataset.py',
    'pandaset': 'pcdet/datasets/pandaset/pandaset_dataset.py',
    'waymo': 'pcdet/datasets/waymo/waymo_dataset.py',
}


def _literal_maps(path):
    """Every `map_name_to_kitti = {...}` dict literal in a file, as real dicts."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    maps = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(t, ast.Name) and t.id == 'map_name_to_kitti' for t in node.targets):
            continue
        try:
            maps.append(ast.literal_eval(node.value))
        except ValueError:  # a non-literal entry: not something this test can judge
            continue
    return maps


@pytest.mark.parametrize('dataset,relpath', sorted(DATASET_FILES.items()))
def test_kitti_eval_map_covers_da_class_names(dataset, relpath):
    path = REPO / relpath
    maps = _literal_maps(path)
    assert maps, f'no map_name_to_kitti literal found in {relpath} - has it been renamed?'

    for i, m in enumerate(maps):
        missing = [c for c in DA_CLASS_NAMES if c not in m]
        assert not missing, (
            f'{relpath} map_name_to_kitti #{i} cannot map {missing} - a model with '
            f'CLASS_NAMES={DA_CLASS_NAMES} evaluated on {dataset} raises KeyError AFTER training '
            f'completes. Add a passthrough entry (e.g. "Cyclist": "Cyclist").'
        )


@pytest.mark.parametrize('dataset,relpath', sorted(DATASET_FILES.items()))
def test_kitti_eval_map_passthroughs_are_identities(dataset, relpath):
    """A passthrough must not silently RENAME a class.

    The failure this guards is subtler than the KeyError and would not crash: mapping 'Cyclist' to
    anything but 'Cyclist' would score one class's detections against another's GT and report a
    plausible, wrong AP.
    """
    for i, m in enumerate(_literal_maps(REPO / relpath)):
        for c in DA_CLASS_NAMES:
            if c in m:
                assert m[c] == c, (
                    f'{relpath} map_name_to_kitti #{i} maps {c!r} -> {m[c]!r}; a KITTI-vocabulary '
                    f'name must map to itself or the AP is computed against the wrong class.'
                )
