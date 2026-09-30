"""tools/analysis/kitti_eval_cpu.py swaps ONE function of the official KITTI evaluator: the CUDA
rotated-IoU kernel. These tests pin that the swap is exact where the evaluator uses it.

The real kernel cannot run on a CPU node, but it can under numba's CUDA simulator, which must be
switched on before numba is first imported - so the comparison runs in a subprocess.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools' / 'analysis'))


def test_matches_the_cuda_kernel_under_the_simulator():
    script = textwrap.dedent(f"""
        import os, sys
        os.environ['NUMBA_ENABLE_CUDASIM'] = '1'
        sys.path.insert(0, {str(ROOT / 'tools')!r}); sys.path.insert(0, {str(ROOT / 'tools' / 'analysis')!r})
        import _init_path  # the live repo's pcdet, not the image's stale editable install
        import numpy as np
        from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval
        from kitti_eval_cpu import rotate_iou_cpu
        rng = np.random.default_rng(0)
        def boxes(n):
            return np.stack([rng.uniform(-3, 3, n), rng.uniform(-3, 3, n), rng.uniform(1, 5, n),
                             rng.uniform(1, 3, n), rng.uniform(-np.pi, np.pi, n)], 1).astype(np.float32)
        a, b = boxes(12), boxes(9)
        iou_ref, iou = rotate_iou_gpu_eval(a, b, -1), rotate_iou_cpu(a, b, -1)
        inter_ref, inter = rotate_iou_gpu_eval(a, b, 2), rotate_iou_cpu(a, b, 2)
        assert (iou_ref > 0).sum() > 20, 'fixture has too few overlapping pairs to test anything'
        assert np.abs(iou_ref - iou).max() < 1e-3, np.abs(iou_ref - iou).max()
        assert np.abs(inter_ref - inter).max() < 1e-2, np.abs(inter_ref - inter).max()
        # the sign that matters: the same boxes with the angle NOT negated must disagree
        flipped = a.copy(); flipped[:, 4] *= -1
        assert np.abs(rotate_iou_cpu(flipped, b, -1) - iou_ref).max() > 0.1
        print('OK')
    """)
    out = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, cwd=ROOT / 'tools')
    assert out.returncode == 0 and 'OK' in out.stdout, out.stdout + out.stderr


def test_refuses_criteria_the_evaluator_never_uses():
    from kitti_eval_cpu import rotate_iou_cpu
    box = np.array([[0, 0, 4, 2, 0.3]], np.float32)
    for criterion in (0, 1):
        with pytest.raises(ValueError):
            rotate_iou_cpu(box, box, criterion)


def test_identical_boxes_and_disjoint_boxes():
    from kitti_eval_cpu import rotate_iou_cpu
    a = np.array([[0, 0, 4, 2, 0.7]], np.float32)
    far = np.array([[50, 50, 4, 2, 0.7]], np.float32)
    assert rotate_iou_cpu(a, a, -1)[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert rotate_iou_cpu(a, a, 2)[0, 0] == pytest.approx(8.0, abs=1e-3)
    assert rotate_iou_cpu(a, far, -1)[0, 0] == 0.0
