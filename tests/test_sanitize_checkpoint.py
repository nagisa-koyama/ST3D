"""tools/analysis/sanitize_checkpoint.py: third-party checkpoints are opened weights-only and re-saved as tensors only.

The refusal case is the point: a pickle that calls a non-tensor global must exit 3 WITHOUT the call running.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'analysis' / 'sanitize_checkpoint.py'


def run(src, out):
    return subprocess.run([sys.executable, '-I', str(SCRIPT), str(src), str(out)], capture_output=True, text=True)


class _Marker:
    """Unpickling this would create the marker file; weights_only must refuse before that happens."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, 'w'))


def test_openpcdet_style_checkpoint_round_trips(tmp_path):
    state = {'backbone_3d.conv_input.0.weight': torch.randn(3, 3, 3, 4, 16),
             'dense_head.conv_cls.bias': torch.randn(2)}
    src, out = tmp_path / 'in' / 'ckpt.pth', tmp_path / 'ckpt_sanitized.pth'
    src.parent.mkdir()
    torch.save({'epoch': 30, 'it': 1234, 'model_state': state, 'optimizer_state': None, 'version': 'pcdet+0.3.0'}, src)
    r = run(src, out)
    assert r.returncode == 0, r.stderr
    info = json.loads(r.stdout)
    assert info['n_tensors'] == 2 and info['n_non_tensor_entries'] == 0
    assert info['epoch'] == 30 and info['version'] == 'pcdet+0.3.0'
    assert info['key_prefixes'] == ['backbone_3d', 'dense_head']
    saved = torch.load(out, map_location='cpu', weights_only=True)
    assert set(saved) == {'model_state', 'epoch', 'version', 'source_sha256'}
    for k, v in state.items():
        assert torch.equal(saved['model_state'][k], v)
    assert saved['source_sha256'] == info['source_sha256']


def test_bare_state_dict_is_accepted(tmp_path):
    src, out = tmp_path / 'sd.pth', tmp_path / 'sd_out.pth'
    torch.save({'a.weight': torch.ones(2)}, src)
    r = run(src, out)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)['n_tensors'] == 1


def test_malicious_pickle_is_refused_without_executing(tmp_path):
    marker = tmp_path / 'pwned'
    src, out = tmp_path / 'evil.pth', tmp_path / 'evil_out.pth'
    torch.save({'model_state': {'w': torch.ones(1)}, 'payload': _Marker(str(marker))}, src)
    assert not marker.exists()
    r = run(src, out)
    assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
    info = json.loads(r.stdout)
    assert 'weights_only_error' in info
    assert any(g.endswith('open') for g in info['pickle_globals']), info['pickle_globals']
    assert not marker.exists(), 'the pickle payload ran'
    assert not out.exists()


def test_static_scan_lists_globals_of_a_clean_file(tmp_path):
    src, out = tmp_path / 'c.pth', tmp_path / 'c_out.pth'
    torch.save({'model_state': {'w': torch.ones(1)}}, src)
    info = json.loads(run(src, out).stdout)
    assert all(g.startswith(('torch.', 'collections.')) for g in info['pickle_globals']), info['pickle_globals']


def test_numpy_scalar_epoch_is_accepted_but_other_globals_still_refused(tmp_path):
    import numpy as np
    src, out = tmp_path / 'np.pth', tmp_path / 'np_out.pth'
    torch.save({'epoch': np.int64(7), 'model_state': {'w': torch.ones(1)}}, src)
    r = run(src, out)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert json.loads(r.stdout)['n_tensors'] == 1
