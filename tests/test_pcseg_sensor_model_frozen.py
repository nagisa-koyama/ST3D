"""render_study/sensor_models.json is frozen once written (experiments_md 20261116_01 §8.9): the render caches record its
sha256 and pcocc pins it, so pcseg_render_sensor_model.py must refuse to rewrite it in place."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'analysis'))

import pcseg_render_sensor_model as sensor_model                                                  # noqa: E402


def test_refuses_an_out_dir_that_already_holds_sensor_models(tmp_path, monkeypatch):
    frozen = tmp_path / 'sensor_models.json'
    frozen.write_text('{"pandar64": {"origin": [0.78, 0.0, 1.86]}}')
    before = frozen.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['pcseg_render_sensor_model.py', '--out', str(tmp_path)])
    with pytest.raises(SystemExit, match='frozen'):
        sensor_model.main()
    assert frozen.read_bytes() == before
