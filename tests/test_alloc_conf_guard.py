"""The max_split_size_mb guard (tools/_alloc_conf_guard.py, experiments_md/20260926_05).

torch < 2.9 has a use-after-free in the CUDA caching allocator's release_available_cached_blocks,
reachable only when max_split_size_mb is configured - and the container image configures it for
every job. The guard strips that one option before CUDA starts. These tests pin the parsing, the
version gate, and the wiring that makes it run first in every entry point.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / 'tools'
sys.path.insert(0, str(TOOLS))
import _alloc_conf_guard as guard  # noqa: E402


@pytest.mark.parametrize('conf, expected', [
    ('max_split_size_mb:128', ''),
    ('max_split_size_mb:128,expandable_segments:True', 'expandable_segments:True'),
    ('garbage_collection_threshold:0.6,max_split_size_mb:64', 'garbage_collection_threshold:0.6'),
    ('garbage_collection_threshold:0.6 max_split_size_mb:64', 'garbage_collection_threshold:0.6'),
    ('expandable_segments:True', 'expandable_segments:True'),
])
def test_strip_keeps_every_other_option(conf, expected):
    assert guard.strip_max_split_size(conf) == expected


def test_image_default_is_removed_entirely():
    env = {guard.ENV: 'max_split_size_mb:128'}
    assert guard.apply(env, torch_version=(2, 5)) == 'max_split_size_mb:128'
    assert guard.ENV not in env


def test_other_options_survive():
    env = {guard.ENV: 'max_split_size_mb:128,expandable_segments:True'}
    guard.apply(env, torch_version=(2, 5))
    assert env[guard.ENV] == 'expandable_segments:True'


@pytest.mark.parametrize('version', [(2, 9), (2, 10), (3, 0)])
def test_fixed_torch_is_left_alone(version):
    env = {guard.ENV: 'max_split_size_mb:128'}
    assert guard.apply(env, torch_version=version) is None
    assert env[guard.ENV] == 'max_split_size_mb:128'


def test_unknown_torch_version_still_strips():
    # Losing a fragmentation tweak costs far less than the crash, so an unreadable version strips.
    env = {guard.ENV: 'max_split_size_mb:128'}
    guard.apply(env, torch_version=None)
    assert guard.ENV not in env


@pytest.mark.parametrize('env', [{}, {guard.ENV: ''}, {guard.ENV: 'expandable_segments:True'}])
def test_nothing_to_do(env):
    before = dict(env)
    assert guard.apply(env, torch_version=(2, 5)) is None
    assert env == before


def test_importing_init_path_strips_the_real_environment():
    """End to end, in a fresh interpreter: the guard must actually be wired into _init_path."""
    env = dict(os.environ, **{guard.ENV: 'max_split_size_mb:128,expandable_segments:True'})
    code = ('import os, sys; sys.path.insert(0, %r); import _init_path; '
            'print(repr(os.environ.get(%r)))' % (str(TOOLS), guard.ENV))
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                         check=True)
    version = guard._torch_version()
    if version is not None and version >= guard.FIXED_IN:
        assert out.stdout.strip() == repr('max_split_size_mb:128,expandable_segments:True')
    else:
        assert out.stdout.strip() == repr('expandable_segments:True')
        assert 'dropped' in out.stderr


@pytest.mark.parametrize('entry', ['train.py', 'test.py', 'adaptive_train.py'])
def test_entry_points_import_init_path_first(entry):
    """The guard only works if it runs before anything can initialise CUDA."""
    tree = ast.parse((TOOLS / entry).read_text())
    first = next(node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom)))
    assert isinstance(first, ast.Import) and first.names[0].name == '_init_path', (
        '%s must import _init_path before anything else; it installs the allocator guard' % entry)
