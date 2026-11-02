"""DDP launch: the training entry points must take their rank from LOCAL_RANK when the launcher passes no flag.

Defect: `tools/train.py` declared `--local_rank` with `default=0`. torch 2.5's launcher passes
`--local-rank` (hyphen; argparse rejects it as unrecognised) or, with `--use-env`, passes NOTHING and
sets only the `LOCAL_RANK` environment variable, which a default of 0 ignored: every process became
rank 0 and collided in `init_process_group`. No DDP job in this repo could run against this torch
version. Window: until ST3D 28e1b0b (2026-09-22), which defaults the argument to
`int(os.environ.get('LOCAL_RANK', 0))`; `tools/adaptive_train.py` reads the variable at launch instead
(`args.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))`).
Reports: experiments_md/20260922_06, memory/repo/index.md "Defect windows" (DDP impossible).

Static: the expressions are extracted with `ast` (no pcdet import chain, no GPU) and then EVALUATED
under a patched environment, so the test checks what the code does, not what it says. `tools/test.py`
still carries `default=0`; it is never launched under DDP here (eval_checkpoint.sh uses
`--launcher none`), so it is deliberately not asserted.
"""
import argparse
import ast
import os
import types
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'


def _add_argument_default(entry_point, flag):
    tree = ast.parse((TOOLS / entry_point).read_text(encoding='utf-8'))
    found = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument'
                and node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == flag):
            found += [kw.value for kw in node.keywords if kw.arg == 'default']
    assert len(found) == 1, '%s: expected exactly one %s default, found %d' % (entry_point, flag, len(found))
    return found[0]


def _eval(node, **names):
    return eval(compile(ast.Expression(body=node), '<default>', 'eval'), {'os': os, 'int': int, **names})


def _train_py_default():
    return _add_argument_default('train.py', '--local_rank')


@pytest.mark.parametrize('rank', ['0', '1', '3'])
def test_train_py_default_reads_local_rank_from_the_environment(monkeypatch, rank):
    monkeypatch.setenv('LOCAL_RANK', rank)
    assert _eval(_train_py_default()) == int(rank)


def test_train_py_default_is_zero_without_the_variable(monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    assert _eval(_train_py_default()) == 0


def test_an_explicit_flag_still_overrides_the_environment(monkeypatch):
    monkeypatch.setenv('LOCAL_RANK', '3')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=_eval(_train_py_default()))
    assert parser.parse_args([]).local_rank == 3
    assert parser.parse_args(['--local_rank', '5']).local_rank == 5


def test_pre_fix_default_ignored_the_environment(monkeypatch):
    """Reference: the pre-28e1b0b declaration (`default=0`) under --use-env makes every rank 0."""
    monkeypatch.setenv('LOCAL_RANK', '3')
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    assert parser.parse_args([]).local_rank == 0
    assert _eval(_train_py_default()) == 3, 'train.py must not be back on the constant default'


def _adaptive_train_override():
    """The `args.local_rank = ...` assignment in adaptive_train.py's main(), and its line."""
    tree = ast.parse((TOOLS / 'adaptive_train.py').read_text(encoding='utf-8'))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if (isinstance(t, ast.Attribute) and t.attr == 'local_rank' and isinstance(t.value, ast.Name)
                and t.value.id == 'args'):
            found.append(node)
    assert len(found) == 1, 'adaptive_train.py: expected exactly one `args.local_rank = ...`'
    return found[0]


@pytest.mark.parametrize('rank', ['0', '2'])
def test_adaptive_train_overrides_the_parsed_rank_from_the_environment(monkeypatch, rank):
    monkeypatch.setenv('LOCAL_RANK', rank)
    args = types.SimpleNamespace(local_rank=0)
    assert _eval(_adaptive_train_override().value, args=args) == int(rank)


def test_adaptive_train_keeps_the_parsed_rank_without_the_variable(monkeypatch):
    monkeypatch.delenv('LOCAL_RANK', raising=False)
    args = types.SimpleNamespace(local_rank=7)
    assert _eval(_adaptive_train_override().value, args=args) == 7


def test_adaptive_train_resolves_the_rank_before_initialising_the_process_group():
    text = (TOOLS / 'adaptive_train.py').read_text(encoding='utf-8')
    override_line = _adaptive_train_override().lineno
    init_lines = [i + 1 for i, l in enumerate(text.splitlines()) if 'init_dist_%s' in l]
    assert init_lines, 'adaptive_train.py no longer dispatches to init_dist_<launcher>'
    assert override_line < min(init_lines), 'LOCAL_RANK must be read before init_dist_* uses args.local_rank'
