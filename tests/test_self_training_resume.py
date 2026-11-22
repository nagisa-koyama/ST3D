"""A self-training row resumed with --ckpt keeps its pseudo-labels (experiments_md 20261003_05 §5 item 49b, job 34902).

train.py puts ps_label_dir in the run's W&B directory, and a resumed run gets a NEW W&B directory. Until 2026-11-22 the
resume therefore found no pseudo-labels: the row started with an empty dict, and under FROZEN_TEACHER_SINGLE_PASS the next
update epoch was not skipped either (ps_labels_generated stayed False), so 34902 regenerated at epoch 8 into an empty
memory ensemble and raised KeyError. These tests pin the fix: the checkpoint run's labels are read (and copied for the
record), a resumed run that holds them counts them as generated, and train.py hands the checkpoint to the self-training
loop. CPU only.
"""
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))
import _init_path  # noqa: F401,E402
from pcdet.utils import self_training_utils as stu  # noqa: E402


def _run(tmp_path, name, labels=None, epoch=0):
    files = tmp_path / name / 'files'
    (files / 'ckpt').mkdir(parents=True)
    (files / 'ps_label').mkdir()
    (files / 'ckpt' / 'checkpoint_epoch_8.pth').write_bytes(b'')
    if labels is not None:
        with open(files / 'ps_label' / ('ps_label_e%d.pkl' % epoch), 'wb') as f:
            pickle.dump(labels, f)
    return files


def test_a_resume_in_a_new_run_reads_the_checkpoint_runs_labels(tmp_path):
    old = _run(tmp_path, 'run-old', {'007070': {'gt_boxes': 'E0'}})
    new = _run(tmp_path, 'run-new')
    stu.PSEUDO_LABELS.clear()
    ps_pkl, in_hand = stu.resume_pseudo_labels(str(new / 'ps_label'), 8, str(old / 'ckpt' / 'checkpoint_epoch_8.pth'))
    assert ps_pkl == str(old / 'ps_label' / 'ps_label_e0.pkl') and in_hand is True
    assert stu.PSEUDO_LABELS['007070'] == {'gt_boxes': 'E0'}         # the frame 34902 could not find
    assert (new / 'ps_label' / 'ps_label_e0.pkl').exists()            # copied into the new run for the record
    stu.PSEUDO_LABELS.clear()


def test_the_runs_own_labels_win_and_a_fresh_start_is_unchanged(tmp_path, monkeypatch):
    from easydict import EasyDict
    monkeypatch.setattr(stu.cfg, 'SELF_TRAIN', EasyDict(), raising=False)   # start 0 reads SELF_TRAIN.INIT_PS
    old = _run(tmp_path, 'run-old', {'a': 'OLD'})
    own = _run(tmp_path, 'run-own', {'a': 'OWN'})
    stu.PSEUDO_LABELS.clear()
    ps_pkl, in_hand = stu.resume_pseudo_labels(str(own / 'ps_label'), 8, str(old / 'ckpt' / 'checkpoint_epoch_8.pth'))
    assert ps_pkl == str(own / 'ps_label' / 'ps_label_e0.pkl') and in_hand and stu.PSEUDO_LABELS['a'] == 'OWN'
    stu.PSEUDO_LABELS.clear()
    fresh = _run(tmp_path, 'run-fresh')
    assert stu.resume_pseudo_labels(str(fresh / 'ps_label'), 0, str(old / 'ckpt' / 'checkpoint_epoch_8.pth')) \
        == (None, False)                                              # start 0: nothing inherited, epoch-0 pass runs
    assert stu.resume_pseudo_labels(str(fresh / 'ps_label'), 8, None) == (None, False)
    stu.PSEUDO_LABELS.clear()


def test_the_loop_takes_the_resume_helper_and_skips_frozen_passes_with_its_result():
    st = (ROOT / 'tools' / 'train_utils' / 'train_st_utils.py').read_text()
    loop = st[st.index('def train_model_st('):]
    assert 'resume_ckpt=None' in loop[:loop.index(':\n')]
    assert 'ps_pkl, ps_labels_generated = self_training_utils.resume_pseudo_labels(ps_label_dir, start_epoch, ' \
           'resume_ckpt)' in loop
    assert 'check_already_exsit_pseudo_label(' not in loop            # the old resume path is gone
    assert 'if update_ps_label and ps_labels_generated and frozen_single_pass:' in loop


def test_train_py_hands_the_checkpoint_to_self_training_only():
    src = (ROOT / 'tools' / 'train.py').read_text()
    assert "if cfg.get('SELF_TRAIN', None) and args.ckpt is not None:\n" \
           "        train_func_kwargs['resume_ckpt'] = args.ckpt" in src
