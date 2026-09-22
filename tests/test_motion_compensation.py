"""Per-object motion compensation: the rigid box-to-box transform and its guards.

Ego compensation alone leaves a moving object smeared, so the points it contributed in earlier
sweeps fall outside its box in the anchor frame. Over 30 frames that costs a moving object the
whole benefit of accumulating - x1.01 against a static object's x2.74.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pcdet.datasets.motion_compensation import (  # noqa: E402
    boxes_to_frame, interpolate_boxes, move_points_between_boxes)


def box(x, y, z=0.0, dx=4.0, dy=2.0, dz=2.0, yaw=0.0):
    return np.array([x, y, z, dx, dy, dz, yaw], dtype=np.float64)


def test_a_point_inside_a_moving_box_follows_it():
    pts = np.array([[10.0, 0.0, 0.0, 0.3]])
    out = move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert out[0][0] == pytest.approx(25.0)
    assert out[0][3] == pytest.approx(0.3), 'intensity must be carried through untouched'


def test_a_point_outside_every_box_is_left_alone():
    pts = np.array([[10.0, 50.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert out[0][1] == pytest.approx(50.0)


def test_rotation_is_applied_about_the_box_centre():
    """A box that turns 90 degrees takes its points around with it."""
    pts = np.array([[11.0, 0.0, 0.0, 0.0]])          # 1 m ahead of the centre
    out = move_points_between_boxes(pts, {'a': box(10, 0, yaw=0.0)},
                                    {'a': box(10, 0, yaw=np.pi / 2)})
    assert out[0][0] == pytest.approx(10.0, abs=1e-9)
    assert out[0][1] == pytest.approx(1.0, abs=1e-9)


def test_a_track_missing_from_the_anchor_frame_is_skipped():
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'gone': box(10, 0)}, {'other': box(25, 0)})
    assert out[0][0] == pytest.approx(10.0)


def test_a_point_is_claimed_by_at_most_one_track():
    """Overlapping boxes must not move the same point twice."""
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    then = {'a': box(10, 0), 'b': box(10.5, 0)}
    now = {'a': box(30, 0), 'b': box(60, 0)}
    out = move_points_between_boxes(pts, then, now)
    assert out[0][0] in (pytest.approx(30.0), pytest.approx(60.0))
    assert out[0][0] != pytest.approx(80.0), 'moved twice'


def test_containment_is_tested_before_anything_moves():
    """Otherwise a box moved into another box's path would sweep up its points."""
    pts = np.array([[10.0, 0.0, 0.0, 0.0], [30.0, 0.0, 0.0, 0.0]])
    out = move_points_between_boxes(pts, {'a': box(10, 0), 'b': box(30, 0)},
                                    {'a': box(30, 0), 'b': box(80, 0)})
    assert sorted(np.round(out[:, 0], 6)) == [30.0, 80.0]


def test_the_input_array_is_not_modified():
    pts = np.array([[10.0, 0.0, 0.0, 0.0]])
    move_points_between_boxes(pts, {'a': box(10, 0)}, {'a': box(25, 0)})
    assert pts[0][0] == pytest.approx(10.0)


def test_interpolation_is_linear_in_position():
    mid = interpolate_boxes({'a': box(0, 0)}, {'a': box(10, 0)}, 0.5)
    assert mid['a'][0] == pytest.approx(5.0)


def test_interpolation_takes_the_shortest_angular_path():
    """Across the +-pi wrap the box must not spin the long way round."""
    mid = interpolate_boxes({'a': box(0, 0, yaw=3.0)}, {'a': box(0, 0, yaw=-3.0)}, 0.5)
    assert abs(abs(mid['a'][6]) - np.pi) < 0.15, 'interpolated %.3f' % mid['a'][6]


def test_a_track_in_only_one_keyframe_is_dropped():
    """It has no defined position in between, and inventing one sweeps up stray points."""
    assert interpolate_boxes({'a': box(0, 0)}, {'b': box(1, 0)}, 0.5) == {}


def test_boxes_to_frame_is_identity_between_a_frame_and_itself():
    S = np.eye(4); S[:3, 3] = [5.0, -2.0, 1.0]
    got = boxes_to_frame(np.array([box(3, 4, yaw=0.5)]), np.array(['car']), ['t0'], S, S)
    assert np.allclose(got['t0'], box(3, 4, yaw=0.5))


def test_boxes_to_frame_applies_the_relative_transform():
    """Two ego frames 10 m apart put the same global box 10 m differently."""
    A = np.eye(4)
    B = np.eye(4); B[0, 3] = -10.0          # ego B sits 10 m further along x in global
    got = boxes_to_frame(np.array([box(0, 0)]), np.array(['car']), ['t0'], A, B)
    assert got['t0'][0] == pytest.approx(-10.0)


def test_classes_filter_excludes_untracked_categories():
    got = boxes_to_frame(np.array([box(0, 0), box(5, 0)]), np.array(['car', 'barrier']),
                         ['t0', 't1'], np.eye(4), np.eye(4), classes={'car'})
    assert list(got) == ['t0']


def test_boxes_without_a_track_id_are_skipped():
    got = boxes_to_frame(np.array([box(0, 0)]), np.array(['car']), [None], np.eye(4), np.eye(4))
    assert got == {}


# --------------------------------------------------------------------------------------------
# The key must be refused where it cannot be honoured. A config key a loader silently ignores is
# worse than one that fails: the run looks like it did what was asked and is quietly a different
# experiment. This project has paid for that failure mode more than once - a correction configured
# but never installed, an N=15 that was really N=10.
# --------------------------------------------------------------------------------------------

from pcdet.datasets.motion_compensation import assert_not_supported  # noqa: E402


class _Cfg(dict):
    def get(self, k, d=None):
        return dict.get(self, k, d)


def test_kitti_raises_rather_than_ignoring_the_key():
    with pytest.raises(NotImplementedError) as e:
        assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=True), 'KittiDataset')
    msg = str(e.value)
    assert 'KittiDataset' in msg
    assert 'no sequences' in msg, 'the reason KITTI can never support it'


def test_the_message_says_the_other_four_DO_support_it():
    """The refusal used to cover Waymo and PandaSet too; both now accumulate, so it must not."""
    with pytest.raises(NotImplementedError) as e:
        assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=True), 'KittiDataset')
    msg = str(e.value)
    for name in ('nuScenes', 'Lyft', 'Waymo', 'PandaSet'):
        assert name in msg, '%s supports it and the message should say so' % name
    assert 'obj_ids' in msg and 'uuid' in msg and 'instance_token' in msg


def test_absent_or_false_is_silent():
    assert_not_supported(_Cfg(), 'KittiDataset')
    assert_not_supported(_Cfg(GT_BOXES_MOTION_COMPENSATION=False), 'KittiDataset')


def test_kitti_is_the_only_loader_still_guarded():
    root = Path(__file__).resolve().parent.parent
    guarded = {m for m in ('kitti/kitti_dataset', 'waymo/waymo_dataset',
                           'pandaset/pandaset_dataset', 'nuscenes/nuscenes_dataset',
                           'lyft/lyft_dataset')
               if 'assert_not_supported(self.dataset_cfg' in
               (root / 'pcdet' / 'datasets' / (m + '.py')).read_text(encoding='utf-8')}
    assert guarded == {'kitti/kitti_dataset'}, guarded


# --------------------------------------------------------------------------------------------
# Neighbouring-frame accumulation, for the datasets that annotate every frame.

from pcdet.datasets.motion_compensation import (  # noqa: E402
    build_sequence_index, preceding_frames)


def _seq_infos(seq, frames):
    return [{'s': seq, 'f': f} for f in frames]


def test_the_index_is_keyed_by_sequence_and_frame_not_by_position():
    infos = _seq_infos('a', [0, 1, 2]) + _seq_infos('b', [0, 1])
    idx = build_sequence_index(infos, lambda i: i['s'], lambda i: i['f'])
    assert idx[('a', 2)] == 2 and idx[('b', 0)] == 3
    assert len(idx) == len(infos)


def test_preceding_frames_are_nearest_first_and_within_the_sequence():
    infos = _seq_infos('a', range(5)) + _seq_infos('b', range(5))
    idx = build_sequence_index(infos, lambda i: i['s'], lambda i: i['f'])
    got = preceding_frames(idx, 'a', 4, max_sweeps=3)
    assert [infos[j]['f'] for j in got] == [3, 2]
    assert all(infos[j]['s'] == 'a' for j in got)


def test_the_start_of_a_sequence_simply_yields_fewer_frames():
    idx = build_sequence_index(_seq_infos('a', range(5)), lambda i: i['s'], lambda i: i['f'])
    assert preceding_frames(idx, 'a', 1, max_sweeps=5) == [0]
    assert preceding_frames(idx, 'a', 0, max_sweeps=5) == []


def test_a_gap_stops_the_walk_rather_than_being_skipped():
    """Jumping a missing frame would accumulate across a discontinuity in ego motion."""
    idx = build_sequence_index(_seq_infos('a', [0, 1, 3, 4]), lambda i: i['s'], lambda i: i['f'])
    got = preceding_frames(idx, 'a', 4, max_sweeps=5)
    assert [i for i in got] == [idx[('a', 3)]], 'frame 2 is missing, so the walk must stop there'


def test_max_sweeps_of_one_asks_for_nothing():
    idx = build_sequence_index(_seq_infos('a', range(5)), lambda i: i['s'], lambda i: i['f'])
    assert preceding_frames(idx, 'a', 4, max_sweeps=1) == []


@pytest.mark.parametrize('mod', [
    'pcdet/datasets/waymo/waymo_dataset.py',
    'pcdet/datasets/pandaset/pandaset_dataset.py',
])
def test_the_per_frame_annotated_loaders_accumulate_and_compensate(mod):
    src = (Path(__file__).resolve().parent.parent / mod).read_text(encoding='utf-8')
    assert 'build_sequence_index' in src and 'preceding_frames' in src, 'must accumulate'
    assert 'should_compensate(self.dataset_cfg, self.training, self.logger)' in src, \
        'and must go through the shared rule rather than its own condition'
    assert 'move_points_between_boxes' in src


def test_waymo_indexes_sweeps_before_SAMPLED_INTERVAL_thins_the_infos():
    """SAMPLED_INTERVAL keeps every 2nd training frame; an index built after it finds no
    neighbour at all, and accumulation would be a silent no-op."""
    src = (Path(__file__).resolve().parent.parent
           / 'pcdet/datasets/waymo/waymo_dataset.py').read_text(encoding='utf-8')
    i_index = src.index('build_sequence_index(')
    i_sample = src.index('if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:')
    assert i_index < i_sample, 'the index must be built from the unsampled list'


# --------------------------------------------------------------------------------------------
# The compensator over devkit metadata: timeline units, bracketing, index alignment.

import json  # noqa: E402

from pcdet.datasets.motion_compensation import (  # noqa: E402
    DevkitSweepCompensator, find_devkit_meta_dir)

EPOCH = 1537932026.0  # a real nuScenes keyframe time, in SECONDS - what the info builders store


def _meta(tmp_path, n_frames=4, n_scenes=1):
    """A minimal devkit metadata dir plus matching infos: one track, moving +10 m/s in x."""
    d = tmp_path / 'v1.0-trainval'
    d.mkdir(parents=True, exist_ok=True)
    samples, anns, infos = [], [], []
    for s in range(n_scenes):
        for k in range(n_frames):
            tok, ann = 'sample-%d-%d' % (s, k), 'ann-%d-%d' % (s, k)
            samples.append({'token': tok, 'scene_token': 'scene-%d' % s})
            anns.append({'token': ann, 'instance_token': 'inst-0'})
            infos.append({
                'token': tok,
                'timestamp': EPOCH + 0.5 * k,          # keyframes are 2 Hz
                'ref_from_car': np.eye(4),
                'car_from_global': np.eye(4),
                'gt_boxes': np.array([[10.0 * (0.5 * k), 0, 0, 4, 2, 2, 0]]),
                'gt_names': np.array(['car']),
                'gt_boxes_token': np.array([ann]),
            })
    json.dump(samples, open(d / 'sample.json', 'w'))
    json.dump(anns, open(d / 'sample_annotation.json', 'w'))
    return d, infos


def test_the_timeline_is_in_seconds_not_microseconds(tmp_path):
    """The info builders already divide by 1e6; dividing again collapses the scene to an instant.

    That is not a small error. It compressed a 20 s scene into 1.2 ms, so every keyframe landed on
    the same time, the bracketing search returned an arbitrary one, and a parked car appeared to
    move 136 m in 0.3 s.
    """
    d, infos = _meta(tmp_path, n_frames=4)
    c = DevkitSweepCompensator(infos, d)
    ts = np.array([f['t'] for f in c.frames])
    assert np.allclose(np.diff(ts), 0.5), 'keyframe spacing must survive as 0.5 s'


def test_microsecond_timestamps_are_accepted_too(tmp_path):
    """Epoch seconds are ~1.5e9 and epoch microseconds ~1.5e15, so the two are never ambiguous."""
    d, infos = _meta(tmp_path, n_frames=4)
    for info in infos:
        info['timestamp'] = info['timestamp'] * 1e6
    c = DevkitSweepCompensator(infos, d)
    assert np.allclose(np.diff([f['t'] for f in c.frames]), 0.5)


def test_a_sweep_between_keyframes_is_interpolated(tmp_path):
    d, infos = _meta(tmp_path, n_frames=4)
    c = DevkitSweepCompensator(infos, d)
    anchor = infos[2]                                    # t = EPOCH + 1.0, box at x = 10
    S = np.eye(4)
    now = c.boxes_at(anchor['token'], EPOCH + 1.0, S)
    then = c.boxes_at(anchor['token'], EPOCH + 0.75, S)  # half a keyframe gap back
    assert np.isclose(now['inst-0'][0], 10.0)
    assert np.isclose(then['inst-0'][0], 7.5), 'must interpolate, not clamp to a keyframe'


def test_displacement_grows_with_the_sweep_lag(tmp_path):
    """The symptom that exposed the unit bug was a median move identical at every lag."""
    d, infos = _meta(tmp_path, n_frames=4)
    c = DevkitSweepCompensator(infos, d)
    anchor, S = infos[3], np.eye(4)
    at = EPOCH + 1.5
    now = c.boxes_at(anchor['token'], at, S)
    moves = [abs(now['inst-0'][0] - c.boxes_at(anchor['token'], at - lag, S)['inst-0'][0])
             for lag in (0.25, 0.5, 0.75)]
    assert moves == sorted(moves) and moves[0] < moves[-1], moves
    assert np.allclose(moves, [2.5, 5.0, 7.5])


def test_a_skipped_info_does_not_shift_the_frame_index(tmp_path):
    """tok2frame indexes self.frames, which an info with an unknown scene never enters."""
    d, infos = _meta(tmp_path, n_frames=4)
    infos.insert(0, dict(infos[0], token='not-in-sample-json'))
    c = DevkitSweepCompensator(infos, d)
    assert len(c.frames) == 4 and 'not-in-sample-json' not in c.tok2frame
    for info in infos[1:]:
        assert np.isclose(c.frames[c.tok2frame[info['token']]]['t'], info['timestamp'])


def test_scenes_do_not_bracket_across_each_other(tmp_path):
    d, infos = _meta(tmp_path, n_frames=4, n_scenes=2)
    c = DevkitSweepCompensator(infos, d)
    assert len(c.scene_order) == 2
    assert all(len(v) == 4 for v in c.scene_order.values())


def test_find_devkit_meta_dir_handles_both_layouts(tmp_path):
    for root, sub in ((tmp_path / 'nusc', 'v1.0-trainval'), (tmp_path / 'lyft', 'data')):
        (root / sub).mkdir(parents=True)
        for name in ('sample.json', 'sample_annotation.json'):
            (root / sub / name).write_text('[]')
        assert find_devkit_meta_dir(root, 'v1.0-trainval') == root / sub


def test_find_devkit_meta_dir_names_what_it_tried(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        find_devkit_meta_dir(tmp_path, 'v1.0-trainval')
    assert 'sample_annotation.json' in str(e.value) and str(tmp_path) in str(e.value)


# --------------------------------------------------------------------------------------------
# The rule: accumulation and compensation ship together.

from pcdet.datasets.motion_compensation import should_compensate  # noqa: E402


class _Log:
    def __init__(self):
        self.info_msgs, self.warnings = [], []

    def info(self, m):
        self.info_msgs.append(m)

    def warning(self, m):
        self.warnings.append(m)


def test_accumulation_turns_compensation_on_without_being_asked():
    """The point of the rule: a config that accumulates cannot forget to compensate."""
    assert should_compensate(_Cfg(MAX_SWEEPS=15), training=True) is True


def test_a_single_frame_does_not_compensate():
    assert should_compensate(_Cfg(MAX_SWEEPS=1), training=True) is False
    assert should_compensate(_Cfg(), training=True) is False


def test_an_eval_dataset_never_compensates():
    assert should_compensate(_Cfg(MAX_SWEEPS=15), training=False) is False


def test_a_pseudo_label_target_never_compensates_even_at_depth():
    """USE_PSEUDO_LABEL marks the unlabelled target, which is also built with training=True."""
    assert should_compensate(_Cfg(MAX_SWEEPS=15, USE_PSEUDO_LABEL=True), training=True) is False


def test_asking_for_it_on_a_pseudo_label_target_RAISES():
    """Not a warning: its infos carry real gt_boxes, so this would consume the target's labels."""
    with pytest.raises(ValueError) as e:
        should_compensate(_Cfg(MAX_SWEEPS=15, USE_PSEUDO_LABEL=True,
                               GT_BOXES_MOTION_COMPENSATION=True), training=True)
    assert 'USE_PSEUDO_LABEL' in str(e.value) and 'SOURCE' in str(e.value)


def test_explicit_false_is_honoured_but_warns():
    log = _Log()
    cfg = _Cfg(MAX_SWEEPS=15, GT_BOXES_MOTION_COMPENSATION=False)
    assert should_compensate(cfg, training=True, logger=log) is False
    assert len(log.warnings) == 1 and 'ablation' in log.warnings[0]


def test_turning_it_on_by_default_is_logged():
    log = _Log()
    assert should_compensate(_Cfg(MAX_SWEEPS=15), training=True, logger=log) is True
    assert any('ON by default' in m for m in log.info_msgs)


def test_asking_when_it_cannot_apply_says_why():
    log = _Log()
    should_compensate(_Cfg(MAX_SWEEPS=1, GT_BOXES_MOTION_COMPENSATION=True),
                      training=True, logger=log)
    assert any('nothing to compensate' in m for m in log.info_msgs)


@pytest.mark.parametrize('mod', [
    'pcdet/datasets/nuscenes/nuscenes_dataset.py',
    'pcdet/datasets/lyft/lyft_dataset.py',
])
def test_both_accumulating_loaders_go_through_the_rule(mod):
    """A loader with its own inline condition would drift from the rule the tests above pin."""
    src = (Path(__file__).resolve().parent.parent / mod).read_text(encoding='utf-8')
    assert 'if should_compensate(self.dataset_cfg, self.training, self.logger):' in src
    assert "self.dataset_cfg.get('GT_BOXES_MOTION_COMPENSATION'" not in src, \
        'the loader must not re-implement the decision'
