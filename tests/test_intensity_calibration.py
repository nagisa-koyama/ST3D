"""Intensity statistics and the per-(ring, channel) quantile map (pcdet/datasets/intensity_calibration.py)."""
import numpy as np
import pytest

from pcdet.datasets import intensity_calibration as ic


def _box(x, y, cls, size=4.0):
    return [x, y, 0.0, size, size, 2.0, 0.0, cls]


class TestPointChannels:
    def test_background_class_and_ignored(self):
        pts = np.array([[0, 0, 0, 0.1], [10, 0, 0, 0.2], [20, 0, 0, 0.3], [30, 0, 0, 0.4]], dtype=np.float32)
        boxes = np.array([_box(10, 0, 1), _box(20, 0, 2), _box(30, 0, -1)], dtype=np.float32)
        assert ic.point_channels(pts, boxes).tolist() == [0, 1, 2, -1]

    def test_no_boxes_is_all_background(self):
        pts = np.zeros((5, 4), dtype=np.float32)
        assert (ic.point_channels(pts, None) == 0).all()
        assert (ic.point_channels(pts, np.zeros((0, 8))) == 0).all()

    def test_degenerate_boxes_are_ignored(self):
        pts = np.array([[10, 0, 0, 0.1]], dtype=np.float32)
        boxes = np.array([[10, 0, 0, 0.0, 4, 2, 0, 1]], dtype=np.float32)
        assert ic.point_channels(pts, boxes).tolist() == [0]

    def test_class_column_is_required(self):
        with pytest.raises(ValueError, match='class column'):
            ic.point_channels(np.zeros((1, 4), np.float32), np.zeros((1, 7), np.float32))


class TestQuantilesAndMap:
    def _hist(self, samples):
        return np.histogram(samples, bins=ic.NUM_LEVELS, range=(0, 1))[0].astype(float)

    def test_quantiles_recover_a_known_distribution(self):
        x = np.random.RandomState(0).uniform(0.2, 0.6, 200000)
        q = ic.quantiles(self._hist(x), np.array([0.0, 0.5, 1.0]))
        assert np.allclose(q, [0.2, 0.4, 0.6], atol=2e-3)

    def test_empty_histogram_gives_nan(self):
        assert np.isnan(ic.quantiles(np.zeros(ic.NUM_LEVELS))).all()

    def test_map_transfers_the_target_distribution(self):
        rs = np.random.RandomState(1)
        src = rs.beta(2, 8, 100000); tgt = rs.uniform(0.1, 0.5, 100000)
        H = lambda a: self._hist(a).reshape(1, 1, -1)
        src_q, tgt_q, fb = ic.build_intensity_map(H(src), H(tgt), [[0]], min_points=100)
        out = ic.apply_intensity_map(src, np.zeros(len(src), int), np.zeros(len(src), int), src_q, tgt_q)
        assert fb.tolist() == [[0]]
        assert abs(np.median(out) - 0.3) < 0.01 and out.min() > 0.09 and out.max() < 0.51

    def test_sparse_group_falls_back_to_the_ring_pool_then_identity(self):
        h = np.zeros((2, 2, ic.NUM_LEVELS))
        h[0, 0, 100] = 5000; h[0, 1, 900] = 10      # ring 0: class channel too sparse -> pooled
        src_q, tgt_q, fb = ic.build_intensity_map(h, h, [[0], [1]], min_points=100)
        assert fb.tolist() == [[0, 1], [2, 2]]       # ring 1 empty -> identity

    def test_dequantisation_removes_the_comb(self):
        """Integer source levels must not map onto a few isolated target values."""
        rs = np.random.RandomState(2)
        raw = rs.choice([1, 2, 3, 5, 10], 50000).astype(float)
        src = ic.normalise(raw, scale=255, step=1.0, rng=rs)
        tgt = rs.uniform(0, 0.4, 50000)
        H = lambda a: self._hist(a).reshape(1, 1, -1)
        src_q, tgt_q, _ = ic.build_intensity_map(H(src), H(tgt), [[0]], min_points=100)
        out = ic.apply_intensity_map(src, np.zeros(len(src), int), np.zeros(len(src), int), src_q, tgt_q)
        occupied = np.histogram(out, bins=40, range=(0, 0.4))[0] > 0
        assert occupied.mean() > 0.9

    def test_group_hist_sums_channels(self):
        h = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
        g = ic.group_hist(h, [[1], [0, 3]])
        assert np.array_equal(g[:, 0], h[:, 1]) and np.array_equal(g[:, 1], h[:, 0] + h[:, 3])


class TestKeepRawPointsHook:
    def test_points_raw_keeps_every_column(self):
        from easydict import EasyDict
        from pcdet.datasets.dataset import DatasetTemplate

        class Enc:
            def forward(self, d):
                d['points'] = d['points'][:, :3]; return d

        class Proc:
            def forward(self, data_dict):
                return data_dict

        ds = DatasetTemplate.__new__(DatasetTemplate)
        ds.keep_raw_points = True
        ds.point_feature_encoder, ds.data_processor = Enc(), Proc()
        ds.training, ds.beam_distill_cfg, ds.beam_drop_cfg = False, None, None
        ds.map_ontology_dataset_to_model, ds.dataset_class_names, ds.class_names = None, ['Car'], ['Car']
        ds.dataset_cfg, ds.unsupervised = EasyDict({}), False
        pts = np.random.rand(10, 4).astype(np.float32)
        out = ds.prepare_data({'points': pts.copy()})
        assert out['points'].shape[1] == 3 and out['points_raw'].shape[1] == 4
        assert np.allclose(out['points_raw'], pts)
