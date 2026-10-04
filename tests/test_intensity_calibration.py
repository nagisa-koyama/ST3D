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


class TestTestTimeMapStage:
    """DataProcessor.map_intensity_to_reference (the test-time calibration stage)."""

    def _stage(self, tmp_path, from_q, to_q, edges=(0.0, 75.0), step=0.0):
        from easydict import EasyDict
        from pcdet.datasets.processor.data_processor import DataProcessor
        path = tmp_path / 't.npz'
        np.savez(path, edges=np.asarray(edges), from_q=np.asarray(from_q), to_q=np.asarray(to_q), from_step=step)
        dp = DataProcessor.__new__(DataProcessor)
        return dp.map_intensity_to_reference(config=EasyDict({'TABLES': str(path), 'INTENSITY_INDEX': 3}))

    def test_identity_tables_leave_intensity_unchanged(self, tmp_path):
        L = np.linspace(0, 1, 201)
        stage = self._stage(tmp_path, [L], [L])
        pts = np.random.RandomState(0).rand(1000, 4).astype(np.float32); pts[:, :2] *= 50
        out = stage(data_dict={'points': pts.copy()})['points']
        assert np.allclose(out[:, 3], pts[:, 3], atol=1e-5) and np.array_equal(out[:, :3], pts[:, :3])

    def test_maps_onto_the_reference_distribution_per_ring(self, tmp_path):
        L = np.linspace(0, 1, 201)
        rs = np.random.RandomState(1)
        # ring 0 (0-20 m): this sensor uniform on [0, 0.2] -> reference uniform on [0.5, 1.0]
        # ring 1 (20-75 m): this sensor uniform on [0, 1] -> reference constant-ish low [0, 0.1]
        stage = self._stage(tmp_path, [L * 0.2, L], [0.5 + 0.5 * L, 0.1 * L], edges=(0.0, 20.0, 75.0))
        near = np.column_stack([rs.uniform(1, 19, 5000), np.zeros(5000), np.zeros(5000), rs.uniform(0, 0.2, 5000)])
        far = np.column_stack([rs.uniform(21, 70, 5000), np.zeros(5000), np.zeros(5000), rs.uniform(0, 1, 5000)])
        out = stage(data_dict={'points': np.vstack([near, far]).astype(np.float32)})['points']
        assert abs(np.median(out[:5000, 3]) - 0.75) < 0.02 and out[:5000, 3].min() >= 0.499
        assert out[5000:, 3].max() <= 0.1001 and abs(np.median(out[5000:, 3]) - 0.05) < 0.01

    def test_deterministic_for_the_same_frame(self, tmp_path):
        L = np.linspace(0, 1, 201)
        stage = self._stage(tmp_path, [L], [L ** 2], step=1 / 255.0)
        pts = (np.random.RandomState(2).rand(500, 4) * [40, 40, 1, 1]).astype(np.float32)
        a = stage(data_dict={'points': pts.copy()})['points']; b = stage(data_dict={'points': pts.copy()})['points']
        assert np.array_equal(a, b)

    def test_empty_frame(self, tmp_path):
        L = np.linspace(0, 1, 201)
        stage = self._stage(tmp_path, [L], [L])
        assert len(stage(data_dict={'points': np.zeros((0, 4), np.float32)})['points']) == 0


class TestAblateIntensityStage:
    """DataProcessor.ablate_intensity (eval-only diagnostic)."""

    def _stage(self, **cfg):
        from easydict import EasyDict
        from pcdet.datasets.processor.data_processor import DataProcessor
        dp = DataProcessor.__new__(DataProcessor)
        return dp.ablate_intensity(config=EasyDict(cfg))

    def test_shuffle_keeps_the_frame_distribution_and_geometry(self):
        pts = np.random.RandomState(3).rand(2000, 4).astype(np.float32)
        out = self._stage(MODE='shuffle')(data_dict={'points': pts.copy()})['points']
        assert np.array_equal(np.sort(out[:, 3]), np.sort(pts[:, 3]))
        assert np.array_equal(out[:, :3], pts[:, :3])
        assert np.mean(out[:, 3] == pts[:, 3]) < 0.01

    def test_shuffle_is_deterministic(self):
        pts = np.random.RandomState(4).rand(500, 4).astype(np.float32)
        st = self._stage(MODE='shuffle')
        assert np.array_equal(st(data_dict={'points': pts.copy()})['points'], st(data_dict={'points': pts.copy()})['points'])

    def test_constant(self):
        pts = np.random.RandomState(5).rand(100, 5).astype(np.float32)
        out = self._stage(MODE='constant', VALUE=0.25)(data_dict={'points': pts.copy()})['points']
        assert np.all(out[:, 3] == np.float32(0.25)) and np.array_equal(out[:, [0, 1, 2, 4]], pts[:, [0, 1, 2, 4]])

    def test_unknown_mode_refused(self):
        with pytest.raises(ValueError):
            self._stage(MODE='zero')

    def test_empty_frame(self):
        assert len(self._stage(MODE='shuffle')(data_dict={'points': np.zeros((0, 4), np.float32)})['points']) == 0
