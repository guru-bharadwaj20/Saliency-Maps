import numpy as np

from saliency_driving.metrics import (
    adaptive_fmeasure,
    average_precision,
    compute_iou,
    evaluate_detections,
    mae,
    s_measure,
)


class TestIoU:
    def test_identical(self):
        assert compute_iou([0, 0, 10, 10], [0, 0, 10, 10]) > 0.999

    def test_disjoint(self):
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_half_overlap(self):
        assert abs(compute_iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-6


class TestAveragePrecision:
    GTS = {"a": [[0, 0, 10, 10], [20, 20, 40, 40]], "b": [[5, 5, 15, 15]]}

    def test_perfect_predictions(self):
        preds = {n: [[*b, 0.9, 1] for b in boxes] for n, boxes in self.GTS.items()}
        ap, _, _ = average_precision(preds, self.GTS)
        assert abs(ap - 1.0) < 1e-6

    def test_all_wrong(self):
        bad = {"a": [[100, 100, 110, 110, 0.9, 1]], "b": []}
        ap, _, _ = average_precision(bad, self.GTS)
        assert ap == 0.0

    def test_no_predictions(self):
        ap, recall, precision = average_precision({}, self.GTS)
        assert ap == 0.0 and recall == 0.0

    def test_report_f1(self):
        preds = {n: [[*b, 0.9, 1] for b in boxes] for n, boxes in self.GTS.items()}
        report = evaluate_detections(preds, self.GTS)
        assert abs(report["F1"] - 1.0) < 1e-3
        assert report["total_GT"] == 3


class TestSaliencyMetrics:
    def test_mae_identical(self, square_gt):
        assert mae(square_gt, square_gt) == 0.0

    def test_mae_known_value(self, square_gt):
        assert abs(mae(np.zeros_like(square_gt), square_gt) - 0.16) < 1e-6

    def test_fmeasure_perfect(self, square_gt):
        assert adaptive_fmeasure(square_gt, square_gt) > 0.99

    def test_smeasure_perfect(self, square_gt):
        assert s_measure(square_gt, square_gt) > 0.95

    def test_smeasure_orders_predictions(self, square_gt):
        noisy = np.clip(square_gt + np.random.RandomState(0).normal(0, 0.1, square_gt.shape),
                        0, 1).astype(np.float32)
        assert s_measure(noisy, square_gt) > s_measure(1 - square_gt, square_gt)

    def test_smeasure_empty_gt(self):
        empty = np.zeros((50, 50), np.float32)
        assert s_measure(empty, empty) == 1.0
