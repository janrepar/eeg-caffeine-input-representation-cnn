"""Small regression tests for validation and statistical-reporting helpers."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.train_eval.hyperparameters import aggregate_loso_hyperparameters
from src.train_eval.run_statistical_tests import bonferroni_adjust
from src.train_eval.split import create_groupkfold_splits


class CoreLogicTests(unittest.TestCase):
    def test_groupkfold_keeps_subjects_in_one_test_fold(self):
        subjects = np.repeat(np.array([311, 315, 340, 342, 346, 368]), 3)
        splits = create_groupkfold_splits(subjects, n_splits=3)

        test_counts = np.zeros(len(subjects), dtype=int)
        for split in splits:
            self.assertFalse(np.any(split["train_mask"] & split["test_mask"]))
            test_counts += split["test_mask"].astype(int)
            for subject in np.unique(subjects):
                subject_mask = subjects == subject
                self.assertIn(int(split["test_mask"][subject_mask].sum()), (0, subject_mask.sum()))

        np.testing.assert_array_equal(test_counts, np.ones(len(subjects), dtype=int))

    def test_loso_hyperparameters_use_median_and_mode(self):
        records = [
            {"outer_test_subject": "311", "best_params": {"learning_rate": 0.001, "filters": 8, "dropout": 0.2}},
            {"outer_test_subject": "315", "best_params": {"learning_rate": 0.003, "filters": 16, "dropout": 0.4}},
            {"outer_test_subject": "340", "best_params": {"learning_rate": 0.005, "filters": 16, "dropout": 0.6}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "nested_optimization_summary.json"
            summary.write_text(json.dumps(records), encoding="utf-8")
            parameters, metadata = aggregate_loso_hyperparameters(summary)

        self.assertEqual(parameters["filters"], 16)
        self.assertEqual(parameters["learning_rate"], 0.003)
        self.assertEqual(parameters["dropout"], 0.4)
        self.assertEqual(metadata["aggregation_by_parameter"]["filters"], "mode")
        self.assertEqual(metadata["aggregation_by_parameter"]["learning_rate"], "median")

    def test_bonferroni_adjustment_is_bounded(self):
        adjusted = bonferroni_adjust(np.array([0.01, 0.4, 0.8]))
        np.testing.assert_allclose(adjusted, np.array([0.04, 1.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
