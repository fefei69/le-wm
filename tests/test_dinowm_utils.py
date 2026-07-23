import unittest

import numpy as np
import torch

from dinowm_utils import compute_column_stats, select_nested_episode_subset


class FakeDataset:
    def __init__(self):
        self.lengths = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        self.offsets = np.concatenate(([0], np.cumsum(self.lengths[:-1])))
        self.clip_indices = [
            (episode, start)
            for episode, length in enumerate(self.lengths)
            for start in range(length - 3)
        ]
        rows = int(self.lengths.sum())
        self.columns = {
            "signal": np.stack(
                [np.arange(rows, dtype=np.float32), np.ones(rows)], axis=1
            )
        }

    def get_col_data(self, column):
        return self.columns[column]


class DinoWMUtilsTest(unittest.TestCase):
    def test_duration_subsets_are_episode_atomic_and_nested(self):
        dataset = FakeDataset()
        candidates = np.arange(5)
        small = select_nested_episode_subset(
            dataset, candidates, target_hours=0.04, sample_hz=1, seed=7
        )
        medium = select_nested_episode_subset(
            dataset, candidates, target_hours=0.18, sample_hz=1, seed=7
        )
        full = select_nested_episode_subset(
            dataset, candidates, target_hours=1, sample_hz=1, seed=7
        )

        self.assertLessEqual(
            set(small.episode_indices), set(medium.episode_indices)
        )
        self.assertLessEqual(
            set(medium.episode_indices), set(full.episode_indices)
        )
        self.assertEqual(set(full.episode_indices), set(candidates))
        self.assertTrue(
            all(
                dataset.clip_indices[index][0] in set(small.episode_indices)
                for index in small.clip_indices
            )
        )

    def test_column_stats_use_only_selected_episode_rows(self):
        dataset = FakeDataset()
        stats = compute_column_stats(dataset, "signal", [1, 3])
        rows = np.concatenate(
            [
                np.arange(
                    dataset.offsets[episode],
                    dataset.offsets[episode] + dataset.lengths[episode],
                )
                for episode in (1, 3)
            ]
        )
        expected = torch.from_numpy(dataset.columns["signal"][rows]).float()

        torch.testing.assert_close(stats.mean, expected.mean(0, keepdim=True))
        torch.testing.assert_close(
            stats.std[:, :1], expected.std(0, keepdim=True)[:, :1]
        )
        # A constant dimension is made safe for z-score normalization.
        torch.testing.assert_close(stats.std[:, 1:], torch.ones(1, 1))


if __name__ == "__main__":
    unittest.main()
