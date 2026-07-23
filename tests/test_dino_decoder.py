import unittest

import torch

from dino_decoder import DinoPatchDecoder
from train_dino_decoder import indices_for_episodes


class FakeDataset:
    clip_indices = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)]


class DinoPatchDecoderTest(unittest.TestCase):
    def setUp(self):
        self.decoder = DinoPatchDecoder(
            embedding_dim=8,
            image_size=31,
            channels=16,
            num_residual_blocks=1,
            residual_channels=4,
        )

    def test_decodes_frame_tokens_and_resizes(self):
        output = self.decoder(torch.randn(2, 4, 8))
        self.assertEqual(output.shape, (2, 3, 31, 31))

    def test_preserves_temporal_dimensions(self):
        output = self.decoder(torch.randn(2, 3, 4, 8))
        self.assertEqual(output.shape, (2, 3, 3, 31, 31))

    def test_rejects_non_square_token_grid(self):
        with self.assertRaisesRegex(ValueError, "must be square"):
            self.decoder(torch.randn(2, 5, 8))

    def test_selects_all_clips_from_requested_episodes(self):
        self.assertEqual(indices_for_episodes(FakeDataset(), [0, 2]), [0, 1, 3, 4])


if __name__ == "__main__":
    unittest.main()
