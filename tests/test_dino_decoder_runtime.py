import torch
from pathlib import Path
import tempfile
import unittest

from dino_decoder import DinoPatchDecoder
from scripts.render_real_latent_alignment import (
    decode_dino_latents,
    load_dino_decoder,
)


class DinoDecoderRuntimeTests(unittest.TestCase):
    def test_decoder_accepts_flattened_runtime_patch_tokens(self):
        decoder = DinoPatchDecoder(
            embedding_dim=2,
            image_size=32,
            channels=4,
            num_residual_blocks=1,
            residual_channels=2,
        ).eval()
        flattened = torch.randn(3, 4 * 2)
        frames = decode_dino_latents(
            decoder,
            flattened,
            num_patches=4,
            embedding_dim=2,
            batch_size=2,
        )
        self.assertEqual(frames.shape, (3, 32, 32, 3))
        self.assertEqual(frames.dtype, torch.empty((), dtype=torch.uint8).numpy().dtype)

    def test_decoder_rejects_non_square_patch_count(self):
        decoder = DinoPatchDecoder(
            embedding_dim=2,
            image_size=32,
            channels=4,
            num_residual_blocks=1,
            residual_channels=2,
        )
        with self.assertRaisesRegex(ValueError, "square"):
            decoder(torch.randn(1, 3, 2))

    def test_runtime_loader_reads_hpc_decoder_checkpoint_format(self):
        expected = DinoPatchDecoder(
            embedding_dim=2,
            image_size=32,
            channels=4,
            num_residual_blocks=1,
            residual_channels=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "decoder_best.pt")
            torch.save(
                {
                    "state_dict": expected.state_dict(),
                    "args": {
                        "image_size": 32,
                        "channels": 4,
                        "residual_blocks": 1,
                        "residual_channels": 2,
                    },
                    "embedding_dim": 2,
                    "num_patches": 4,
                    "source_checkpoint": "/checkpoint/weights_epoch_10.pt",
                },
                path,
            )
            loaded, metadata = load_dino_decoder(path, torch.device("cpu"))
        self.assertEqual(loaded.embedding_dim, 2)
        self.assertEqual(loaded.image_size, 32)
        self.assertEqual(metadata["num_patches"], 4)
        self.assertEqual(
            metadata["checkpoint"], "/checkpoint/weights_epoch_10.pt"
        )


if __name__ == "__main__":
    unittest.main()
