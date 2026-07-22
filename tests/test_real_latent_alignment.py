import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from scripts.render_real_latent_alignment import render_alignment


class RealLatentAlignmentTests(unittest.TestCase):
    def test_render_has_three_aligned_rows_and_empty_first_prediction(self):
        raw = [
            np.full((224, 224, 3), value, dtype=np.uint8)
            for value in (20, 40, 60)
        ]
        decoded_z = np.stack(
            [np.full((224, 224, 3), value, dtype=np.uint8) for value in (80, 100, 120)]
        )
        decoded_z_hat = np.stack(
            [np.full((224, 224, 3), value, dtype=np.uint8) for value in (140, 160, 180)]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "alignment.png")
            render_alignment(
                path,
                raw,
                decoded_z,
                decoded_z_hat,
                tile_size=64,
            )
            image = np.asarray(Image.open(path))

        self.assertEqual(image.shape, (300, 382, 3))
        # Raw and decoded-z cells line up at the same time-column centers.
        np.testing.assert_array_equal(image[62, 222], [20, 20, 20])
        np.testing.assert_array_equal(image[152, 286], [100, 100, 100])
        # At t=1, row three uses z_hat from the preceding action at t=0.
        np.testing.assert_array_equal(image[242, 286], [140, 140, 140])
        # The terminal z_hat is intentionally not shifted under an absent t=3 frame.
        self.assertEqual(image.shape[1], 190 + 3 * 64)


if __name__ == "__main__":
    unittest.main()
