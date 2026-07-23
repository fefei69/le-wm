import numpy as np
import unittest

from scripts.analyze_data_coverage import gripper_tip_candidates


class AnalyzeDataCoverageTests(unittest.TestCase):
    def test_gripper_tip_candidates_rejects_large_patch_and_unbacked_speckle(self):
        image = np.full((224, 224, 3), 180, dtype=np.uint8)

        # Dark gripper body with a small red pusher tip.
        image[62:78, 96:112] = (25, 25, 25)
        image[72:80, 101:108] = (105, 25, 20)

        # The box patch is much too large, while this isolated red speckle lacks
        # the dark gripper context.
        image[90:110, 130:155] = (150, 60, 50)
        image[115:118, 45:48] = (120, 25, 20)

        candidates = gripper_tip_candidates(image)

        self.assertEqual(candidates.shape, (1, 3))
        np.testing.assert_allclose(candidates[0, :2], (104.0, 75.5), atol=1.0)
        self.assertEqual(candidates[0, 2], 56)


if __name__ == "__main__":
    unittest.main()
