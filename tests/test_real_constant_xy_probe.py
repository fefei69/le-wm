import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import imageio.v3 as iio
import numpy as np

from scripts.record_real_constant_xy import (
    ProbeRecorder,
    build_geometry,
    load_source_event,
)


class ConstantXYGeometryTests(unittest.TestCase):
    def test_default_trajectory_has_51_states_and_expected_endpoint(self):
        geometry = build_geometry(
            start_xy=(0.1898409503403485, 0.02137995091021234),
            action_xy=(0.005, 0.0),
            steps=50,
            x_bounds=(0.0183, 0.45),
            y_bounds=(-0.26, 0.26),
            boundary_margin_m=0.005,
        )
        self.assertEqual(geometry.targets_xy.shape, (51, 2))
        np.testing.assert_allclose(
            geometry.final_xy,
            [0.4398409503403485, 0.02137995091021234],
            rtol=0.0,
            atol=1e-12,
        )
        self.assertGreater(0.45 - geometry.final_xy[0], 0.010)

    def test_trajectory_is_rejected_before_crossing_safety_margin(self):
        with self.assertRaisesRegex(ValueError, "outside the workspace"):
            build_geometry(
                start_xy=(0.44, 0.0),
                action_xy=(0.005, 0.0),
                steps=2,
                x_bounds=(0.0183, 0.45),
                y_bounds=(-0.26, 0.26),
                boundary_margin_m=0.005,
            )

    def test_action_above_collection_cap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "action norm"):
            build_geometry(
                start_xy=(0.20, 0.0),
                action_xy=(0.011, 0.0),
                steps=1,
                x_bounds=(0.0183, 0.45),
                y_bounds=(-0.26, 0.26),
                boundary_margin_m=0.0,
            )


class ConstantXYArtifactTests(unittest.TestCase):
    def test_source_event_uses_pre_action_measured_pose(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            frames = run / "frames"
            frames.mkdir()
            reference = frames / "trial_001_step_007.png"
            iio.imwrite(reference, np.zeros((224, 224, 3), dtype=np.uint8))
            event = {
                "type": "autonomous_step",
                "observation_frame": "frames/trial_001_step_007.npy",
                "measured_pose_before": [0.19, 0.02, 0.03, 0.0, 1.57, 0.0],
                "target_xy": [0.20, 0.02],
            }
            (run / "events.jsonl").write_text(json.dumps(event) + "\n")
            self.assertEqual(load_source_event(reference), event)

    @mock.patch("scripts.record_real_constant_xy.write_rgb_video")
    def test_recorder_persists_aligned_png_and_state_event(self, write_video):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            goal = root / "goal.png"
            model = np.zeros((224, 224, 3), dtype=np.uint8)
            raw = np.zeros((480, 640, 3), dtype=np.uint8)
            iio.imwrite(source, model)
            iio.imwrite(goal, model)
            recorder = ProbeRecorder(
                root / "run",
                {"trajectory": {"fps": 5.0}},
                model,
                source,
                model,
                goal,
            )
            recorder.state(0, model, raw, target_xy_m=[0.19, 0.02])
            recorder.close(status="completed")

            self.assertTrue((root / "run/frames/frame_000.png").is_file())
            self.assertTrue((root / "run/raw_frames/frame_000.png").is_file())
            events = [
                json.loads(line)
                for line in (root / "run/events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(events[0]["type"], "state_captured")
            self.assertEqual(events[0]["step"], 0)
            self.assertEqual(events[-1]["captured_state_count"], 1)
            self.assertEqual(write_video.call_count, 2)


if __name__ == "__main__":
    unittest.main()
