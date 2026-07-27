import json
from pathlib import Path
import tempfile
import unittest

import imageio.v3 as iio
import cv2
import numpy as np

from scripts.postprocess_real_run import (
    HEADER_HEIGHT,
    compose_video_frame,
    default_fps,
    detect_box_centroid,
    detect_box_pose,
    detect_goal_gripper_tip,
    load_trial,
    orientation_goal_errors_deg,
    resolve_run_path,
    track_gripper_tip,
    tracking_samples,
)


class PostprocessRealRunTests(unittest.TestCase):
    @staticmethod
    def tracking_frame(tip_u: int, tip_v: int) -> np.ndarray:
        frame = np.full((224, 224, 3), 180, dtype=np.uint8)
        frame[tip_v - 10 : tip_v + 5, tip_u - 8 : tip_u + 9] = (25, 25, 25)
        frame[tip_v - 3 : tip_v + 5, tip_u - 3 : tip_u + 4] = (105, 25, 20)
        frame[90:110, 130:155] = (150, 60, 50)
        return frame

    def test_detects_box_patch_and_small_red_gripper_tip(self):
        frame = self.tracking_frame(80, 72)

        np.testing.assert_allclose(detect_box_centroid(frame), (142, 99.5), atol=1)
        np.testing.assert_allclose(detect_goal_gripper_tip(frame), (80, 72.5), atol=1)

    def test_box_pose_fits_corners_and_long_axis_modulo_180(self):
        frame = np.full((224, 224, 3), 180, dtype=np.uint8)
        expected_center = np.array([125.0, 105.0], dtype=np.float32)
        expected_angle_deg = 32.0
        corners = cv2.boxPoints(
            ((float(expected_center[0]), float(expected_center[1])), (42.0, 8.0), expected_angle_deg)
        )
        cv2.fillConvexPoly(
            frame, np.rint(corners).astype(np.int32), (150, 60, 50)
        )

        center, fitted_corners, orientation_deg = detect_box_pose(frame)
        angle_error = orientation_goal_errors_deg(
            np.asarray([orientation_deg], dtype=np.float32), expected_angle_deg
        )[0]

        np.testing.assert_allclose(center, expected_center, atol=1.0)
        self.assertEqual(fitted_corners.shape, (4, 2))
        self.assertTrue(np.isfinite(fitted_corners).all())
        self.assertLess(float(angle_error), 2.0)

    def test_gripper_track_uses_continuity_to_reject_lower_distractor(self):
        first = self.tracking_frame(80, 72)
        second = self.tracking_frame(82, 73)
        second[102:118, 174:190] = (25, 25, 25)
        second[108:116, 179:186] = (105, 25, 20)

        track = track_gripper_tip([first, second])

        np.testing.assert_allclose(track[0], (80, 72.5), atol=1)
        np.testing.assert_allclose(track[1], (82, 73.5), atol=1)

    def test_bare_run_id_resolves_under_real_robot_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_root = Path(directory)
            expected = repo_root / "real_robot_runs" / "run_001"
            expected.mkdir(parents=True)
            self.assertEqual(
                resolve_run_path(Path("run_001"), repo_root=repo_root),
                expected.resolve(),
            )

    def test_loads_aligned_trial_frames_costs_and_elapsed_time(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            frames_path = run_path / "frames"
            frames_path.mkdir()
            goal = np.full((224, 224, 3), 90, dtype=np.uint8)
            first = np.full((224, 224, 3), 10, dtype=np.uint8)
            second = np.full((224, 224, 3), 20, dtype=np.uint8)
            iio.imwrite(frames_path / "trial_001_goal.png", goal)
            iio.imwrite(frames_path / "trial_001_step_001.png", first)
            iio.imwrite(frames_path / "trial_001_step_002.png", second)
            terminal = np.full((224, 224, 3), 30, dtype=np.uint8)
            iio.imwrite(frames_path / "trial_001_terminal.png", terminal)
            (run_path / "metadata.json").write_text(
                json.dumps(
                    {"artifact_manifest": {"runtime": {"tick_hz": 7.0}}}
                )
            )
            events = [
                {
                    "type": "trial_started",
                    "trial_id": 1,
                    "goal_frame": "frames/trial_001_goal.png",
                },
                {
                    "type": "autonomous_step",
                    "trial_id": 1,
                    "step": 1,
                    "monotonic_ns": 1_000_000_000,
                    "observation_frame": "frames/trial_001_step_001.png",
                    "cost": 2.0,
                    "goal_distance": 3.0,
                    "solve_time_s": 0.4,
                },
                {
                    "type": "autonomous_step",
                    "trial_id": 1,
                    "step": 2,
                    "monotonic_ns": 2_500_000_000,
                    "observation_frame": "frames/trial_001_step_002.png",
                    "cost": 1.5,
                    "goal_distance": 2.5,
                    "solve_time_s": 0.3,
                },
                {
                    "type": "trial_outcome",
                    "trial_id": 1,
                    "outcome": "success",
                    "monotonic_ns": 3_000_000_000,
                    "terminal_frame": "frames/trial_001_terminal.png",
                },
            ]
            (run_path / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )

            trial = load_trial(run_path, 1)
            self.assertEqual([record.step for record in trial.steps], [1, 2])
            self.assertEqual([record.cost for record in trial.steps], [2.0, 1.5])
            self.assertEqual(
                [record.elapsed_s for record in trial.steps], [0.0, 1.5]
            )
            self.assertEqual(trial.outcome, "success")
            self.assertEqual(trial.terminal_path, (frames_path / "trial_001_terminal.png").resolve())
            self.assertEqual(trial.terminal_elapsed_s, 2.0)
            samples = tracking_samples(trial)
            self.assertEqual(samples[-1][0], "terminal_post_action")
            self.assertEqual(samples[-1][1], 2)
            self.assertEqual(samples[-1][2], 2.0)
            self.assertEqual(default_fps(trial.metadata), 7.0)

            composed = compose_video_frame(first, goal, step=1, cost=2.0)
            self.assertEqual(composed.shape, (260, 448, 3))
            np.testing.assert_array_equal(
                composed[HEADER_HEIGHT:, :224], first
            )
            np.testing.assert_array_equal(
                composed[HEADER_HEIGHT:, 224:], goal
            )

    def test_rejects_noncontiguous_action_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            frames_path = run_path / "frames"
            frames_path.mkdir()
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
            iio.imwrite(frames_path / "goal.png", frame)
            iio.imwrite(frames_path / "step.png", frame)
            (run_path / "metadata.json").write_text("{}")
            events = [
                {
                    "type": "trial_started",
                    "trial_id": 1,
                    "goal_frame": "frames/goal.png",
                },
                {
                    "type": "autonomous_step",
                    "trial_id": 1,
                    "step": 2,
                    "observation_frame": "frames/step.png",
                    "cost": 1.0,
                    "goal_distance": 1.0,
                    "solve_time_s": 0.1,
                },
            ]
            (run_path / "events.jsonl").write_text(
                "".join(json.dumps(event) + "\n" for event in events)
            )
            with self.assertRaisesRegex(ValueError, "contiguous"):
                load_trial(run_path, 1)


if __name__ == "__main__":
    unittest.main()
