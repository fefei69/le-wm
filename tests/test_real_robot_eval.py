import argparse
import imageio.v3 as iio
import json
import numpy as np
from pathlib import Path
import tempfile
import torch
import unittest

from real_robot_eval import (
    ACTION_CAP_M,
    DEFAULT_SETTLED_LINEAR_SPEED_M_S,
    GoalFrameSelectorState,
    PlannerProcess,
    RunRecorder,
    UIEvents,
    _workspace_bounds,
    load_external_goal,
    motion_completion_status,
    prioritize_ui_events,
    snapshot_follows_settle,
    transform_planner_action,
    validate_measured_pose,
    validate_planned_action,
    validate_planner_result,
    validate_transition_prediction,
)
from real_robot_planner import (
    CEMConfig,
    clamp_action_norm,
    keyboard_action_vocabulary,
    quantize_actions,
)


class RealRobotSafetyTests(unittest.TestCase):
    def test_goal_frame_selector_navigation_uses_one_or_five_frame_stride(self):
        state = GoalFrameSelectorState(frame_count=12)
        self.assertEqual(state.frame_index, 0)
        self.assertFalse(state.move(-1))
        self.assertTrue(state.move(1))
        self.assertEqual(state.frame_index, 1)

        state.set_stride(5)
        self.assertTrue(state.move(1))
        self.assertEqual(state.frame_index, 6)
        self.assertTrue(state.move(1))
        self.assertEqual(state.frame_index, 11)
        self.assertFalse(state.move(1))
        self.assertTrue(state.move(-1))
        self.assertEqual(state.frame_index, 6)

        with self.assertRaisesRegex(ValueError, "stride"):
            state.set_stride(2)
        with self.assertRaisesRegex(ValueError, "direction"):
            state.move(0)
        with self.assertRaisesRegex(ValueError, "at least one"):
            GoalFrameSelectorState(frame_count=0)

    def test_validate_planned_action_accepts_bounded_xy_delta(self):
        action = validate_planned_action([0.006, 0.008])
        np.testing.assert_allclose(action, [0.006, 0.008])
        self.assertEqual(action.dtype, np.float32)

    def test_validate_planned_action_rejects_unsafe_output(self):
        unsafe_actions = [
            [0.011, 0.0],
            [np.nan, 0.0],
            [np.inf, 0.0],
            [0.0],
            [0.0, 0.0, 0.0],
        ]
        for action in unsafe_actions:
            with self.subTest(action=action), self.assertRaises(ValueError):
                validate_planned_action(action)

    def test_negated_planner_action_is_bounded_and_self_inverse(self):
        planner_action = np.array([0.006, -0.008], dtype=np.float32)
        robot_action = transform_planner_action(
            planner_action, negate=True, cap_m=ACTION_CAP_M
        )
        np.testing.assert_allclose(robot_action, [-0.006, 0.008])
        planner_feedback = transform_planner_action(
            robot_action, negate=True, cap_m=ACTION_CAP_M
        )
        np.testing.assert_allclose(planner_feedback, planner_action)
        np.testing.assert_allclose(planner_action, [0.006, -0.008])

        with self.assertRaises(ValueError):
            transform_planner_action([0.011, 0.0], negate=True)

    def test_clamp_action_norm_preserves_direction_and_enforces_cap(self):
        actions = torch.tensor([[[0.006, 0.008], [0.03, 0.04], [0.0, 0.0]]])
        projected = clamp_action_norm(actions, ACTION_CAP_M)
        torch.testing.assert_close(projected[0, 0], torch.tensor([0.006, 0.008]))
        torch.testing.assert_close(projected[0, 1], torch.tensor([0.006, 0.008]))
        self.assertLessEqual(
            float(torch.linalg.vector_norm(projected, dim=-1).max()),
            ACTION_CAP_M,
        )

    def test_cem_configuration_refuses_cap_above_collection_contract(self):
        with self.assertRaisesRegex(ValueError, "action_cap_m"):
            CEMConfig(action_cap_m=ACTION_CAP_M + 0.001).validate()
        with self.assertRaisesRegex(ValueError, "finite"):
            CEMConfig(goal_tolerance=np.nan).validate()
        with self.assertRaisesRegex(ValueError, "action_mode"):
            CEMConfig(action_mode="unknown").validate()

    def test_keyboard_vocabulary_matches_collector_actions_under_cap(self):
        vocabulary = keyboard_action_vocabulary(0.005)
        self.assertEqual(vocabulary.shape, (17, 2))
        norms = np.linalg.norm(vocabulary, axis=1)
        np.testing.assert_allclose(
            np.unique(np.round(norms, decimals=7)),
            [0.0, 0.0025, 0.005],
            atol=1e-7,
        )
        self.assertTrue(
            np.any(np.all(np.isclose(vocabulary, [0.005, 0.0]), axis=1))
        )
        self.assertTrue(
            np.any(np.all(np.isclose(vocabulary, [0.0, 0.005]), axis=1))
        )
        diagonal = np.float32(0.005 / np.sqrt(2.0))
        self.assertTrue(
            np.any(
                np.all(
                    np.isclose(vocabulary, [diagonal, -diagonal], atol=1e-7),
                    axis=1,
                )
            )
        )

    def test_quantize_actions_returns_exact_keyboard_tokens(self):
        vocabulary = torch.from_numpy(keyboard_action_vocabulary(0.005))
        proposals = torch.tensor(
            [[[0.0047, 0.0010], [-0.0032, -0.0031], [0.0001, -0.0001]]]
        )
        quantized = quantize_actions(proposals, vocabulary)
        torch.testing.assert_close(quantized[0, 0], torch.tensor([0.005, 0.0]))
        diagonal = torch.tensor([-0.005, -0.005]) / np.sqrt(2.0)
        torch.testing.assert_close(quantized[0, 1], diagonal)
        torch.testing.assert_close(quantized[0, 2], torch.zeros(2))
        for action in quantized.reshape(-1, 2):
            self.assertTrue(torch.any(torch.all(vocabulary == action, dim=1)))

    def test_safety_events_take_precedence_over_autonomy(self):
        self.assertEqual(
            prioritize_ui_events(
                UIEvents(focus_lost=True, autonomous_toggle=True)
            ),
            ("pause", "window focus lost"),
        )
        self.assertEqual(
            prioritize_ui_events(
                UIEvents(pause_requested=True, goal_requested=True)
            ),
            ("pause", "operator pause"),
        )
        self.assertEqual(
            prioritize_ui_events(UIEvents(goal_requested=True, autonomous_toggle=True)),
            ("goal", None),
        )

    def test_validate_planner_result_rejects_nonfinite_or_unsafe_result(self):
        valid = {
            "action": np.array([0.001, 0.0], dtype=np.float32),
            "plan": np.zeros((3, 2), dtype=np.float32),
            "cost": 0.2,
            "goal_distance": 0.1,
            "solve_time_s": 0.05,
            "at_goal": False,
        }
        action, plan, *_ = validate_planner_result(
            valid, cap_m=ACTION_CAP_M, horizon=3
        )
        self.assertEqual(action.shape, (2,))
        self.assertEqual(plan.shape, (3, 2))

        for key in ("cost", "goal_distance", "solve_time_s"):
            invalid = dict(valid)
            invalid[key] = np.nan
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_planner_result(invalid, cap_m=ACTION_CAP_M, horizon=3)

        invalid_plan = dict(valid)
        invalid_plan["plan"] = np.array(
            [[0.02, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        with self.assertRaises(ValueError):
            validate_planner_result(
                invalid_plan, cap_m=ACTION_CAP_M, horizon=3
            )

    def test_validate_transition_prediction_checks_action_and_latent_shape(self):
        result = {
            "prediction_action": np.array([0.0025, 0.0], dtype=np.float32),
            "encoded_latent": np.arange(4, dtype=np.float32),
            "predicted_next_latent": np.arange(4, dtype=np.float32) + 1,
        }
        encoded, predicted = validate_transition_prediction(
            result, action=[0.0025, 0.0], latent_dim=4
        )
        np.testing.assert_array_equal(encoded, np.arange(4, dtype=np.float32))
        np.testing.assert_array_equal(
            predicted, np.arange(4, dtype=np.float32) + 1
        )
        with self.assertRaisesRegex(ValueError, "different action"):
            validate_transition_prediction(
                result, action=[-0.0025, 0.0], latent_dim=4
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_transition_prediction(
                result, action=[0.0025, 0.0], latent_dim=3
            )

    def test_external_goal_loads_image_and_validates_video_arguments(self):
        class UnexpectedTransform:
            def apply(self, frame):
                raise AssertionError("224x224 goal should not be transformed")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "goal.png")
            expected = np.full((224, 224, 3), 73, dtype=np.uint8)
            iio.imwrite(path, expected)
            goal, source = load_external_goal(
                goal_image=path,
                goal_video=None,
                goal_video_frame=None,
                transform_profile=UnexpectedTransform(),
            )
            np.testing.assert_array_equal(goal, expected)
            self.assertEqual(source["kind"], "image")
            self.assertEqual(source["path"], str(path.resolve()))

        with self.assertRaisesRegex(ValueError, "requires --goal-video"):
            load_external_goal(
                goal_image=None,
                goal_video=None,
                goal_video_frame=3,
                transform_profile=UnexpectedTransform(),
            )
        with self.assertRaisesRegex(ValueError, "requires --goal-video-frame"):
            load_external_goal(
                goal_image=None,
                goal_video=Path("episode.mp4"),
                goal_video_frame=None,
                transform_profile=UnexpectedTransform(),
            )

    def test_validate_measured_pose_checks_tracking_and_workspace(self):
        pose = np.array([0.14, 0.02, 0.03, 0.0, np.pi / 2, 0.0])
        measured = validate_measured_pose(
            pose,
            target_xy=np.array([0.14, 0.02]),
            fixed_z=0.03,
            orientation=np.array([0.0, np.pi / 2, 0.0]),
            xy_tolerance_m=0.005,
            z_tolerance_m=0.002,
            orientation_tolerance_rad=0.1,
            bounds=((0.10, 0.20), (-0.10, 0.10)),
            require_tracking=True,
        )
        np.testing.assert_allclose(measured, pose)

        with self.assertRaisesRegex(ValueError, "tracking error"):
            validate_measured_pose(
                pose,
                target_xy=np.array([0.16, 0.02]),
                fixed_z=0.03,
                orientation=np.array([0.0, np.pi / 2, 0.0]),
                xy_tolerance_m=0.005,
                z_tolerance_m=0.002,
                orientation_tolerance_rad=0.1,
                bounds=((0.10, 0.20), (-0.10, 0.10)),
                require_tracking=True,
            )

    def test_motion_completion_requires_target_and_low_speed(self):
        complete, error, speed = motion_completion_status(
            [0.141, 0.019, 0.03, 0.0, np.pi / 2, 0.0],
            [0.010, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_xy=np.array([0.14, 0.02]),
            xy_tolerance_m=0.005,
            settled_linear_speed_m_s=0.02,
        )
        self.assertTrue(complete)
        self.assertLess(error, 0.005)
        self.assertAlmostEqual(speed, 0.010)

        too_fast, *_ = motion_completion_status(
            [0.14, 0.02],
            [0.021, 0.0, 0.0],
            target_xy=np.array([0.14, 0.02]),
            xy_tolerance_m=0.005,
            settled_linear_speed_m_s=0.02,
        )
        off_target, *_ = motion_completion_status(
            [0.146, 0.02],
            [0.0, 0.0, 0.0],
            target_xy=np.array([0.14, 0.02]),
            xy_tolerance_m=0.005,
            settled_linear_speed_m_s=0.02,
        )
        self.assertFalse(too_fast)
        self.assertFalse(off_target)

    def test_default_settle_gate_allows_normal_control_velocity(self):
        self.assertAlmostEqual(DEFAULT_SETTLED_LINEAR_SPEED_M_S, 0.1)
        complete, _, speed = motion_completion_status(
            [0.14, 0.02],
            [0.0354, 0.0, 0.0],
            target_xy=np.array([0.14, 0.02]),
            xy_tolerance_m=0.005,
            settled_linear_speed_m_s=DEFAULT_SETTLED_LINEAR_SPEED_M_S,
        )
        self.assertTrue(complete)
        self.assertAlmostEqual(speed, 0.0354)

    def test_planner_snapshot_must_be_strictly_after_settle(self):
        self.assertTrue(snapshot_follows_settle(101, 100))
        self.assertFalse(snapshot_follows_settle(100, 100))
        self.assertFalse(snapshot_follows_settle(99, 100))
        self.assertTrue(snapshot_follows_settle(1, None))

    def test_workspace_bounds_must_be_finite(self):
        parser = argparse.ArgumentParser()
        args = argparse.Namespace(
            x_min=-np.inf,
            x_max=np.inf,
            y_min=-0.1,
            y_max=0.1,
            start_x=0.14,
            start_y=0.02,
            execute=True,
            allow_unbounded_xy=False,
        )
        with self.assertRaises(SystemExit):
            _workspace_bounds(args, parser)

    def test_run_recorder_flushes_metadata_frames_and_events(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RunRecorder(Path(directory), {"mode": "DRY RUN"})
            expected = np.array(
                [
                    [[255, 0, 0], [0, 255, 0]],
                    [[0, 0, 255], [255, 255, 255]],
                ],
                dtype=np.uint8,
            )
            frame = recorder.save_frame("goal", expected)
            latent = recorder.save_latent(
                "trial_001_step_001", [1.0, 2.0, 3.0], predicted=False
            )
            predicted = recorder.save_latent(
                "trial_001_step_001", [1.5, 2.5, 3.5], predicted=True
            )
            recorder.event("trial_started", frame=frame)
            run_path = recorder.path
            recorder.close()

            metadata = json.loads((run_path / "metadata.json").read_text())
            event = json.loads((run_path / "events.jsonl").read_text())
            self.assertEqual(metadata["mode"], "DRY RUN")
            self.assertEqual(event["type"], "trial_started")
            self.assertEqual(Path(frame).suffix, ".png")
            self.assertTrue((run_path / frame).is_file())
            np.testing.assert_array_equal(iio.imread(run_path / frame), expected)
            np.testing.assert_array_equal(
                np.load(run_path / latent), np.array([1.0, 2.0, 3.0], np.float32)
            )
            np.testing.assert_array_equal(
                np.load(run_path / predicted),
                np.array([1.5, 2.5, 3.5], np.float32),
            )

    def test_planner_round_trip_has_a_response_deadline(self):
        class NoResponseConnection:
            def send(self, request):
                self.request = request

            def poll(self, timeout):
                return False

            def close(self):
                pass

        planner = PlannerProcess(
            argparse.Namespace(planner_timeout=0.001), Path.cwd()
        )
        planner.connection = NoResponseConnection()
        try:
            with self.assertRaisesRegex(TimeoutError, "did not respond"):
                planner._round_trip({"op": "plan"})
        finally:
            planner.close()


if __name__ == "__main__":
    unittest.main()
