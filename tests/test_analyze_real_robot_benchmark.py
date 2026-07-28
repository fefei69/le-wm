import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_real_robot_benchmark import (
    ExperimentSource,
    load_run_metric,
    progress_fraction,
    select_repetitions,
)


class AnalyzeRealRobotBenchmarkTests(unittest.TestCase):
    def test_selects_latest_three_and_supersedes_earliest_extra(self):
        sources = []
        for index in range(4):
            run_path = Path(f"/tmp/run_{index}")
            sources.append(
                ExperimentSource(
                    experiment_path=run_path / "experiment.json",
                    run_path=run_path,
                    case_name="test04",
                    method_name="discrete_mcts_endpoint",
                    started_at=f"2026-07-27T17:0{index}:00",
                    experiment={},
                )
            )

        selected, superseded = select_repetitions(sources, 3)

        self.assertEqual([source.run_path.name for source in selected], ["run_1", "run_2", "run_3"])
        self.assertEqual([source.run_path.name for source in superseded], ["run_0"])

    def test_nonterminal_run_uses_marked_last_observation_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory) / "run_001"
            analysis = run_path / "analysis" / "overview_trial_001"
            analysis.mkdir(parents=True)
            metadata = {
                "argv": {
                    "world_model": "discrete_e2_1",
                    "solver": "mcts",
                    "horizon": 5,
                    "replan_every": 1,
                    "cost_mode": "latent_mse",
                    "max_actions": 50,
                }
            }
            (run_path / "metadata.json").write_text(json.dumps(metadata))
            summary = {
                "outcome": "abort",
                "step_count": 10,
                "tracking": {
                    "terminal_frame_available": False,
                    "box": {
                        "start_error_px": 20.0,
                        "final_error_px": 5.0,
                        "final_orientation_error_deg": 3.0,
                        "best_error_px": 2.0,
                        "detection_rate": 1.0,
                        "orientation_detection_rate": 1.0,
                    },
                    "ee": {
                        "start_error_px": 10.0,
                        "final_error_px": 8.0,
                        "best_error_px": 7.0,
                        "detection_rate": 0.9,
                    },
                },
            }
            (analysis / "summary.json").write_text(json.dumps(summary))
            (analysis / "planning_metrics.csv").write_text(
                "step,solve_time_s\n1,2.0\n2,4.0\n"
            )
            (run_path / "events.jsonl").write_text(
                json.dumps({"type": "fault", "reason": "test fault"}) + "\n"
            )
            experiment = {
                "postprocess": {"analysis_path": str(analysis)},
            }
            source = ExperimentSource(
                experiment_path=run_path / "experiment.json",
                run_path=run_path,
                case_name="test02",
                method_name="discrete_mcts_endpoint",
                started_at="2026-07-27T00:00:00",
                experiment=experiment,
            )

            record = load_run_metric(source, "selected")

            self.assertFalse(record.terminal_available)
            self.assertEqual(record.endpoint_source, "last_pre_action")
            self.assertEqual(record.box_endpoint_error_px, 5.0)
            self.assertEqual(record.box_progress_fraction, 0.75)
            self.assertEqual(record.action_fraction, 0.2)
            self.assertEqual(record.mean_solve_time_s, 3.0)
            self.assertTrue(record.faulted)

    def test_progress_preserves_regression_and_missing_values(self):
        self.assertEqual(progress_fraction(10.0, 15.0), -0.5)
        self.assertTrue(progress_fraction(0.0, 1.0) != progress_fraction(0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
