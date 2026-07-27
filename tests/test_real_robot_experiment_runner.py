import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import scripts.run_real_robot_experiment as runner
from scripts.run_real_robot_experiment import (
    build_evaluator_command,
    identify_new_run,
    load_registry,
    resolve_experiment,
    snapshot_run_directories,
    validate_extra_eval_args,
    write_provenance,
)


class RealRobotExperimentRunnerTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict:
        repo = root / "method_repo"
        (repo / "scripts").mkdir(parents=True)
        (repo / "scripts" / "eval_pushbox_real.sh").write_text("#!/bin/sh\n")
        episode = repo / "datasets_videos" / "session" / "ep_005.mp4"
        episode.parent.mkdir(parents=True)
        episode.write_bytes(b"fixture")
        return {
            "schema_version": 1,
            "defaults": {"trial_id": 1, "postprocess_args": []},
            "cases": {
                "case_a": {
                    "dataset_episode": "datasets_videos/session/ep_005.mp4",
                    "initial_step": 80,
                    "goal_step": 160,
                }
            },
            "methods": {
                "method_a": {
                    "adapter": "discrete",
                    "repo": "method_repo",
                    "world_model": "discrete_e2_1",
                    "solver": "mcts",
                    "extra_args": ["--horizon", "5"],
                }
            },
        }

    def test_registry_resolves_case_method_and_builds_exact_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self._fixture(root)
            experiment = resolve_experiment(
                registry,
                case_name="case_a",
                method_name="method_a",
                repo_root=root,
            )
            command = build_evaluator_command(
                experiment,
                mode="execute",
                extra_args=["--max-actions", "25"],
            )
            self.assertEqual(experiment.trial_id, 1)
            self.assertEqual(experiment.repo, (root / "method_repo").resolve())
            self.assertEqual(command[:3], ["bash", str(experiment.launcher), "--planner"])
            self.assertIn("mcts", command)
            self.assertIn("datasets_videos/session/ep_005.mp4", command)
            self.assertIn("--initial-step", command)
            self.assertIn("80", command)
            self.assertIn("--goal-step", command)
            self.assertIn("160", command)
            self.assertEqual(command[-2:], ["--max-actions", "25"])
            self.assertIn("--execute", command)

    def test_registry_rejects_noncausal_case(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self._fixture(root)
            registry["cases"]["case_a"]["goal_step"] = 80
            with self.assertRaisesRegex(ValueError, "initial_step < goal_step"):
                resolve_experiment(
                    registry,
                    case_name="case_a",
                    method_name="method_a",
                    repo_root=root,
                )

    def test_load_registry_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(
                json.dumps(
                    {"schema_version": 2, "cases": {"a": {}}, "methods": {"b": {}}}
                )
            )
            with self.assertRaisesRegex(ValueError, "schema_version 1"):
                load_registry(path)

    def test_extra_arguments_cannot_override_case_or_method_identity(self):
        self.assertEqual(
            validate_extra_eval_args(
                ["--max-actions", "25"], label="test arguments"
            ),
            ("--max-actions", "25"),
        )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            validate_extra_eval_args(
                ["--solver=mcts"], label="test arguments"
            )
        with self.assertRaisesRegex(ValueError, "cannot override"):
            validate_extra_eval_args(
                ["--dataset-episode", "other.mp4"], label="test arguments"
            )

    def test_new_run_detection_requires_one_complete_run(self):
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            before = snapshot_run_directories(output_root)
            run = output_root / "20260724_120000_123"
            run.mkdir()
            (run / "metadata.json").write_text("{}\n")
            (run / "events.jsonl").write_text("")
            self.assertEqual(identify_new_run(before, output_root), run.resolve())

            second = output_root / "20260724_120001_124"
            second.mkdir()
            (second / "metadata.json").write_text("{}\n")
            (second / "events.jsonl").write_text("")
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                identify_new_run(before, output_root)

    def test_provenance_write_is_json_and_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.json"
            write_provenance(path, {"case_name": "case_a", "schema_version": 1})
            self.assertEqual(json.loads(path.read_text())["case_name"], "case_a")
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_main_runs_evaluator_then_postprocessor_and_records_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = self._fixture(root)
            registry["methods"]["method_a"]["repo"] = str(root / "method_repo")
            config_path = root / "experiments.json"
            config_path.write_text(json.dumps(registry))
            run_path = root / "method_repo" / "real_robot_runs" / "new_run"
            calls = []

            def fake_run(command, *, cwd, check):
                calls.append((command, Path(cwd), check))
                if len(calls) == 1:
                    run_path.mkdir(parents=True)
                    (run_path / "metadata.json").write_text("{}\n")
                    (run_path / "events.jsonl").write_text("")
                else:
                    (
                        run_path / "analysis" / "overview_trial_001"
                    ).mkdir(parents=True)
                return mock.Mock(returncode=0)

            with mock.patch.object(runner.subprocess, "run", side_effect=fake_run):
                status = runner.main(
                    [
                        "--config",
                        str(config_path),
                        "--case",
                        "case_a",
                        "--method",
                        "method_a",
                        "--dry-run",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertEqual(len(calls), 2)
            self.assertIn("eval_pushbox_real.sh", " ".join(calls[0][0]))
            self.assertIn("postprocess_discrete_real_run.sh", " ".join(calls[1][0]))
            provenance = json.loads((run_path / "experiment.json").read_text())
            self.assertEqual(provenance["case_name"], "case_a")
            self.assertEqual(provenance["method_name"], "method_a")
            self.assertEqual(provenance["evaluator_exit_code"], 0)
            self.assertEqual(provenance["postprocess"]["exit_code"], 0)
            self.assertEqual(
                Path(provenance["postprocess"]["analysis_path"]),
                run_path / "analysis" / "overview_trial_001",
            )


if __name__ == "__main__":
    unittest.main()
