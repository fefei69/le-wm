#!/usr/bin/env python3
"""Aggregate named real-robot experiments across LeWM and discrete-WM runs.

The analysis deliberately separates three ideas that are easy to conflate:

* continuous tracking quality (box position/orientation and EE position);
* whether a settled post-action terminal frame exists; and
* a conservative threshold-based success diagnostic.

For legacy/faulted attempts without a terminal frame, the continuous endpoint
columns use the last valid pre-action observation and mark its source. Those
attempts never count as conservative successes. Planner objective values are
not compared across world models because their latent spaces and scales differ.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lewm-benchmark-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ("test01", "test02", "test04", "test05")
METHOD_ORDER = (
    "lewm_cem",
    "dinowm_cem",
    "dinowm_categorical_cem",
    "discrete_cem",
    "discrete_mcts",
    "discrete_mcts_endpoint",
    "discrete_mcts_r5",
)
BALANCED_METHODS = tuple(
    method for method in METHOD_ORDER if method != "discrete_mcts_endpoint"
)
ENDPOINT_ABLATION_METHODS = (
    "discrete_mcts",
    "discrete_mcts_endpoint",
    "discrete_mcts_r5",
)
METHOD_LABELS = {
    "lewm_cem": "LeWM\nCEM",
    "dinowm_cem": "DINO-WM\nCEM",
    "dinowm_categorical_cem": "DINO-WM\nCat-CEM",
    "discrete_cem": "Discrete WM\nCEM r1",
    "discrete_mcts": "Discrete WM\nMCTS r1 settle",
    "discrete_mcts_endpoint": "Discrete WM\nMCTS r1 endpoint",
    "discrete_mcts_r5": "Discrete WM\nMCTS r5 endpoint",
}
METHOD_COLORS = {
    method: plt.get_cmap("tab10")(index)
    for index, method in enumerate(METHOD_ORDER)
}
CASE_COLORS = {
    case: plt.get_cmap("Dark2")(index)
    for index, case in enumerate(DEFAULT_CASES)
}
METHOD_SETTING_KEYS = (
    "world_model",
    "solver",
    "action_mode",
    "horizon",
    "replan_every",
    "cost_mode",
    "cost_c",
    "cost_lambda",
    "num_samples",
    "n_steps",
    "iterations",
    "topk",
    "elite_count",
    "cem_batch_size",
    "uct_c",
    "mcts_recommendation",
    "mcts_final_topk",
    "mcts_final_repeats",
    "action_cap",
    "max_actions",
    "seed",
    "dataset",
    "artifact_manifest",
    "planner_config",
    "policy",
    "checkpoint",
    "cidm_checkpoint",
)


@dataclass(frozen=True)
class ExperimentSource:
    experiment_path: Path
    run_path: Path
    case_name: str
    method_name: str
    started_at: str
    experiment: dict[str, Any]


@dataclass(frozen=True)
class RunMetric:
    case_name: str
    method_name: str
    run_id: str
    run_path: str
    started_at: str
    selection_status: str
    outcome: str
    step_count: int
    max_actions: int
    action_fraction: float
    terminal_available: bool
    endpoint_source: str
    faulted: bool
    fault_reason: str
    box_start_error_px: float
    box_endpoint_error_px: float
    box_endpoint_orientation_error_deg: float
    box_best_error_px: float
    box_progress_fraction: float
    box_detection_rate: float
    box_orientation_detection_rate: float
    ee_start_error_px: float
    ee_endpoint_error_px: float
    ee_best_error_px: float
    ee_progress_fraction: float
    ee_detection_rate: float
    mean_solve_time_s: float
    median_solve_time_s: float
    setting_sha256: str


@dataclass(frozen=True)
class MetricSpec:
    field: str
    title: str
    ylabel: str
    scale: float = 1.0
    higher_is_better: bool = False


PLOT_METRICS = (
    MetricSpec("box_endpoint_error_px", "Box endpoint position", "error (px)"),
    MetricSpec(
        "box_endpoint_orientation_error_deg",
        "Box endpoint orientation",
        "error (deg, modulo 180)",
    ),
    MetricSpec("ee_endpoint_error_px", "EE endpoint position", "error (px)"),
    MetricSpec(
        "box_progress_fraction",
        "Box goal progress",
        "progress (%)",
        scale=100.0,
        higher_is_better=True,
    ),
    MetricSpec(
        "action_fraction",
        "Executed action budget",
        "budget executed (%)",
        scale=100.0,
        higher_is_better=True,
    ),
    MetricSpec("mean_solve_time_s", "Planning runtime", "mean solve time (s)"),
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lewm-runs",
        type=Path,
        default=REPO_ROOT / "real_robot_runs",
    )
    parser.add_argument(
        "--discrete-runs",
        type=Path,
        default=REPO_ROOT.parent / "discrete-la-wm" / "real_robot_runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "real_robot_benchmark",
    )
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(valid.mean()) if len(valid) else float("nan")


def finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(np.median(valid)) if len(valid) else float("nan")


def finite_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    valid = array[np.isfinite(array)]
    return float(valid.std(ddof=1)) if len(valid) > 1 else 0.0


def progress_fraction(start: float, endpoint: float) -> float:
    if not math.isfinite(start) or not math.isfinite(endpoint) or start <= 0.0:
        return float("nan")
    return (start - endpoint) / start


def canonical_setting(metadata: dict[str, Any]) -> dict[str, Any]:
    argv = metadata.get("argv", {})
    if not isinstance(argv, dict):
        raise ValueError("metadata argv must be an object")
    return {key: argv.get(key) for key in METHOD_SETTING_KEYS if key in argv}


def setting_hash(setting: dict[str, Any]) -> str:
    encoded = json.dumps(setting, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def discover_experiments(
    roots: Sequence[Path], cases: Sequence[str]
) -> list[ExperimentSource]:
    selected_cases = set(cases)
    experiments: list[ExperimentSource] = []
    for root in roots:
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"run root not found: {root}")
        for path in sorted(root.glob("*/experiment.json")):
            experiment = load_object(path)
            case_name = str(experiment.get("case_name", ""))
            method_name = str(experiment.get("method_name", ""))
            if case_name not in selected_cases or method_name not in METHOD_ORDER:
                continue
            experiments.append(
                ExperimentSource(
                    experiment_path=path.resolve(),
                    run_path=path.parent.resolve(),
                    case_name=case_name,
                    method_name=method_name,
                    started_at=str(experiment.get("started_at", path.parent.name)),
                    experiment=experiment,
                )
            )
    return experiments


def select_repetitions(
    experiments: Sequence[ExperimentSource], repetitions: int
) -> tuple[list[ExperimentSource], list[ExperimentSource]]:
    if repetitions < 1:
        raise ValueError("--repetitions must be positive")
    grouped: dict[tuple[str, str], list[ExperimentSource]] = {}
    for experiment in experiments:
        grouped.setdefault(
            (experiment.case_name, experiment.method_name), []
        ).append(experiment)
    selected: list[ExperimentSource] = []
    superseded: list[ExperimentSource] = []
    for sources in grouped.values():
        ordered = sorted(sources, key=lambda item: (item.started_at, item.run_path.name))
        selected.extend(ordered[-repetitions:])
        superseded.extend(ordered[:-repetitions])
    return selected, superseded


def analysis_path(source: ExperimentSource) -> Path:
    postprocess = source.experiment.get("postprocess", {})
    if isinstance(postprocess, dict):
        raw = postprocess.get("analysis_path")
        if isinstance(raw, str) and raw:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = source.run_path / candidate
            if candidate.is_dir():
                return candidate.resolve()
    return source.run_path / "analysis" / "overview_trial_001"


def read_fault(run_path: Path) -> tuple[bool, str]:
    events_path = run_path / "events.jsonl"
    if not events_path.is_file():
        return False, ""
    faulted = False
    reason = ""
    for line_number, line in enumerate(events_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON at {events_path}:{line_number}"
            ) from exc
        if event.get("type") == "fault":
            faulted = True
            reason = str(event.get("reason", reason))
        if event.get("type") == "run_finished" and event.get("faulted"):
            faulted = True
            reason = str(event.get("fault_reason", reason))
    return faulted, reason


def read_solve_times(path: Path) -> tuple[float, float]:
    if not path.is_file():
        return float("nan"), float("nan")
    values: list[float] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = finite_float(row.get("solve_time_s"))
            if math.isfinite(value):
                values.append(value)
    return finite_mean(values), finite_median(values)


def load_run_metric(source: ExperimentSource, selection_status: str) -> RunMetric:
    metadata_path = source.run_path / "metadata.json"
    summary_path = analysis_path(source) / "summary.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    metadata = load_object(metadata_path)
    summary = load_object(summary_path)
    tracking = summary.get("tracking", {})
    box = tracking.get("box", {}) if isinstance(tracking, dict) else {}
    ee = tracking.get("ee", {}) if isinstance(tracking, dict) else {}
    if not isinstance(box, dict) or not isinstance(ee, dict):
        raise ValueError(f"invalid tracking summary: {summary_path}")
    terminal_available = bool(tracking.get("terminal_frame_available", False))
    box_start = finite_float(box.get("start_error_px"))
    box_endpoint = finite_float(box.get("final_error_px"))
    ee_start = finite_float(ee.get("start_error_px"))
    ee_endpoint = finite_float(ee.get("final_error_px"))
    argv = metadata.get("argv", {})
    max_actions = int(argv.get("max_actions", 50))
    step_count = int(summary.get("step_count", 0))
    mean_solve, median_solve = read_solve_times(
        analysis_path(source) / "planning_metrics.csv"
    )
    faulted, fault_reason = read_fault(source.run_path)
    setting = canonical_setting(metadata)
    return RunMetric(
        case_name=source.case_name,
        method_name=source.method_name,
        run_id=source.run_path.name,
        run_path=str(source.run_path),
        started_at=source.started_at,
        selection_status=selection_status,
        outcome=str(summary.get("outcome", "unknown")),
        step_count=step_count,
        max_actions=max_actions,
        action_fraction=(step_count / max_actions if max_actions > 0 else float("nan")),
        terminal_available=terminal_available,
        endpoint_source=("terminal" if terminal_available else "last_pre_action"),
        faulted=faulted,
        fault_reason=fault_reason,
        box_start_error_px=box_start,
        box_endpoint_error_px=box_endpoint,
        box_endpoint_orientation_error_deg=finite_float(
            box.get("final_orientation_error_deg")
        ),
        box_best_error_px=finite_float(box.get("best_error_px")),
        box_progress_fraction=progress_fraction(box_start, box_endpoint),
        box_detection_rate=finite_float(box.get("detection_rate")),
        box_orientation_detection_rate=finite_float(
            box.get("orientation_detection_rate")
        ),
        ee_start_error_px=ee_start,
        ee_endpoint_error_px=ee_endpoint,
        ee_best_error_px=finite_float(ee.get("best_error_px")),
        ee_progress_fraction=progress_fraction(ee_start, ee_endpoint),
        ee_detection_rate=finite_float(ee.get("detection_rate")),
        mean_solve_time_s=mean_solve,
        median_solve_time_s=median_solve,
        setting_sha256=setting_hash(setting),
    )


def verify_settings(
    sources: Sequence[ExperimentSource], records: Sequence[RunMetric]
) -> dict[str, Any]:
    source_by_run = {source.run_path.name: source for source in sources}
    report: dict[str, Any] = {"methods": {}}
    for method in METHOD_ORDER:
        method_records = [record for record in records if record.method_name == method]
        if not method_records:
            continue
        signatures: dict[str, dict[str, Any]] = {}
        runs: list[dict[str, Any]] = []
        for record in method_records:
            source = source_by_run[record.run_id]
            metadata = load_object(source.run_path / "metadata.json")
            setting = canonical_setting(metadata)
            signature = setting_hash(setting)
            signatures[signature] = setting
            runs.append(
                {
                    "case": record.case_name,
                    "run_id": record.run_id,
                    "setting_sha256": signature,
                }
            )
        if len(signatures) != 1:
            raise ValueError(
                f"selected repetitions for {method} have inconsistent settings: "
                f"{sorted(signatures)}"
            )
        signature, setting = next(iter(signatures.items()))
        report["methods"][method] = {
            "consistent": True,
            "setting_sha256": signature,
            "setting": setting,
            "runs": runs,
        }
    return report


def records_for(
    records: Sequence[RunMetric], case: str, method: str
) -> list[RunMetric]:
    return [
        record
        for record in records
        if record.case_name == case and record.method_name == method
    ]


def aggregate_group(records: Sequence[RunMetric]) -> dict[str, Any]:
    metrics = (
        "box_endpoint_error_px",
        "box_endpoint_orientation_error_deg",
        "ee_endpoint_error_px",
        "box_progress_fraction",
        "ee_progress_fraction",
        "action_fraction",
        "mean_solve_time_s",
        "box_detection_rate",
        "box_orientation_detection_rate",
        "ee_detection_rate",
    )
    result: dict[str, Any] = {
        "n": len(records),
        "n_terminal": sum(record.terminal_available for record in records),
        "terminal_rate": finite_mean(
            float(record.terminal_available) for record in records
        ),
        "n_abort": sum(record.outcome == "abort" for record in records),
        "n_faulted": sum(record.faulted for record in records),
        "fault_rate": finite_mean(float(record.faulted) for record in records),
    }
    for metric in metrics:
        values = [float(getattr(record, metric)) for record in records]
        result[f"mean_{metric}"] = finite_mean(values)
        result[f"std_{metric}"] = finite_std(values)
        result[f"median_{metric}"] = finite_median(values)
    return result


def make_case_method_summary(
    records: Sequence[RunMetric], cases: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for method in METHOD_ORDER:
            group = records_for(records, case, method)
            if not group:
                continue
            rows.append(
                {"case_name": case, "method_name": method, **aggregate_group(group)}
            )
    return rows


def case_summary_lookup(
    rows: Sequence[dict[str, Any]], case: str, method: str
) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in rows
            if row["case_name"] == case and row["method_name"] == method
        ),
        None,
    )


def common_cases(
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
    repetitions: int,
) -> list[str]:
    return [
        case
        for case in cases
        if all(
            (row := case_summary_lookup(case_rows, case, method)) is not None
            and int(row["n"]) == repetitions
            for method in methods
        )
    ]


def make_method_summary(
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = (
        "box_endpoint_error_px",
        "box_endpoint_orientation_error_deg",
        "ee_endpoint_error_px",
        "box_progress_fraction",
        "ee_progress_fraction",
        "action_fraction",
        "mean_solve_time_s",
        "terminal_rate",
        "fault_rate",
    )
    rows: list[dict[str, Any]] = []
    for method in methods:
        cells = [
            row
            for case in cases
            if (row := case_summary_lookup(case_rows, case, method)) is not None
        ]
        if not cells:
            continue
        result: dict[str, Any] = {
            "method_name": method,
            "case_count": len(cells),
            "run_count": sum(int(cell["n"]) for cell in cells),
        }
        for metric in metrics:
            cell_key = metric if metric in {"terminal_rate", "fault_rate"} else f"mean_{metric}"
            values = [finite_float(cell.get(cell_key)) for cell in cells]
            result[f"mean_case_{metric}"] = finite_mean(values)
            result[f"std_case_{metric}"] = finite_std(values)
            result[f"median_case_{metric}"] = finite_median(values)
        rows.append(result)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def method_tick_labels(methods: Sequence[str]) -> list[str]:
    return [METHOD_LABELS.get(method, method) for method in methods]


def plot_run_metric(
    ax: plt.Axes,
    records: Sequence[RunMetric],
    case: str,
    methods: Sequence[str],
    spec: MetricSpec,
) -> None:
    for x, method in enumerate(methods):
        group = records_for(records, case, method)
        values = np.asarray(
            [float(getattr(record, spec.field)) * spec.scale for record in group],
            dtype=np.float64,
        )
        offsets = np.linspace(-0.16, 0.16, len(group)) if group else np.array([])
        for offset, value, record in zip(offsets, values, group):
            if not math.isfinite(float(value)):
                continue
            marker = "o" if record.terminal_available else "X"
            ax.scatter(
                x + offset,
                value,
                marker=marker,
                s=55,
                color=METHOD_COLORS[method],
                edgecolor="black",
                linewidth=0.6,
                zorder=3,
            )
        valid = values[np.isfinite(values)]
        if len(valid):
            mean = float(valid.mean())
            std = float(valid.std(ddof=1)) if len(valid) > 1 else 0.0
            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt="D",
                color="black",
                markerfacecolor="white",
                markersize=5,
                capsize=4,
                linewidth=1.2,
                zorder=4,
            )
        else:
            ax.text(
                x,
                0.03,
                "not run",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                rotation=90,
                color="0.5",
                fontsize=8,
            )
    if spec.field.endswith("progress_fraction"):
        ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xticks(range(len(methods)), method_tick_labels(methods), rotation=25, ha="right")
    style_axis(ax, spec.title, spec.ylabel)


def plot_case_figure(
    output: Path,
    records: Sequence[RunMetric],
    case: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, spec in zip(axes.flat, PLOT_METRICS):
        plot_run_metric(ax, records, case, METHOD_ORDER, spec)
    handles = [
        Line2D([], [], marker="o", linestyle="", color="0.2", label="settled terminal"),
        Line2D([], [], marker="X", linestyle="", color="0.2", label="last pre-action (no terminal)"),
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="",
            label="mean ± 1 SD",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        f"Real-robot benchmark: {case} (three selected repetitions per method)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_case_mean_metric(
    ax: plt.Axes,
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
    spec: MetricSpec,
) -> None:
    key = f"mean_{spec.field}"
    for case in cases:
        values = []
        for method in methods:
            row = case_summary_lookup(case_rows, case, method)
            values.append(
                finite_float(row.get(key)) * spec.scale if row is not None else np.nan
            )
        ax.plot(
            range(len(methods)),
            values,
            "o-",
            color=CASE_COLORS.get(case, "0.5"),
            alpha=0.72,
            linewidth=1.2,
            markersize=5,
            label=case,
        )
    for x, method in enumerate(methods):
        values = []
        for case in cases:
            row = case_summary_lookup(case_rows, case, method)
            if row is not None:
                values.append(finite_float(row.get(key)) * spec.scale)
        mean = finite_mean(values)
        std = finite_std(values)
        if math.isfinite(mean):
            ax.errorbar(
                x,
                mean,
                yerr=std,
                fmt="D",
                color="black",
                markerfacecolor="white",
                markersize=6,
                capsize=4,
                linewidth=1.4,
                zorder=5,
            )
    if spec.field.endswith("progress_fraction"):
        ax.axhline(0.0, color="0.35", linewidth=0.8)
    ax.set_xticks(range(len(methods)), method_tick_labels(methods), rotation=25, ha="right")
    style_axis(ax, spec.title, spec.ylabel)


def plot_cross_case(
    output: Path,
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
    title: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    for ax, spec in zip(axes.flat, PLOT_METRICS):
        plot_case_mean_metric(ax, case_rows, cases, methods, spec)
    handles = [
        Line2D([], [], marker="o", color=CASE_COLORS.get(case, "0.5"), label=case)
        for case in cases
    ]
    handles.append(
        Line2D(
            [],
            [],
            marker="D",
            markerfacecolor="white",
            markeredgecolor="black",
            linestyle="",
            label="mean ± SD across case means",
        )
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(
    output: Path,
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
    dpi: int,
) -> None:
    specs = (
        ("mean_box_endpoint_error_px", "Box position error (px)", "viridis_r", 1.0),
        (
            "mean_box_endpoint_orientation_error_deg",
            "Box orientation error (deg)",
            "viridis_r",
            1.0,
        ),
        ("mean_ee_endpoint_error_px", "EE position error (px)", "viridis_r", 1.0),
        ("terminal_rate", "Settled terminal records (%)", "RdYlGn", 100.0),
    )
    fig, axes = plt.subplots(2, 2, figsize=(17, 9), constrained_layout=True)
    for ax, (key, title, cmap, scale) in zip(axes.flat, specs):
        matrix = np.full((len(cases), len(methods)), np.nan, dtype=np.float64)
        for row_index, case in enumerate(cases):
            for column_index, method in enumerate(methods):
                row = case_summary_lookup(case_rows, case, method)
                if row is not None:
                    matrix[row_index, column_index] = finite_float(row.get(key)) * scale
        image = ax.imshow(matrix, aspect="auto", cmap=cmap)
        for row_index in range(len(cases)):
            for column_index in range(len(methods)):
                value = matrix[row_index, column_index]
                if math.isfinite(float(value)):
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        bbox={"facecolor": "white", "alpha": 0.55, "edgecolor": "none"},
                    )
        ax.set_xticks(range(len(methods)), method_tick_labels(methods), rotation=25, ha="right")
        ax.set_yticks(range(len(cases)), cases)
        ax.set_title(title, fontweight="bold")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle(
        "Balanced case × method tracking summary (three repetitions per cell)",
        fontsize=16,
        fontweight="bold",
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def conservative_success(record: RunMetric, threshold_px: float) -> bool:
    return bool(
        record.terminal_available
        and not record.faulted
        and record.outcome != "abort"
        and math.isfinite(record.box_endpoint_error_px)
        and record.box_endpoint_error_px <= threshold_px
    )


def plot_success_sweep(
    output: Path,
    records: Sequence[RunMetric],
    methods: Sequence[str],
    dpi: int,
) -> list[dict[str, Any]]:
    thresholds = np.linspace(0.0, 50.0, 201)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    table: list[dict[str, Any]] = []
    for method in methods:
        group = [record for record in records if record.method_name == method]
        rates = [
            finite_mean(float(conservative_success(record, threshold)) for record in group)
            for threshold in thresholds
        ]
        axes[0].plot(
            thresholds,
            np.asarray(rates) * 100.0,
            color=METHOD_COLORS[method],
            linewidth=2,
            label=METHOD_LABELS[method].replace("\n", " "),
        )
        for threshold in (2.0, 5.0, 10.0, 20.0, 30.0, 50.0):
            table.append(
                {
                    "method_name": method,
                    "threshold_px": threshold,
                    "conservative_success_rate": finite_mean(
                        float(conservative_success(record, threshold))
                        for record in group
                    ),
                    "n": len(group),
                }
            )
        terminal_errors = [
            record.box_endpoint_error_px
            for record in group
            if record.terminal_available and math.isfinite(record.box_endpoint_error_px)
        ]
        axes[1].boxplot(
            terminal_errors,
            positions=[methods.index(method)],
            widths=0.55,
            patch_artist=True,
            boxprops={"facecolor": METHOD_COLORS[method], "alpha": 0.55},
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "0.25"},
            capprops={"color": "0.25"},
            flierprops={
                "marker": "o",
                "markerfacecolor": METHOD_COLORS[method],
                "markeredgecolor": "black",
                "markersize": 4,
            },
        )
    axes[0].set_xlabel("box endpoint threshold (px)")
    style_axis(axes[0], "Conservative threshold-success sweep", "success rate (%)")
    axes[0].set_ylim(-2, 102)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xticks(range(len(methods)), method_tick_labels(methods), rotation=25, ha="right")
    style_axis(
        axes[1],
        "Terminal-only box errors",
        "error (px); non-terminal attempts omitted",
    )
    fig.suptitle(
        "Success is diagnostic, not an operator-recorded task label",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return table


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def render(value: Any) -> str:
        if isinstance(value, float):
            return "—" if not math.isfinite(value) else f"{value:.3f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def method_wins(
    case_rows: Sequence[dict[str, Any]],
    cases: Sequence[str],
    methods: Sequence[str],
    key: str,
    higher_is_better: bool = False,
) -> dict[str, int]:
    wins = {method: 0 for method in methods}
    for case in cases:
        values = []
        for method in methods:
            row = case_summary_lookup(case_rows, case, method)
            if row is not None:
                value = finite_float(row.get(key))
                if math.isfinite(value):
                    values.append((method, value))
        if not values:
            continue
        best_value = (max if higher_is_better else min)(value for _, value in values)
        for method, value in values:
            if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-9):
                wins[method] += 1
    return wins


def write_conclusions(
    path: Path,
    cases: Sequence[str],
    selected: Sequence[RunMetric],
    superseded: Sequence[RunMetric],
    case_rows: Sequence[dict[str, Any]],
    balanced_rows: Sequence[dict[str, Any]],
    endpoint_rows: Sequence[dict[str, Any]],
    endpoint_cases: Sequence[str],
) -> None:
    balanced_by_method = {row["method_name"]: row for row in balanced_rows}
    endpoint_by_method = {row["method_name"]: row for row in endpoint_rows}

    def best(metric: str, *, higher: bool = False) -> dict[str, Any]:
        valid = [
            row
            for row in balanced_rows
            if math.isfinite(finite_float(row.get(metric)))
        ]
        return (max if higher else min)(valid, key=lambda row: float(row[metric]))

    best_box = best("mean_case_box_endpoint_error_px")
    best_orientation = best("mean_case_box_endpoint_orientation_error_deg")
    best_ee = best("mean_case_ee_endpoint_error_px")
    best_progress = best("mean_case_box_progress_fraction", higher=True)
    box_wins = method_wins(
        case_rows,
        cases,
        BALANCED_METHODS,
        "mean_box_endpoint_error_px",
    )
    orientation_wins = method_wins(
        case_rows,
        cases,
        BALANCED_METHODS,
        "mean_box_endpoint_orientation_error_deg",
    )
    ee_wins = method_wins(
        case_rows,
        cases,
        BALANCED_METHODS,
        "mean_ee_endpoint_error_px",
    )
    nonterminal = [record for record in selected if not record.terminal_available]
    faulted = [record for record in selected if record.faulted]
    boundary_faults = sum(
        "outside commissioned bounds" in record.fault_reason for record in faulted
    )
    settle_faults = sum("did not settle" in record.fault_reason for record in faulted)
    categorical = balanced_by_method["dinowm_categorical_cem"]
    gaussian_dino = balanced_by_method["dinowm_cem"]
    categorical_box_improvement = 100.0 * (
        gaussian_dino["mean_case_box_endpoint_error_px"]
        - categorical["mean_case_box_endpoint_error_px"]
    ) / gaussian_dino["mean_case_box_endpoint_error_px"]
    threshold_10_rates = {
        method: finite_mean(
            float(conservative_success(record, 10.0))
            for record in selected
            if record.method_name == method
        )
        for method in BALANCED_METHODS
    }
    settle_r1 = endpoint_by_method["discrete_mcts"]
    endpoint_r1 = endpoint_by_method["discrete_mcts_endpoint"]
    endpoint_r5 = endpoint_by_method["discrete_mcts_r5"]
    endpoint_r1_box_delta = 100.0 * (
        endpoint_r1["mean_case_box_endpoint_error_px"]
        - settle_r1["mean_case_box_endpoint_error_px"]
    ) / settle_r1["mean_case_box_endpoint_error_px"]
    endpoint_r5_box_delta = 100.0 * (
        endpoint_r5["mean_case_box_endpoint_error_px"]
        - settle_r1["mean_case_box_endpoint_error_px"]
    ) / settle_r1["mean_case_box_endpoint_error_px"]
    coverage_rows = []
    for case in cases:
        for method in METHOD_ORDER:
            group = records_for(selected, case, method)
            coverage_rows.append(
                (
                    case,
                    METHOD_LABELS[method].replace("\n", " "),
                    len(group),
                    sum(record.terminal_available for record in group),
                    sum(record.faulted for record in group),
                )
            )
    balanced_table = []
    for method in sorted(
        BALANCED_METHODS,
        key=lambda name: balanced_by_method[name]["mean_case_box_endpoint_error_px"],
    ):
        row = balanced_by_method[method]
        balanced_table.append(
            (
                METHOD_LABELS[method].replace("\n", " "),
                row["mean_case_box_endpoint_error_px"],
                row["mean_case_box_endpoint_orientation_error_deg"],
                row["mean_case_ee_endpoint_error_px"],
                row["mean_case_box_progress_fraction"] * 100.0,
                row["mean_case_terminal_rate"] * 100.0,
                row["mean_case_mean_solve_time_s"],
            )
        )
    per_case_winners = []
    for case in cases:
        cells = [
            row
            for method in BALANCED_METHODS
            if (row := case_summary_lookup(case_rows, case, method)) is not None
        ]
        per_case_winners.append(
            (
                case,
                METHOD_LABELS[
                    min(cells, key=lambda row: row["mean_box_endpoint_error_px"])[
                        "method_name"
                    ]
                ].replace("\n", " "),
                METHOD_LABELS[
                    min(
                        cells,
                        key=lambda row: row[
                            "mean_box_endpoint_orientation_error_deg"
                        ],
                    )["method_name"]
                ].replace("\n", " "),
                METHOD_LABELS[
                    min(cells, key=lambda row: row["mean_ee_endpoint_error_px"])[
                        "method_name"
                    ]
                ].replace("\n", " "),
            )
        )
    endpoint_table = [
        (
            METHOD_LABELS[row["method_name"]].replace("\n", " "),
            row["mean_case_box_endpoint_error_px"],
            row["mean_case_box_endpoint_orientation_error_deg"],
            row["mean_case_ee_endpoint_error_px"],
            row["mean_case_box_progress_fraction"] * 100.0,
            row["mean_case_terminal_rate"] * 100.0,
        )
        for row in endpoint_rows
    ]
    lines = [
        "# Real-robot benchmark analysis",
        "",
        "## Scope and inclusion",
        "",
        f"Cases: `{', '.join(cases)}`. `test03` is intentionally omitted. ",
        "`discrete_cem_r5` was not run. The balanced comparison contains the six ",
        "methods with three selected repetitions in all four cases. The endpoint ",
        f"MCTS ablation uses `{', '.join(endpoint_cases)}` because the r1 endpoint ",
        "method was not run on `test01`.",
        "",
        "The latest three timestamped attempts are selected per case/method. This ",
        "keeps the three complete replacements for `test04 × discrete_mcts_endpoint`; ",
        f"{len(superseded)} earlier extra attempt(s) remain in the failure audit.",
        "",
        f"Of {len(selected)} selected attempts, {len(nonterminal)} lack a settled ",
        f"terminal frame and {len(faulted)} recorded a robot/runtime fault. Their ",
        "continuous endpoint values use the explicitly marked last pre-action ",
        "observation. They are retained to avoid survivorship bias, but never count ",
        "as conservative threshold successes.",
        "",
        "## Balanced descriptive results",
        "",
        markdown_table(
            (
                "method",
                "box px ↓",
                "orientation deg ↓",
                "EE px ↓",
                "box progress % ↑",
                "terminal % ↑",
                "solve s ↓",
            ),
            balanced_table,
        ),
        "",
        f"Lowest across-case mean box endpoint error: **{METHOD_LABELS[best_box['method_name']].replace(chr(10), ' ')}** ",
        f"({best_box['mean_case_box_endpoint_error_px']:.2f} px). Lowest orientation ",
        f"error: **{METHOD_LABELS[best_orientation['method_name']].replace(chr(10), ' ')}** ",
        f"({best_orientation['mean_case_box_endpoint_orientation_error_deg']:.2f}°). ",
        f"Lowest EE error: **{METHOD_LABELS[best_ee['method_name']].replace(chr(10), ' ')}** ",
        f"({best_ee['mean_case_ee_endpoint_error_px']:.2f} px). Highest mean normalized ",
        f"box progress: **{METHOD_LABELS[best_progress['method_name']].replace(chr(10), ' ')}** ",
        f"({best_progress['mean_case_box_progress_fraction'] * 100.0:.1f}%).",
        "",
        f"DINO-WM categorical CEM reduced mean box error by {categorical_box_improvement:.1f}% ",
        "relative to matched-budget DINO-WM Gaussian CEM, with effectively the ",
        "same solve time. It did not win an individual case on mean box error; its ",
        "overall advantage came from consistently avoiding the largest case-level ",
        "errors. At a conservative 10 px terminal threshold, the observed rates were ",
        "`"
        + ", ".join(
            f"{method}={threshold_10_rates[method] * 100.0:.1f}%"
            for method in BALANCED_METHODS
        )
        + "`.",
        "",
        "Per-case metric winners:",
        "",
        markdown_table(
            ("case", "box position", "box orientation", "EE position"),
            per_case_winners,
        ),
        "",
        "Case-win counts are descriptive because each cell has only three runs: ",
        f"box `{json.dumps(box_wins, sort_keys=True)}`, orientation ",
        f"`{json.dumps(orientation_wins, sort_keys=True)}`, EE ",
        f"`{json.dumps(ee_wins, sort_keys=True)}`.",
        "",
        "## Discrete-MCTS objective/cadence ablation",
        "",
        markdown_table(
            (
                "method",
                "box px ↓",
                "orientation deg ↓",
                "EE px ↓",
                "box progress % ↑",
                "terminal % ↑",
            ),
            endpoint_table,
        ),
        "",
        "This comparison separates r1 settling cost, r1 endpoint cost, and r5 ",
        "endpoint cost over their shared cases. Read the case-connected points in ",
        "`endpoint_mcts_ablation.png`; the three-case average alone can hide ",
        "case-specific reversals.",
        "",
        f"On those shared cases, r1 endpoint cost had {endpoint_r1_box_delta:.1f}% ",
        f"higher mean box error than r1 settling cost, while r5 endpoint execution ",
        f"had {endpoint_r5_box_delta:.1f}% higher error. The r5 method was much faster ",
        f"per planning call ({endpoint_r5['mean_case_mean_solve_time_s']:.2f} s versus ",
        f"{settle_r1['mean_case_mean_solve_time_s']:.2f} s), but that did not translate ",
        "to better overall box placement. It did help on `test04`, so the effect is ",
        "case-dependent rather than a universal failure of r5 execution.",
        "",
        "## Coverage audit",
        "",
        markdown_table(
            ("case", "method", "selected", "terminal", "faulted"),
            coverage_rows,
        ),
        "",
        f"All {len(faulted)} selected faults occurred in discrete-WM runs: ",
        f"{boundary_faults} were commissioned-workspace boundary violations and ",
        f"{settle_faults} was a settling timeout. The LeWM and both DINO-WM methods ",
        "produced settled terminal observations in every selected repetition. This ",
        "reliability difference is part of real-world performance and should not be ",
        "removed by a terminal-only ranking.",
        "",
        "## Interpretation limits",
        "",
        "- These are continuous visual tracking metrics, not human-recorded binary ",
        "  task-success labels. `success_threshold_sweep.png` therefore shows a ",
        "  transparent threshold diagnostic instead of declaring one arbitrary ",
        "  success cutoff.",
        "- Non-terminal last-observation values are not equivalent to settled ",
        "  terminal measurements. Use `terminal_only_method_summary.csv` as a ",
        "  sensitivity check, recognizing that it preferentially drops faulted runs.",
        "- Planner costs are intentionally not compared across world models; their ",
        "  latent representations and objective scales differ.",
        "- Three repetitions per case/method are enough for descriptive benchmarking, ",
        "  not strong statistical significance claims.",
        "- EE tracking is an independent visual QA proxy. Lower EE-to-goal error does ",
        "  not necessarily imply better box placement.",
        "",
        "## Generated artifacts",
        "",
        "- `per_case/`: one six-panel figure per case",
        "- `cross_case_balanced.png`: paired case-level balanced comparison",
        "- `cross_case_heatmaps.png`: case × method tracking and terminal coverage",
        "- `endpoint_mcts_ablation.png`: r1/r5 MCTS cost-cadence comparison",
        "- `success_threshold_sweep.png`: conservative position-threshold diagnostic",
        "- CSV files: run-level, cell-level, method-level, terminal-only, settings, ",
        "  threshold, and failure-audit tables",
        "",
    ]
    path.write_text("\n".join(lines))


def terminal_only_summary(
    records: Sequence[RunMetric], cases: Sequence[str], methods: Sequence[str]
) -> list[dict[str, Any]]:
    terminal = [record for record in records if record.terminal_available]
    rows = make_case_method_summary(terminal, cases)
    return make_method_summary(rows, cases, methods)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cases = tuple(args.cases)
    unknown_cases = sorted(set(cases) - set(DEFAULT_CASES))
    if unknown_cases:
        raise ValueError(f"unsupported cases for this benchmark: {unknown_cases}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    per_case_dir = output / "per_case"
    per_case_dir.mkdir(exist_ok=True)

    experiments = discover_experiments(
        (args.lewm_runs, args.discrete_runs), cases
    )
    selected_sources, superseded_sources = select_repetitions(
        experiments, args.repetitions
    )
    selected = [
        load_run_metric(source, "selected") for source in selected_sources
    ]
    superseded = [
        load_run_metric(source, "superseded_extra_attempt")
        for source in superseded_sources
    ]
    selected.sort(key=lambda row: (cases.index(row.case_name), METHOD_ORDER.index(row.method_name), row.started_at))
    superseded.sort(key=lambda row: (row.case_name, row.method_name, row.started_at))

    settings = verify_settings(selected_sources, selected)
    (output / "settings_audit.json").write_text(
        json.dumps(settings, indent=2, sort_keys=True) + "\n"
    )
    write_csv(output / "run_metrics.csv", [asdict(record) for record in selected])
    case_rows = make_case_method_summary(selected, cases)
    write_csv(output / "case_method_summary.csv", case_rows)

    balanced_cases = common_cases(
        case_rows, cases, BALANCED_METHODS, args.repetitions
    )
    if tuple(balanced_cases) != cases:
        raise ValueError(
            f"balanced comparison is incomplete: expected {cases}, got {balanced_cases}"
        )
    balanced_rows = make_method_summary(
        case_rows, balanced_cases, BALANCED_METHODS
    )
    write_csv(output / "balanced_method_summary.csv", balanced_rows)
    write_csv(
        output / "terminal_only_method_summary.csv",
        terminal_only_summary(selected, balanced_cases, BALANCED_METHODS),
    )

    endpoint_cases = common_cases(
        case_rows, cases, ENDPOINT_ABLATION_METHODS, args.repetitions
    )
    endpoint_rows = make_method_summary(
        case_rows, endpoint_cases, ENDPOINT_ABLATION_METHODS
    )
    write_csv(output / "endpoint_mcts_summary.csv", endpoint_rows)

    audit = [
        asdict(record)
        for record in [*selected, *superseded]
        if record.selection_status != "selected"
        or not record.terminal_available
        or record.faulted
        or record.outcome == "abort"
    ]
    write_csv(output / "failure_audit.csv", audit)

    for case in cases:
        plot_case_figure(
            per_case_dir / f"{case}_method_comparison.png",
            selected,
            case,
            args.dpi,
        )
    plot_cross_case(
        output / "cross_case_balanced.png",
        case_rows,
        balanced_cases,
        BALANCED_METHODS,
        "Balanced cross-case comparison (case means connected)",
        args.dpi,
    )
    plot_heatmaps(
        output / "cross_case_heatmaps.png",
        case_rows,
        balanced_cases,
        BALANCED_METHODS,
        args.dpi,
    )
    plot_cross_case(
        output / "endpoint_mcts_ablation.png",
        case_rows,
        endpoint_cases,
        ENDPOINT_ABLATION_METHODS,
        "Discrete MCTS ablation on shared cases: cost objective and execution cadence",
        args.dpi,
    )
    threshold_rows = plot_success_sweep(
        output / "success_threshold_sweep.png",
        selected,
        BALANCED_METHODS,
        args.dpi,
    )
    write_csv(output / "success_thresholds.csv", threshold_rows)
    write_conclusions(
        output / "conclusions.md",
        balanced_cases,
        selected,
        superseded,
        case_rows,
        balanced_rows,
        endpoint_rows,
        endpoint_cases,
    )
    manifest = {
        "schema_version": 1,
        "cases": list(cases),
        "balanced_cases": balanced_cases,
        "balanced_methods": list(BALANCED_METHODS),
        "endpoint_ablation_cases": endpoint_cases,
        "endpoint_ablation_methods": list(ENDPOINT_ABLATION_METHODS),
        "repetitions": args.repetitions,
        "selected_run_count": len(selected),
        "superseded_attempt_count": len(superseded),
        "nonterminal_selected_count": sum(
            not record.terminal_available for record in selected
        ),
        "faulted_selected_count": sum(record.faulted for record in selected),
        "lewm_runs": str(args.lewm_runs.expanduser().resolve()),
        "discrete_runs": str(args.discrete_runs.expanduser().resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(f"selected runs: {len(selected)}")
    print(f"superseded attempts: {len(superseded)}")
    print(f"balanced cases: {', '.join(balanced_cases)}")
    print(f"endpoint ablation cases: {', '.join(endpoint_cases)}")
    print(f"analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
