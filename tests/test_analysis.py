"""Analysis and checkpoint tests use scalar fixtures, not unverified simulation outcomes."""

import csv
import io
import json
import math
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace

import pytest

from warehouse_sim import cli, experiments
from warehouse_sim.analysis import (
    analyze,
    mean_ci,
    paired_degradation,
    summarize_pairs,
    summarize_runs,
)
from warehouse_sim.domain import Config


def observation(strategy="S1", scenario="D0", seed=0, layout="A", **metrics):
    return {
        "strategy": strategy,
        "scenario": scenario,
        "seed": seed,
        "layout": layout,
        "metrics": metrics,
    }


def test_student_t_interval_analytic():
    result = mean_ci([1, 2, 3])
    # df=2 two-sided 95% t critical value = 4.302652729911275; sample SD = 1.
    half_width = 4.302652729911275 / math.sqrt(3)
    assert result["mean"] == 2
    assert result["ci_low"] == pytest.approx(2 - half_width)
    assert result["ci_high"] == pytest.approx(2 + half_width)
    assert result["n"] == 3
    assert result["n_missing"] == 0


def test_singleton_and_missing_values_do_not_invent_ci():
    result = mean_ci([None, 4, float("nan"), float("inf")])
    assert result["mean"] == 4
    assert result["n"] == 1
    assert result["n_missing"] == 3
    assert result["ci_low"] is None and result["ci_high"] is None
    empty = mean_ci([None])
    assert empty["mean"] is None
    assert empty["ci_low"] is None
    assert mean_ci([4, 4])["ci_low"] == 4
    with pytest.raises(ValueError):
        mean_ci([1, 2], confidence=1)


def test_summary_uses_per_run_means_and_keeps_failed_runs():
    runs = [
        observation(mean_completion_time=10, execution_failures=0, completed_tasks=10),
        observation(seed=1, mean_completion_time=30, execution_failures=0, completed_tasks=1),
        observation(seed=2, mean_completion_time=None, execution_failures=1, completed_tasks=0),
    ]
    rows = {row["metric"]: row for row in summarize_runs(runs, expected_replications=4)}
    completion = rows["mean_completion_time"]
    assert completion["mean"] == 20  # Not task-weighted (130 / 11).
    assert completion["n"] == 2
    assert completion["n_missing"] == 1
    assert completion["n_total"] == 3
    assert completion["n_missing_runs"] == 1
    assert rows["failure_rate"]["mean"] == pytest.approx(1 / 3)
    assert rows["failure_rate"]["n"] == 3
    assert rows["execution_failures"]["n"] == 3


def test_pairing_uses_own_strategy_layout_and_seed_nominal():
    runs = [
        observation("S1", "D0", throughput=10),
        observation("S1", "D4", throughput=5),
        observation("S2", "D0", throughput=20),
        observation("S2", "D4", throughput=30),
        observation("S2", "D0", seed=1, throughput=4),
        observation("S2", "D4", seed=1, throughput=8),
        observation("S2", "D0", layout="B", throughput=100),
        observation("S2", "D4", layout="B", throughput=75),
    ]
    pairs = [row for row in paired_degradation(runs) if row["metric"] == "throughput"]
    changes = {(row["strategy"], row["layout"], row["seed"]): row["change"] for row in pairs}
    assert changes == {
        ("S1", "A", 0): -0.5,
        ("S2", "A", 0): 0.5,
        ("S2", "A", 1): 1,
        ("S2", "B", 0): -0.25,
    }
    summary = [
        row for row in summarize_pairs(pairs) if row["strategy"] == "S2" and row["layout"] == "A"
    ][0]
    assert summary["mean"] == 0.75
    assert summary["n"] == 2
    assert summary["ci_low"] is not None


def test_zero_baselines_are_labeled_and_never_pooled_with_ratios():
    runs = [
        observation("S1", "D0", seed=0, throughput=0),
        observation("S1", "D4", seed=0, throughput=5),
        observation("S1", "D0", seed=1, throughput=10),
        observation("S1", "D4", seed=1, throughput=15),
        observation("S1", "D0", seed=2, throughput=0),
        observation("S1", "D4", seed=2, throughput=0),
    ]
    pairs = [row for row in paired_degradation(runs) if row["metric"] == "throughput"]
    summary = {row["mode"]: row for row in summarize_pairs(pairs)}
    assert summary["absolute_change"]["mean"] == 2.5
    assert summary["absolute_change"]["n"] == 2
    assert summary["relative_change"]["mean"] == 0.5
    assert summary["relative_change"]["n"] == 1
    assert summary["relative_change"]["ci_high"] is None


def test_missing_completion_and_nominal_pairs_remain_missing():
    runs = [
        observation("S1", "D0", mean_completion_time=10, execution_failures=0),
        observation("S1", "D4", mean_completion_time=None, execution_failures=1),
        observation("S2", "D4", mean_completion_time=20, execution_failures=0),
        observation("S3", "D0", mean_completion_time=None, execution_failures=1),
        observation("S3", "D4", mean_completion_time=20, execution_failures=0),
    ]
    pairs = [row for row in paired_degradation(runs) if row["metric"] == "mean_completion_time"]
    assert pairs[0]["mode"] == "relative_change"
    assert pairs[0]["change"] is None
    assert pairs[0]["disrupted_failure"] == 1
    assert pairs[1]["mode"] == "missing_baseline"
    assert pairs[2]["mode"] == "missing_baseline"
    assert all(row["change"] is None for row in pairs)
    assert all(row["n"] == 0 for row in summarize_pairs(pairs))


def test_duplicate_run_identities_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        paired_degradation([observation(), observation()])


@pytest.fixture
def fake_worker(monkeypatch):
    calls = []

    def worker(config, spec):
        calls.append(spec)
        metrics: dict[str, int | None] = dict.fromkeys(experiments.METRICS, 0)
        failed = spec["scenario"] != "D0" and spec["seed"] == 1
        metrics.update(
            {
                "mean_completion_time": None if failed else 10 + spec["seed"],
                "throughput": 0 if failed else 2,
                "execution_failures": int(failed),
                "completed_tasks": 0 if failed else config["total_tasks"],
                "unfinished_tasks": config["total_tasks"] if failed else 0,
            }
        )
        return {
            **spec,
            "config": config,
            "metrics": metrics,
            "tasks": [{"id": 0}],
            "vehicles": [{"id": 0}],
            "disruptions": [],
            "trace": [],
            "messages": {},
        }

    monkeypatch.setattr(experiments, "_worker", worker)
    monkeypatch.setattr(experiments, "source_hash", lambda: "fixed-source-hash")
    monkeypatch.setattr(experiments, "package_versions", lambda: {"python": "test"})
    return calls


def small_experiment(directory, **kwargs):
    arguments = {
        "output": directory,
        "replications": 2,
        "workers": 1,
        "strategies": ["S1"],
        "scenarios": ["D0", "D4"],
        "layouts": ["A"],
        "progress": io.StringIO(),
    }
    arguments.update(kwargs)
    return experiments.run_experiment(**arguments)


def test_streamed_checkpoints_csv_and_resume(tmp_path, fake_worker):
    manifest = small_experiment(tmp_path)
    assert manifest["specification"]["seeds"] == [0, 1]
    assert manifest["config"] == asdict(Config())
    assert len(fake_worker) == 4
    assert (tmp_path / "runs" / "A_S1_D4_001.json").exists()
    artifact = json.loads((tmp_path / "runs" / "A_S1_D0_000.json").read_text())
    assert artifact["tasks"] == [{"id": 0}]
    with (tmp_path / "runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "tasks" not in rows[0]
    assert rows[-1]["execution_failures"] == "1"
    assert rows[-1]["mean_completion_time"] == ""
    # A broken append is rebuilt from authoritative JSON, not reused or duplicated.
    with (tmp_path / "runs.csv").open("a") as handle:
        handle.write("partial,csv,row")
    assert small_experiment(tmp_path, resume=True) == manifest
    assert len(fake_worker) == 4
    with (tmp_path / "runs.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert json.loads((tmp_path / "checkpoint.json").read_text())["status"] == "complete"


@pytest.mark.parametrize("change", ["config", "source", "specs", "packages"])
def test_resume_invalidates_incompatible_inputs(tmp_path, fake_worker, monkeypatch, change):
    small_experiment(tmp_path)
    arguments: dict = {"resume": True}
    if change == "config":
        arguments["config"] = replace(Config(), max_ticks=900)
    elif change == "source":
        monkeypatch.setattr(experiments, "source_hash", lambda: "different-source")
    elif change == "packages":
        monkeypatch.setattr(experiments, "package_versions", lambda: {"python": "other"})
    else:
        arguments["scenarios"] = ["D0"]
    with pytest.raises(ValueError, match="Cannot resume"):
        small_experiment(tmp_path, **arguments)
    assert len(fake_worker) == 4


def test_resume_rejects_edited_artifact(tmp_path, fake_worker):
    small_experiment(tmp_path)
    path = tmp_path / "runs" / "A_S1_D0_000.json"
    artifact = json.loads(path.read_text())
    artifact["metrics"]["throughput"] = 999
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError, match="content hash"):
        small_experiment(tmp_path, resume=True)


def test_interruption_keeps_completed_runs_and_resume_only_runs_missing(
    tmp_path,
    fake_worker,
    monkeypatch,
):
    original = experiments._worker

    def interrupted(config, spec):
        if len(fake_worker) == 1:
            raise RuntimeError("worker interrupted")
        return original(config, spec)

    monkeypatch.setattr(experiments, "_worker", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        small_experiment(tmp_path)
    assert len(list((tmp_path / "runs").glob("*.json"))) == 1
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["completed"] == 1
    assert checkpoint["status"] == "interrupted"
    monkeypatch.setattr(experiments, "_worker", original)
    small_experiment(tmp_path, resume=True)
    assert len(fake_worker) == 4
    assert len(list((tmp_path / "runs").glob("*.json"))) == 4


def test_existing_output_requires_explicit_resume(tmp_path, fake_worker):
    small_experiment(tmp_path)
    with pytest.raises(ValueError, match="--resume"):
        small_experiment(tmp_path)


def test_source_hash_ignores_results(tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n")
    before = experiments.source_hash(tmp_path)
    (tmp_path / "results.json").write_text('{"generated": true}')
    assert experiments.source_hash(tmp_path) == before
    (tmp_path / "module.py").write_text("x = 2\n")
    assert experiments.source_hash(tmp_path) != before


def test_atomic_json_preserves_old_checkpoint_on_serialization_error(tmp_path):
    path = tmp_path / "checkpoint.json"
    experiments.atomic_json(path, {"complete": True})
    with pytest.raises(ValueError):
        experiments.atomic_json(path, {"mean": float("nan")})
    assert json.loads(path.read_text()) == {"complete": True}
    assert list(tmp_path.iterdir()) == [path]


def test_analyze_rebuilds_reports_and_agg_plots(tmp_path, fake_worker):
    small_experiment(tmp_path)
    result = analyze(tmp_path)
    assert result["runs"] == 4
    assert len(result["plots"]) == 4
    for path in (tmp_path / "plots").glob("*.png"):
        assert path.read_bytes().startswith(b"\x89PNG")
    with (tmp_path / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    failure = next(
        row for row in rows if row["scenario"] == "D4" and row["metric"] == "failure_rate"
    )
    assert float(failure["mean"]) == 0.5
    assert failure["n"] == "2"
    report = (tmp_path / "report.md").read_text()
    assert "4 / 4" in report
    assert "not a proof of deadlock" in report
    assert "No universal strategy winner" in report
    original = (tmp_path / "summary.csv").read_text()
    analyze(tmp_path, plots=False)
    assert (tmp_path / "summary.csv").read_text() == original


def test_cli_contracts(tmp_path, monkeypatch, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "run",
            "--strategy",
            "S1",
            "--scenario",
            "D4",
            "--layout",
            "A",
            "--seed",
            "0",
            "--output",
            str(tmp_path / "example.json"),
            "--trace",
        ]
    )
    assert args.trace and args.seed == 0
    args = parser.parse_args(
        [
            "experiment",
            "--output",
            str(tmp_path),
            "--replications",
            "30",
            "--workers",
            "4",
            "--config",
            "configs/study.json",
            "--strategies",
            "S1",
            "S2",
            "--scenarios",
            "D0",
            "D4",
            "--layouts",
            "A",
            "B",
            "--resume",
        ]
    )
    assert args.strategies == ["S1", "S2"]
    assert args.scenarios == ["D0", "D4"]
    assert args.resume
    monkeypatch.setattr(cli, "analyze", lambda path: {"input": str(path)})
    assert cli.main(["analyze", "--input", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["input"] == str(tmp_path)
    with pytest.raises(SystemExit):
        parser.parse_args(["experiment", "--output", str(tmp_path), "--workers", "0"])


def test_default_full_design_has_3000_matched_runs():
    manifest = {
        "specification": {
            "layouts": ["A", "B"],
            "strategies": list(experiments.STRATEGIES),
            "scenarios": list(experiments.SCENARIOS),
            "seeds": list(range(30)),
        }
    }
    specs = experiments.experiment_specs(manifest)
    assert len(specs) == 3000
    assert {spec["seed"] for spec in specs if spec["scenario"] == "D0"} == set(range(30))


def test_parallel_submission_and_checkpoints(tmp_path, fake_worker, monkeypatch):
    # Exercise the completion-driven scheduling path without requiring a model module.
    monkeypatch.setattr(experiments, "ProcessPoolExecutor", ThreadPoolExecutor)
    small_experiment(tmp_path, workers=2)
    assert len(fake_worker) == 4
    assert len(list(experiments.iter_artifacts(tmp_path))) == 4
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    assert checkpoint["completed"] == 4


def test_pilot_fixed_candidates_share_design_and_never_auto_select(
    tmp_path,
    fake_worker,
    monkeypatch,
):
    monkeypatch.setattr("warehouse_sim.analysis.analyze", lambda path: None)
    manifest = experiments.run_pilot(tmp_path, workers=1)
    assert manifest["seeds"] == [0, 1]
    assert manifest["strategies"] == ["S1", "S2", "S4", "S8"]
    assert len(fake_worker) == 64
    candidates = manifest["candidates"]
    assert candidates["v4_t48_i8"]["arrival_interval"] == 8
    assert candidates["v4_t48_i5"]["arrival_interval"] == 5
    assert "winner" not in manifest
    with (tmp_path / "pilot_runs.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 64
    assert len({row["candidate"] for row in rows}) == 2
    experiments.run_pilot(tmp_path, workers=1, resume=True)
    assert len(fake_worker) == 64


def test_single_run_imports_model_lazily_and_preserves_trace(tmp_path, monkeypatch):
    module = types.ModuleType("warehouse_sim.model")
    received = {}

    class Model:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def run(self):
            return {"metrics": {}, "trace": [{"tick": 0}]}

    monkeypatch.setattr(module, "WarehouseModel", Model, raising=False)
    monkeypatch.setitem(sys.modules, "warehouse_sim.model", module)
    path = tmp_path / "example.json"
    result = experiments.run_single(scenario="D4", trace=True, output=path)
    assert received["scenario"] == "D4"
    assert received["trace"] is True
    assert received["config"] == Config()
    assert json.loads(path.read_text()) == result
    assert result["trace"] == [{"tick": 0}]


def test_empty_design_group_has_missing_run_counts():
    summary = summarize_runs([], expected_replications=30, expected_groups=[("A", "S1", "D0")])
    assert len(summary) == len(experiments.METRICS) + 1
    assert all(row["n"] == 0 and row["n_missing_runs"] == 30 for row in summary)
    assert all(row["mean"] is None for row in summary)
