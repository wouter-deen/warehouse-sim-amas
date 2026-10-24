"""Failure-aware summaries of run-level observations, not pooled task-level statistics.

Student-t intervals describe uncertainty across independent replication seeds. Paired
changes compare each strategy/layout to its own D0 at the same seed. They are signed
raw relative changes, not direction-adjusted robustness scores or causal guarantees.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import t

from warehouse_sim.experiments import METRICS, iter_artifacts

SUMMARY_FIELDS = (
    "layout",
    "strategy",
    "scenario",
    "metric",
    "n",
    "n_missing",
    "n_total",
    "n_expected",
    "n_missing_runs",
    "mean",
    "ci_low",
    "ci_high",
    "confidence",
)
PAIRED_FIELDS = (
    "layout",
    "strategy",
    "scenario",
    "seed",
    "metric",
    "nominal",
    "disrupted",
    "mode",
    "change",
    "nominal_failure",
    "disrupted_failure",
)
PAIRED_SUMMARY_FIELDS = (
    "layout",
    "strategy",
    "scenario",
    "metric",
    "mode",
    "n",
    "n_missing",
    "n_total",
    "n_expected",
    "n_missing_runs",
    "mean",
    "ci_low",
    "ci_high",
    "confidence",
)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean_ci(values, confidence: float = 0.95) -> dict:
    """Mean and two-sided Student-t CI; fewer than two finite values has no CI.

    None/nonfinite observations count as missing, never as zero. The interval is not
    clipped to metric bounds. This is a marginal CI, with no multiplicity correction.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between zero and one")
    observations = [_number(value) for value in values]
    finite = [value for value in observations if value is not None]
    n = len(finite)
    mean = statistics.mean(finite) if n else None
    low = high = None
    if n >= 2 and mean is not None:
        half_width = float(t.ppf((1 + confidence) / 2, n - 1)) * statistics.stdev(finite)
        half_width /= math.sqrt(n)
        low, high = mean - half_width, mean + half_width
    return {
        "n": n,
        "n_missing": len(observations) - n,
        "n_total": len(observations),
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "confidence": confidence,
    }


def _metric_names(runs: list[dict]) -> list[str]:
    return list(dict.fromkeys([*METRICS, *sorted({k for r in runs for k in r["metrics"]})]))


def summarize_runs(
    runs: list[dict],
    confidence: float = 0.95,
    expected_replications: int | None = None,
    expected_groups: list[tuple[str, str, str]] | None = None,
) -> list[dict]:
    """Long-form summaries of every reported metric, including failures from every run."""
    groups = defaultdict(list)
    for key in expected_groups or []:
        groups[key] = []
    for run in runs:
        groups[(run["layout"], run["strategy"], run["scenario"])].append(run)
    rows = []
    metrics = [*_metric_names(runs), "failure_rate"]
    for (layout, strategy, scenario), group in sorted(groups.items()):
        for metric in metrics:
            key = "execution_failures" if metric == "failure_rate" else metric
            stats = mean_ci((run["metrics"].get(key) for run in group), confidence)
            expected = expected_replications if expected_replications is not None else len(group)
            rows.append(
                {
                    "layout": layout,
                    "strategy": strategy,
                    "scenario": scenario,
                    "metric": metric,
                    **stats,
                    "n_expected": expected,
                    "n_missing_runs": max(0, expected - len(group)),
                }
            )
    return rows


def paired_degradation(runs: list[dict]) -> list[dict]:
    """Return per-seed signed changes, explicitly separating zero-baseline units.

    `relative_change` = (disrupted - nominal) / nominal (a fraction, not percent).
    `absolute_change` = disrupted - nominal when nominal is zero. No epsilon is
    substituted. A missing nominal uses `missing_baseline`; a missing disrupted
    observation retains its nominal's mode but has a missing change.
    """
    index = {}
    for run in runs:
        key = (run["layout"], run["strategy"], run["scenario"], run["seed"])
        if key in index:
            raise ValueError(f"Duplicate run identity: {key}")
        index[key] = run
    rows = []
    metrics = _metric_names(runs)
    for run in runs:
        if run["scenario"] == "D0":
            continue
        nominal_run = index.get((run["layout"], run["strategy"], "D0", run["seed"]))
        for metric in metrics:
            nominal = _number(nominal_run["metrics"].get(metric)) if nominal_run else None
            disrupted = _number(run["metrics"].get(metric))
            change = None
            if nominal is None:
                mode = "missing_baseline"
            elif nominal == 0:
                mode = "absolute_change"
                if disrupted is not None:
                    change = disrupted - nominal
            else:
                mode = "relative_change"
                if disrupted is not None:
                    change = (disrupted - nominal) / nominal
            rows.append(
                {
                    "layout": run["layout"],
                    "strategy": run["strategy"],
                    "scenario": run["scenario"],
                    "seed": run["seed"],
                    "metric": metric,
                    "nominal": nominal,
                    "disrupted": disrupted,
                    "mode": mode,
                    "change": change,
                    "nominal_failure": nominal_run["metrics"].get("execution_failures")
                    if nominal_run
                    else None,
                    "disrupted_failure": run["metrics"].get("execution_failures"),
                }
            )
    return rows


def summarize_pairs(
    pairs: list[dict],
    confidence: float = 0.95,
    expected_replications: int | None = None,
) -> list[dict]:
    """Aggregate only like-unit paired changes; `n` counts usable matched-seed values."""
    groups = defaultdict(list)
    present = defaultdict(set)
    for pair in pairs:
        key = tuple(pair[k] for k in ("layout", "strategy", "scenario", "metric"))
        groups[(*key, pair["mode"])].append(pair["change"])
        present[key].add(pair["seed"])
    rows = []
    for (layout, strategy, scenario, metric, mode), values in sorted(groups.items()):
        count = len(present[(layout, strategy, scenario, metric)])
        expected = expected_replications if expected_replications is not None else count
        rows.append(
            {
                "layout": layout,
                "strategy": strategy,
                "scenario": scenario,
                "metric": metric,
                "mode": mode,
                **mean_ci(values, confidence),
                "n_expected": expected,
                "n_missing_runs": max(0, expected - count),
            }
        )
    return rows


def _write_csv(path: Path, rows, fields) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _strategy_key(strategy: str):
    return int(strategy[1:]) if strategy[1:].isdigit() else strategy


def _plots(root: Path, summary: list[dict], paired: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    plot_dir = root / "plots"
    plot_dir.mkdir(exist_ok=True)
    paths = []
    layouts = sorted({row["layout"] for row in summary})

    def bars(ax, rows, metric, title, mode=None):
        selected = [
            row
            for row in rows
            if row["metric"] == metric and (mode is None or row.get("mode") == mode)
        ]
        strategies = sorted({r["strategy"] for r in selected}, key=_strategy_key)
        scenarios = sorted({r["scenario"] for r in selected})
        if not selected or not any(row["mean"] is not None for row in selected):
            ax.text(0.5, 0.5, "No estimable observations", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        lookup = {(r["strategy"], r["scenario"]): r for r in selected}
        width = 0.8 / max(1, len(scenarios))
        for j, scenario in enumerate(scenarios):
            positions, means, errors = [], [], []
            for i, strategy in enumerate(strategies):
                row = lookup.get((strategy, scenario))
                positions.append(i - 0.4 + width / 2 + j * width)
                means.append(row["mean"] if row and row["mean"] is not None else math.nan)
                errors.append(
                    (row["ci_high"] - row["mean"]) if row and row["ci_high"] is not None else 0
                )
            ax.bar(positions, means, width, yerr=errors, capsize=2, label=scenario)
        ax.set_xticks(range(len(strategies)), strategies)
        ax.set_title(title)
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.legend(fontsize="small")
        ax.grid(axis="y", alpha=0.2)

    for layout in layouts:
        group = [row for row in summary if row["layout"] == layout]
        pairs = [row for row in paired if row["layout"] == layout]
        fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
        for ax, metric, title in zip(
            axes,
            ("mean_completion_time", "throughput", "failure_rate"),
            (
                "Mean completion time (completed tasks only; ticks)",
                "Throughput (model-reported units)",
                "Execution failure rate (all runs)",
            ),
            strict=True,
        ):
            bars(ax, group, metric, title)
        fig.suptitle(f"Layout {layout}: service and failures; mean and 95% Student-t CI")
        path = plot_dir / f"{layout}_service.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
        bars(ax, group, "sum_of_costs", f"Layout {layout}: reported sum of costs (SOC)")
        ax.set_ylabel("Mean SOC; 95% Student-t CI")
        path = plot_dir / f"{layout}_soc.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
        bars(axes[0], group, "utilization", "Mean fleet utilization (busy includes waiting)")
        bars(
            axes[1], group, "communication_volume", "Delivered inter-agent messages (counted once)"
        )
        fig.suptitle(f"Layout {layout}: utilization and communication; 95% t CI")
        path = plot_dir / f"{layout}_coordination.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
        for i, metric in enumerate(("mean_completion_time", "throughput", "sum_of_costs")):
            bars(
                axes[i, 0],
                pairs,
                metric,
                f"{metric}: signed relative change",
                mode="relative_change",
            )
            axes[i, 0].set_ylabel("(disrupted - D0) / D0")
            bars(
                axes[i, 1],
                pairs,
                metric,
                f"{metric}: zero-D0 absolute change",
                mode="absolute_change",
            )
            axes[i, 1].set_ylabel("disrupted - D0 (metric units)")
        fig.suptitle(f"Layout {layout}: matched-seed changes; modes never pooled; 95% t CI")
        path = plot_dir / f"{layout}_robustness.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def _format(value) -> str:
    return "missing" if value is None else f"{value:.4g}"


def _report(root: Path, manifest: dict, runs, summary, paired_summary, plots) -> None:
    spec = manifest["specification"]
    expected = len(spec["layouts"]) * len(spec["strategies"]) * len(spec["scenarios"])
    expected *= len(spec["seeds"])
    failures = sum(run["metrics"].get("execution_failures", 0) for run in runs)
    lines = [
        "# Warehouse experiment report",
        "",
        f"Artifacts available: **{len(runs)} / {expected}**. "
        f"Execution failures among available runs: **{failures} / {len(runs)}**.",
        "",
        "This report is computed from saved run artifacts; unavailable runs are not fabricated. "
        "An incomplete experiment is not a complete factorial comparison.",
        "",
        "## Reproducibility",
        "",
        f"- Experiment fingerprint: `{manifest['fingerprint']}`",
        f"- Source hash: `{manifest['source_hash']}`",
        f"- Matched seeds: `{spec['seeds']}`",
        f"- Strategies: `{spec['strategies']}`; scenarios: `{spec['scenarios']}`; "
        f"layouts: `{spec['layouts']}`.",
        f"- Frozen config: `{json.dumps(manifest['config'], sort_keys=True)}`",
        f"- Package versions: `{json.dumps(manifest['package_versions'], sort_keys=True)}`",
        "",
        "## Estimands and limitations",
        "",
        "- `summary.csv` reports means and two-sided 95% Student-t confidence intervals "
        "across per-run metrics, not pooled tasks. The default design has 30 replications, "
        "but actual n is always reported. Fewer than two finite observations has no CI.",
        "- `n` counts finite observations; `n_missing` counts missing metric values in saved "
        "runs; `n_missing_runs` counts unsaved runs. Missing values are never replaced by zero. "
        "Intervals are marginal, not simultaneous or adjusted for multiple comparisons; "
        "bounded metrics may have intervals extending outside their bounds.",
        "- Mean completion time and pickup delay can be missing. Completion time summarizes "
        "completed tasks only and can look artificially favorable when difficult tasks remain "
        "unfinished. Read it alongside throughput, unfinished tasks and failure rate.",
        "- `failure_rate` is the mean of binary `execution_failures` over **all saved runs**, "
        "including runs with no completed tasks. Simulator exceptions are not reclassified "
        "as measured execution failures: they interrupt the experiment.",
        "- `paired_degradation.csv` compares each disrupted run with its **own strategy, "
        "layout and seed** in D0. `relative_change = (disrupted - nominal) / nominal` is a "
        "signed fraction. Positive throughput/utilization change is an increase (often "
        "favorable), while positive completion-time change is slower; signs are not flipped. "
        "Higher utilization alone is not evidence of better service.",
        "- Zero nominal values use `absolute_change = disrupted - nominal`, labeled in "
        "metric units. `paired_summary.csv` aggregates relative and absolute modes separately "
        "and reports each mode's counts. Missing baselines remain `missing_baseline`, not "
        "infinite or zero degradation. In paired summaries `n_total`/`n_missing` apply to that "
        "mode; `n_missing_runs` applies to the whole strategy/layout/scenario metric group.",
        "- SOC, makespan and other finite-horizon costs must be read with delivered service "
        "and failures: lower cost can reflect unfinished work rather than efficiency. "
        "No universal strategy winner is inferred by this report.",
        "- `deadlock_episodes` diagnoses prolonged no-progress, not a proof of deadlock. "
        "Disruption counts and communication volume describe observed simulation behavior, "
        "not external operational validation.",
        "",
        "## Failure-aware service overview",
        "",
        "| Layout | Strategy | Scenario | Failure rate | Completion time | Throughput "
        "| Unfinished tasks |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lookup = defaultdict(dict)
    for row in summary:
        lookup[(row["layout"], row["strategy"], row["scenario"])][row["metric"]] = row
    for (layout, strategy, scenario), metrics in sorted(lookup.items()):
        cells = []
        for metric in ("failure_rate", "mean_completion_time", "throughput", "unfinished_tasks"):
            row = metrics[metric]
            cells.append(f"{_format(row['mean'])} (n={row['n']}, missing={row['n_missing']})")
        lines.append(f"| {layout} | {strategy} | {scenario} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Full intervals and all metrics are in `summary.csv`. Matched change intervals "
        "are in `paired_summary.csv`; per-seed values are in `paired_degradation.csv`.",
        "",
    ]
    zero_count = sum(row["n_total"] for row in paired_summary if row["mode"] == "absolute_change")
    missing_pairs = sum(row["n_missing"] for row in paired_summary)
    lines += [
        f"Zero-baseline metric pairs: **{zero_count}**. Missing paired metric observations: "
        f"**{missing_pairs}** (counts span metrics, not distinct runs).",
        "",
        "## Plots",
        "",
        "Bars show available-run means with 95% t CIs. Missing bars are not zero; "
        "singletons have no error bar. Robustness panels keep relative and absolute "
        "units separate.",
        "",
    ]
    for path in plots:
        lines += [f"![{path.stem}]({path.relative_to(root).as_posix()})", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def analyze(input_dir: str | Path, plots: bool = True) -> dict:
    """Recompute CSVs, a factual Markdown report and offline Agg plots from checkpoints."""
    root = Path(input_dir)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    # Keep scalar metrics only: potentially large task/vehicle/trace records stay on disk.
    runs = [
        {key: run[key] for key in ("layout", "strategy", "scenario", "seed", "metrics")}
        for run in iter_artifacts(root)
    ]
    spec = manifest["specification"]
    expected = len(spec["seeds"])
    expected_groups = [
        (layout, strategy, scenario)
        for layout in spec["layouts"]
        for strategy in spec["strategies"]
        for scenario in spec["scenarios"]
    ]
    summary = summarize_runs(runs, expected_replications=expected, expected_groups=expected_groups)
    pairs = paired_degradation(runs)
    paired_summary = summarize_pairs(pairs, expected_replications=expected)
    _write_csv(root / "summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(root / "paired_degradation.csv", pairs, PAIRED_FIELDS)
    _write_csv(root / "paired_summary.csv", paired_summary, PAIRED_SUMMARY_FIELDS)
    paths = _plots(root, summary, paired_summary) if plots else []
    _report(root, manifest, runs, summary, paired_summary, paths)
    return {
        "runs": len(runs),
        "summary": str(root / "summary.csv"),
        "paired_degradation": str(root / "paired_degradation.csv"),
        "paired_summary": str(root / "paired_summary.csv"),
        "report": str(root / "report.md"),
        "plots": [str(path) for path in paths],
    }
