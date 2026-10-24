"""Reproducible, checkpointed experiments; model imports happen only inside workers.

Run JSON files are authoritative. CSV is a rebuildable, flattened convenience view;
completed checkpoints survive interruption even between a JSON and CSV write.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from warehouse_sim.domain import SCENARIOS, STRATEGIES, Config

METRICS = (
    "mean_completion_time",
    "throughput",
    "unfinished_tasks",
    "utilization",
    "sum_of_costs",
    "communication_volume",
    "pickup_delay",
    "makespan",
    "sum_of_moves",
    "empty_travel_ratio",
    "idle_time",
    "load_balance_cv",
    "execution_failures",
    "deadlock_episodes",
    "detected_disruptions",
    "completed_tasks",
    "simulation_ticks",
    "disruption_proposals",
    "accepted_disruptions",
    "skipped_disruptions",
    "replans",
    "failed_replans",
)
SCHEMA_VERSION = 1
DEFAULT_CONFIG = Config()


def fingerprint(value: Any) -> str:
    """Hash canonical JSON, rejecting NaN and infinity rather than emitting invalid JSON."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def source_hash(source_root: Path | None = None) -> str:
    """Hash package Python source and build/lock files, never generated result artifacts."""
    root = source_root or Path(__file__).resolve().parent
    paths = sorted(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    if source_root is None:
        project = root.parent.parent
        for name in ("pyproject.toml", "uv.lock"):
            path = project / name
            if path.is_file():
                digest.update(name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for name in ("warehouse-agents-simulation", "mesa", "scipy", "matplotlib", "numpy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def atomic_json(path: Path, value: Any) -> None:
    """Replace a checkpoint only after its complete JSON has been flushed to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config must be a JSON object of Config fields")
    return Config(**data)


def run_single(
    config: Config = DEFAULT_CONFIG,
    strategy: str = "S1",
    scenario: str = "D0",
    layout: str = "A",
    seed: int = 0,
    trace: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one model run, preserving task, vehicle, disruption and optional trace records."""
    _validate_choices((strategy,), (scenario,), (layout,))
    from warehouse_sim.model import WarehouseModel

    result = WarehouseModel(
        config=config, strategy=strategy, scenario=scenario, layout=layout, seed=seed, trace=trace
    ).run()
    if output is not None:
        atomic_json(Path(output), result)
    return result


def _validate_choices(strategies, scenarios, layouts) -> None:
    for label, choices, allowed in (
        ("strategies", strategies, STRATEGIES),
        ("scenarios", scenarios, SCENARIOS),
        ("layouts", layouts, ("A", "B")),
    ):
        if not choices or len(set(choices)) != len(choices):
            raise ValueError(f"{label} must be nonempty and contain no duplicates")
        unknown = set(choices) - set(allowed)
        if unknown:
            raise ValueError(f"Unknown {label}: {sorted(unknown)}")


def artifact_name(layout: str, strategy: str, scenario: str, seed: int) -> str:
    return f"{layout}_{strategy}_{scenario}_{seed:03d}.json"


def experiment_specs(manifest: dict) -> list[dict]:
    spec = manifest["specification"]
    return [
        {"layout": layout, "strategy": strategy, "scenario": scenario, "seed": seed}
        for layout in spec["layouts"]
        for strategy in spec["strategies"]
        for scenario in spec["scenarios"]
        for seed in spec["seeds"]
    ]


def _run_fingerprint(manifest: dict, spec: dict) -> str:
    return fingerprint({"experiment": manifest["fingerprint"], "run": spec})


def _check_result(result: dict, manifest: dict, spec: dict) -> None:
    if any(result.get(key) != value for key, value in spec.items()):
        raise ValueError(f"Run identity mismatch: {spec}")
    if result.get("config") != manifest["config"]:
        raise ValueError(f"Run config mismatch: {spec}")
    if not isinstance(result.get("metrics"), dict) or set(METRICS) - result["metrics"].keys():
        raise ValueError(f"Missing required run metrics: {spec}")
    if result["metrics"]["execution_failures"] not in (0, 1):
        raise ValueError(f"execution_failures must be binary (0/1): {spec}")
    for key in ("tasks", "vehicles", "disruptions", "trace"):
        if not isinstance(result.get(key), list):
            raise ValueError(f"Run must include {key} records: {spec}")
    if not isinstance(result.get("messages"), dict):
        raise ValueError(f"Run must include messages: {spec}")


def read_artifact(path: Path, manifest: dict, spec: dict) -> dict:
    """Reject foreign, edited, or incompatible checkpoints rather than silently reusing them."""
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    _check_result(result, manifest, spec)
    metadata = result.get("_experiment", {})
    if metadata.get("run_fingerprint") != _run_fingerprint(manifest, spec):
        raise ValueError(f"Incompatible checkpoint fingerprint: {path}")
    content = {key: value for key, value in result.items() if key != "_experiment"}
    if metadata.get("content_hash") != fingerprint(content):
        raise ValueError(f"Checkpoint content hash mismatch: {path}")
    return result


def iter_artifacts(directory: str | Path):
    """Yield validated run checkpoints; incomplete experiments can also be analyzed."""
    root = Path(directory)
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    for spec in experiment_specs(manifest):
        path = root / "runs" / artifact_name(**spec)
        if path.exists():
            yield read_artifact(path, manifest, spec)


def flattened_run(result: dict) -> dict:
    row = {key: result[key] for key in ("layout", "strategy", "scenario", "seed")}
    row.update({f"config_{key}": value for key, value in result["config"].items()})
    row.update({key: result["metrics"].get(key) for key in METRICS})
    return row


def _csv_fields(config: Config) -> list[str]:
    return [
        "layout",
        "strategy",
        "scenario",
        "seed",
        *[f"config_{k}" for k in asdict(config)],
        *METRICS,
    ]


def _worker(config: dict, spec: dict) -> dict:
    return run_single(config=Config(**config), **spec)


def run_experiment(
    output: str | Path,
    config: Config = DEFAULT_CONFIG,
    replications: int = 30,
    workers: int = 4,
    strategies=None,
    scenarios=None,
    layouts=None,
    resume: bool = False,
    progress: TextIO | None = None,
) -> dict:
    """Run a matched-seed factorial design with bounded in-flight jobs and atomic checkpoints.

    The default design is 10 strategies x 5 scenarios x 2 layouts x 30 seeds = 3000.
    Resume requires identical config, design, source, schema and dependency versions.
    Worker exceptions abort the experiment; they are not fabricated model failures.
    """
    strategies = list(STRATEGIES if strategies is None else strategies)
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    layouts = list(("A", "B") if layouts is None else layouts)
    _validate_choices(strategies, scenarios, layouts)
    if replications < 1 or workers < 1:
        raise ValueError("replications and workers must be positive")
    root = Path(output)
    manifest_path = root / "manifest.json"
    identity = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "config_fingerprint": fingerprint(asdict(config)),
        "source_hash": source_hash(),
        "package_versions": package_versions(),
        "specification": {
            "strategies": strategies,
            "scenarios": scenarios,
            "layouts": layouts,
            "seeds": list(range(replications)),
            "replications": replications,
            "trace": False,
        },
    }
    expected_fingerprint = fingerprint(identity)
    if manifest_path.exists():
        if not resume:
            raise ValueError("Output already has a manifest; use --resume or a new directory")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        stored_identity = {key: manifest.get(key) for key in identity}
        if (
            manifest.get("fingerprint") != expected_fingerprint
            or fingerprint(stored_identity) != expected_fingerprint
        ):
            raise ValueError("Cannot resume: config, experiment specs, source or packages changed")
    else:
        if root.exists() and any(root.iterdir()):
            raise ValueError("Refusing nonempty output directory without a compatible manifest")
        manifest = {
            **identity,
            "fingerprint": expected_fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
        }
        atomic_json(manifest_path, manifest)
    run_dir = root / "runs"
    run_dir.mkdir(exist_ok=True)
    specs = experiment_specs(manifest)
    expected_names = {artifact_name(**spec) for spec in specs}
    unexpected = {path.name for path in run_dir.glob("*.json")} - expected_names
    if unexpected:
        raise ValueError(f"Unexpected run artifacts in output: {sorted(unexpected)}")

    # Rebuild the derived CSV atomically to recover safely from an interrupted append.
    csv_path = root / "runs.csv"
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".runs.", suffix=".csv")
    pending = []
    completed = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=_csv_fields(config))
            writer.writeheader()
            for spec in specs:
                path = run_dir / artifact_name(**spec)
                if path.exists():
                    writer.writerow(flattened_run(read_artifact(path, manifest, spec)))
                    completed += 1
                else:
                    pending.append(spec)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    stream = progress if progress is not None else sys.stderr

    def checkpoint(status: str) -> None:
        atomic_json(
            root / "checkpoint.json",
            {
                "fingerprint": expected_fingerprint,
                "completed": completed,
                "total": len(specs),
                "status": status,
                "updated_at": datetime.now(UTC).isoformat(),
            },
        )
        print(f"[{completed}/{len(specs)}] {status}", file=stream, flush=True)

    checkpoint("running" if pending else "complete")
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_csv_fields(config))

        def save(result: dict, spec: dict) -> None:
            nonlocal completed
            _check_result(result, manifest, spec)
            result["_experiment"] = {
                "run_fingerprint": _run_fingerprint(manifest, spec),
                "experiment_fingerprint": expected_fingerprint,
                "content_hash": fingerprint(result),
            }
            atomic_json(run_dir / artifact_name(**spec), result)
            writer.writerow(flattened_run(result))
            handle.flush()
            os.fsync(handle.fileno())
            completed += 1
            checkpoint("complete" if completed == len(specs) else "running")

        try:
            if workers == 1:
                for spec in pending:
                    save(_worker(asdict(config), spec), spec)
            elif pending:
                executor = ProcessPoolExecutor(max_workers=workers)
                futures = {}
                remaining = iter(pending)
                try:
                    for _ in range(min(len(pending), workers * 2)):
                        spec = next(remaining)
                        futures[executor.submit(_worker, asdict(config), spec)] = spec
                    while futures:
                        done, _ = wait(futures, return_when=FIRST_COMPLETED)
                        for future in done:
                            spec = futures.pop(future)
                            save(future.result(), spec)
                            next_spec = next(remaining, None)
                            if next_spec is not None:
                                futures[executor.submit(_worker, asdict(config), next_spec)] = (
                                    next_spec
                                )
                finally:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
        except BaseException:
            checkpoint("interrupted")
            raise
    return manifest


def run_pilot(
    output: str | Path,
    config: Config = DEFAULT_CONFIG,
    workers: int = 4,
    replications: int = 2,
    resume: bool = False,
) -> dict:
    """Compare fixed workload candidates, without selecting a favorable strategy or winner.

    Both candidates use the same strategy/layout/scenario/seed design. The study config
    must be frozen manually after inspecting feasibility and failure/service outcomes.
    """
    if replications < 1 or workers < 1:
        raise ValueError("replications and workers must be positive")
    root = Path(output)
    candidates = {
        "v4_t48_i8": replace(config, vehicles=4, total_tasks=48, arrival_interval=8.0),
        "v4_t48_i5": replace(config, vehicles=4, total_tasks=48, arrival_interval=5.0),
    }
    specification = {
        "candidates": {name: asdict(candidate) for name, candidate in candidates.items()},
        "strategies": ["S1", "S2", "S4", "S8"],
        "scenarios": ["D0", "D4"],
        "layouts": ["A", "B"],
        "seeds": list(range(replications)),
        "selection": "Manual feasibility review across all strategies; no automatic selection",
        "source_hash": source_hash(),
        "package_versions": package_versions(),
    }
    manifest_path = root / "pilot.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            old = json.load(handle)
        if not resume or old != specification:
            raise ValueError("Pilot output exists or is incompatible with this pilot specification")
    elif root.exists() and any(root.iterdir()):
        raise ValueError("Refusing nonempty pilot output without a matching pilot.json")
    atomic_json(manifest_path, specification)
    from warehouse_sim.analysis import analyze

    for name, candidate in candidates.items():
        directory = root / name
        run_experiment(
            directory,
            candidate,
            replications,
            workers,
            strategies=specification["strategies"],
            scenarios=specification["scenarios"],
            layouts=specification["layouts"],
            resume=resume,
        )
        analyze(directory)
    with (root / "pilot_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate", *_csv_fields(config)])
        writer.writeheader()
        for name in candidates:
            for result in iter_artifacts(root / name):
                writer.writerow({"candidate": name, **flattened_run(result)})
    return specification
