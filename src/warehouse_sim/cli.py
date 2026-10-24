"""Command-line entry points for individual runs, matched experiments and reporting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse_sim.analysis import analyze
from warehouse_sim.domain import SCENARIOS, STRATEGIES
from warehouse_sim.experiments import load_config, run_experiment, run_pilot, run_single


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warehouse-sim", description="Reproducible warehouse task-allocation experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Execute one simulation and save its full JSON records")
    run.add_argument("--strategy", choices=STRATEGIES, default="S1")
    run.add_argument("--scenario", choices=SCENARIOS, default="D0")
    run.add_argument("--layout", choices=("A", "B"), default="A")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--trace", action="store_true", help="Include per-tick trace records")
    run.add_argument("--config", type=Path, help="JSON object containing Config fields")

    experiment = commands.add_parser(
        "experiment", help="Run a matched-seed factorial matrix, then generate its report"
    )
    experiment.add_argument("--output", type=Path, required=True)
    experiment.add_argument("--replications", type=_positive, default=30)
    experiment.add_argument("--workers", type=_positive, default=4)
    experiment.add_argument("--config", type=Path)
    experiment.add_argument("--strategies", nargs="+", choices=STRATEGIES)
    experiment.add_argument("--scenarios", nargs="+", choices=SCENARIOS)
    experiment.add_argument("--layouts", nargs="+", choices=("A", "B"))
    experiment.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only checkpoints matching config, design, source and package versions",
    )

    analysis = commands.add_parser("analyze", help="Recompute summary, paired changes and plots")
    analysis.add_argument("--input", type=Path, required=True)

    pilot = commands.add_parser(
        "pilot",
        help="Compare fixed workload candidates; manually freeze the study config afterward",
    )
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument(
        "--config", type=Path, help="Base config for shared horizon/planning settings"
    )
    pilot.add_argument("--workers", type=_positive, default=4)
    pilot.add_argument("--replications", type=_positive, default=2)
    pilot.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            result = analyze(args.input)
        elif args.command == "run":
            result = run_single(
                config=load_config(args.config),
                strategy=args.strategy,
                scenario=args.scenario,
                layout=args.layout,
                seed=args.seed,
                trace=args.trace,
                output=args.output,
            )
            result = {"output": str(args.output), "metrics": result["metrics"]}
        elif args.command == "experiment":
            run_experiment(
                output=args.output,
                config=load_config(args.config),
                replications=args.replications,
                workers=args.workers,
                strategies=args.strategies,
                scenarios=args.scenarios,
                layouts=args.layouts,
                resume=args.resume,
            )
            result = analyze(args.output)
        else:
            result = run_pilot(
                output=args.output,
                config=load_config(args.config),
                workers=args.workers,
                replications=args.replications,
                resume=args.resume,
            )
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as error:
        parser.exit(1, f"warehouse-sim: {error}\n")
    except KeyboardInterrupt:
        parser.exit(130, "warehouse-sim: interrupted; saved run checkpoints can be resumed\n")


if __name__ == "__main__":
    raise SystemExit(main())
