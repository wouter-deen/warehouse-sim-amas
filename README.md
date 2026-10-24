# Warehouse agents simulation

A Mesa-based implementation of our task-allocation and route-disruption study.

## Code map

| Module                     | Responsibility                                                                     |
| -------------------------- | ---------------------------------------------------------------------------------- |
| `domain.py`, `layouts.py`  | Experimental tables, configuration, layouts, seeded workloads                      |
| `model.py`                 | Mesa model, environment, tick phases, physical safety, metrics                     |
| `agents.py`, `messages.py` | Mesa agents, local bids, contract-net auctions, recovery, message bus              |
| `routing.py`               | Full-chain space–time A*, tentative estimates, atomic reservations, terminal holds |
| `experiments.py`           | Pilot, parallel factorial runner, provenance, checkpoints                          |
| `analysis.py`              | Student-t CIs, failure-aware summaries, paired robustness, report/plots            |

## Setup

Python 3.13 and uv are recommended:

```sh
uv sync --extra dev
```

`uv.lock` pins the actual dependency versions. Mesa supplies the model, all agent classes, AgentSet activation, and data collection. Routing and protocol logic are implemented explicitly to reproduce the study's deterministic execution rules.

## Run one simulation

```sh
uv run warehouse-sim run --strategy S8 --scenario D4 --layout B --seed 0 --config configs/study.json --trace --output results/example.json
```

Results are saved to `results/example.json`. The optional `--trace` flag includes per-tick simulation states for inspection.

## Reproduce the experiments

```sh
# Small feasibility pilot: two workloads, two seeds, representative strategy families.
uv run warehouse-sim pilot --output results/pilot --workers 4 --replications 2

# The full paper design; checkpoints are saved after every completed run.
uv run warehouse-sim experiment --config configs/study.json --output results/full --replications 30 --workers 4

# Continue an interrupted experiment with exactly the same source, config, and dependencies.
uv run warehouse-sim experiment --config configs/study.json --output results/full --replications 30 --workers 4 --resume

# Rebuild statistics, tables, and figures without rerunning Mesa.
uv run warehouse-sim analyze --input results/full
```

For a quick subset, add e.g. `--strategies S1 S4 S8 --scenarios D0 D4 --layouts A --replications 2` to `experiment`, using a **different output directory**. Include D0 for meaningful paired robustness. The runner refuses to reuse incompatible or edited checkpoints. Python exceptions interrupt a study, they are not relabeled as measured execution failures.

## Results

- `results/full/summary.csv`: means, 95% Student-t CIs, and missing/sample counts for every metric.
- `results/full/paired_degradation.csv`: own-baseline matched-seed changes.
- `results/full/paired_summary.csv`: paired means and CIs, with zero-baseline absolute changes kept separate.
- `results/full/runs.csv`: one flattened row per run.
- `results/full/runs/*.json`: raw task/vehicle timestamps, metrics, disruption events, and message counts.
- `results/full/manifest.json`: experiment/config/source fingerprints, seeds, package versions, and design.
- `results/full/plots/`: service/failures, SOC, utilization/communication, and robustness figures.

Completion time, throughput, unfinished tasks, AGV utilization, SOC, communication volume, and relative degradation are the values defined in the report. Secondary diagnostics include pickup delay, makespan, empty travel, load balance, prolonged stalls, disruption exposure, and replan counts. Failures remain in service/cost summaries. Missing completion-time observations are not replaced by zero.

## Modeling remarks

- Obstruction reporters remain local monitors until they observe clearance
- Occupied-cell event proposals are skipped and logged
- All strategies use the same bounded prioritized space–time planner

## Validation

```sh
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
```

Tests cover all strategies/layouts, forced disruptions, local sensing and clearance, queues and handling, exact metric accounting, deterministic matched scenarios, reservation atomicity and collisions, small-search optimality against BFS, paired statistics, and checkpoint/resume safety.
