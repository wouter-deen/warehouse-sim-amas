"""Two explicit layouts; unique output slots avoid artificial terminal-station deadlocks."""

import random

from warehouse_sim.domain import Config, Layout, TaskSpec


def get_layout(name: str) -> Layout:
    if name not in ("A", "B"):
        raise ValueError(f"Unknown layout: {name}")
    width, height = 17, 13
    if name == "A":
        racks = {(x, y) for x in (3, 4, 7, 8, 11, 12) for y in range(2, 8) if y != 4}
    else:
        racks = {(x, y) for y in (2, 3, 6, 7) for x in range(2, 15) if x not in (5, 11)}
    cells = frozenset((x, y) for x in range(width) for y in range(height) if (x, y) not in racks)
    pickups = tuple(
        sorted(
            c
            for c in cells
            if 1 <= c[1] <= 7 and any(abs(c[0] - r[0]) + abs(c[1] - r[1]) == 1 for r in racks)
        )
    )
    dropoffs = tuple(sorted(c for c in cells if c[1] >= 9))
    parking = tuple((x, 0) for x in (0, 1, 2, 6, 9, 14, 15, 16))
    # Obstructions are aisle events, not service-station failures.
    disruptions = tuple(sorted(c for c in cells if 1 <= c[1] <= 8 and c not in pickups))
    return Layout(name, width, height, cells, pickups, dropoffs, parking, disruptions)


def workload(
    layout: Layout, config: Config, seed: int
) -> tuple[list[TaskSpec], dict[int, tuple[int, int]]]:
    """Independent named RNG streams: changing disruptions cannot change customer demand."""
    if config.total_tasks > len(layout.dropoffs):
        raise ValueError(f"Layout {layout.id} has only {len(layout.dropoffs)} unique output slots")
    if config.vehicles > len(layout.parking) or config.vehicles < layout.zones:
        raise ValueError("Fleet must cover both zones and fit the eight parking cells")
    rng = random.Random(f"workload:{seed}")
    destinations = list(layout.dropoffs)
    rng.shuffle(destinations)
    clock = 0.0
    tasks = []
    for task_id in range(config.total_tasks):
        if task_id:
            clock += rng.expovariate(1 / config.arrival_interval)
        tasks.append(
            TaskSpec(task_id, int(clock), rng.choice(layout.pickups), destinations[task_id])
        )
    starts = {}
    for zone in range(layout.zones):
        candidates = [c for c in layout.parking if layout.zone(c) == zone]
        rng.shuffle(candidates)
        for index, vehicle in enumerate(range(zone, config.vehicles, layout.zones)):
            starts[vehicle] = candidates[index]
    return tasks, starts
