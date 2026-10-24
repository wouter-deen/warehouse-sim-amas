"""Shared immutable experiment and routing contracts (times are integer ticks)."""

from dataclasses import dataclass, field
from typing import Literal

Cell = tuple[int, int]


@dataclass(frozen=True)
class Strategy:
    id: str
    scope: Literal["global", "zone"]
    utility: Literal["route", "availability"]
    commitment: Literal["idle", "queue", "bundle"]


STRATEGIES = {
    s.id: s
    for s in (
        Strategy("S1", "global", "route", "idle"),
        Strategy("S2", "zone", "route", "idle"),
        Strategy("S3", "global", "route", "queue"),
        Strategy("S4", "global", "availability", "queue"),
        Strategy("S5", "zone", "route", "queue"),
        Strategy("S6", "zone", "availability", "queue"),
        Strategy("S7", "global", "route", "bundle"),
        Strategy("S8", "global", "availability", "bundle"),
        Strategy("S9", "zone", "route", "bundle"),
        Strategy("S10", "zone", "availability", "bundle"),
    )
}


@dataclass(frozen=True)
class DisruptionScenario:
    id: str
    frequency: float
    duration: int


SCENARIOS = {
    s.id: s
    for s in (
        DisruptionScenario("D0", 0, 0),
        DisruptionScenario("D1", 2, 10),
        DisruptionScenario("D2", 2, 50),
        DisruptionScenario("D3", 10, 10),
        DisruptionScenario("D4", 10, 50),
    )
}


@dataclass(frozen=True)
class Config:
    vehicles: int = 4
    total_tasks: int = 48
    arrival_interval: float = 8.0
    max_ticks: int = 1600
    planning_horizon: int = 192
    queue_max: int = 3
    pickup_ticks: int = 1
    dropoff_ticks: int = 1
    auction_interval: int = 4
    bundle_batch: int = 3
    deadlock_window: int = 100

    def __post_init__(self):
        for name in (
            "vehicles",
            "total_tasks",
            "max_ticks",
            "planning_horizon",
            "queue_max",
            "auction_interval",
            "bundle_batch",
            "deadlock_window",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.arrival_interval <= 0 or self.pickup_ticks < 0 or self.dropoff_ticks < 0:
            raise ValueError("Invalid arrival interval or handling duration")
        if self.queue_max != 3:
            raise ValueError("The paper fixes maximum queue length at 3")
        if self.bundle_batch > 3:
            raise ValueError("Bundle batch is bounded at 3 for exhaustive small-batch coverage")


@dataclass(frozen=True)
class Layout:
    id: str
    width: int
    height: int
    cells: frozenset[Cell]
    pickups: tuple[Cell, ...]
    dropoffs: tuple[Cell, ...]
    parking: tuple[Cell, ...]
    disruption_cells: tuple[Cell, ...]
    zones: int = 2

    def zone(self, cell: Cell) -> int:
        return min(self.zones - 1, cell[0] * self.zones // self.width)

    def neighbors(self, cell: Cell) -> tuple[Cell, ...]:
        x, y = cell
        return tuple(p for p in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)) if p in self.cells)


@dataclass(frozen=True)
class TaskSpec:
    id: int
    created: int
    pickup: Cell
    dropoff: Cell


@dataclass
class TaskRecord:
    spec: TaskSpec
    assigned_to: int | None = None
    assigned_tick: int | None = None
    pickup_tick: int | None = None
    dropoff_tick: int | None = None


@dataclass(frozen=True)
class Operation:
    """One required service operation; zero-duration operations still emit a completion event."""

    task_id: int
    kind: Literal["pickup", "dropoff"]
    cell: Cell
    duration: int


@dataclass(frozen=True)
class RouteStep:
    """Position at tick `tick`, following the action over [tick-1, tick)."""

    tick: int
    cell: Cell
    action: Literal["move", "wait", "handle"]
    completed: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class Plan:
    start: Cell
    start_tick: int
    steps: tuple[RouteStep, ...]
    initial_events: tuple[tuple[int, str], ...] = ()

    @property
    def end(self) -> Cell:
        return self.steps[-1].cell if self.steps else self.start

    @property
    def end_tick(self) -> int:
        return self.steps[-1].tick if self.steps else self.start_tick

    @property
    def cost(self) -> int:
        return self.end_tick - self.start_tick


@dataclass
class VehicleState:
    id: int
    cell: Cell
    zone: int
    queue: list[int] = field(default_factory=list)
    carrying: int | None = None
    moves: int = 0
    empty_moves: int = 0
    waits: int = 0
    handles: int = 0
    busy: int = 0
    idle: int = 0
    completed: int = 0
    schedule_version: int = 0


def operations_for(tasks: list[TaskSpec], config: Config) -> tuple[Operation, ...]:
    return tuple(
        op
        for task in tasks
        for op in (
            Operation(task.id, "pickup", task.pickup, config.pickup_ticks),
            Operation(task.id, "dropoff", task.dropoff, config.dropoff_ticks),
        )
    )
