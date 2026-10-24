"""Collision-safe reservations and full-chain space-time A*.

``ReservationTable.search`` plans at most ``horizon`` ticks after ``start_tick``.
Its result finishes at the last operation's cell (or the start for no operations),
where it must be safe to stay indefinitely. Search does not change reservations;
``commit`` / ``replace_many`` revalidate against the latest schedules.

For coordinated replanning, clone the table, stage all affected vehicles with
``hold``, and plan/commit their replacements on the clone in priority order.
Publish the resulting plans with one ``replace_many`` on the original table.
Staged holds deliberately bypass collision checking, so a clone is provisional
until a successful atomic validation includes *all* its plans.
"""

from heapq import heappop, heappush
from itertools import combinations, count
from typing import Literal

from warehouse_sim.domain import Cell, Layout, Operation, Plan, RouteStep

Event = tuple[int, str]
Action = Literal["move", "wait", "handle"]
State = tuple[Cell, int, int, int]  # cell, tick, next operation, active handling remaining


def _integer(value: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _distance(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _position(plan: Plan, tick: int) -> Cell:
    # Before a rebased plan's start, its start cell is held as well.
    if tick <= plan.start_tick:
        return plan.start
    if tick >= plan.end_tick:
        return plan.end
    return plan.steps[tick - plan.start_tick - 1].cell


class ReservationTable:
    """One immutable authoritative Plan per vehicle, with indefinite terminal holds.

    ``plans`` is public for inspection; use transactional methods for publication.
    ``position`` also supports historical ticks, which are retained when replanning.
    Invalid/conflicting commits return False without changing any reservation;
    invalid search arguments or staging holds raise ValueError.
    """

    def __init__(self, layout: Layout, initial_positions: dict[int, Cell]):
        if any(cell not in layout.cells for cell in initial_positions.values()):
            raise ValueError("Initial positions must be traversable layout cells")
        if len(set(initial_positions.values())) != len(initial_positions):
            raise ValueError("Initial positions must be distinct")
        self.layout = layout
        self.plans = {vehicle: Plan(cell, 0, ()) for vehicle, cell in initial_positions.items()}

    def clone(self) -> "ReservationTable":
        """Copy the mutable store; immutable layout/plans can be shared safely."""
        other = object.__new__(type(self))
        other.layout = self.layout
        other.plans = self.plans.copy()
        return other

    def position(self, vehicle: int, tick: int) -> Cell:
        if not _integer(tick):
            raise ValueError("tick must be a nonnegative integer")
        return _position(self.plans[vehicle], tick)

    def available(self, vehicle: int) -> tuple[Cell, int]:
        plan = self.plans[vehicle]
        return plan.end, plan.end_tick

    def can_step(self, vehicle: int, source: Cell, target: Cell, arrival_tick: int) -> bool:
        """Check a unit move/wait, ignoring this vehicle's own reservations."""
        if (
            not _integer(arrival_tick)
            or arrival_tick == 0
            or source not in self.layout.cells
            or target not in self.layout.cells
            or _distance(source, target) > 1
        ):
            return False
        for other, plan in self.plans.items():
            if other == vehicle:
                continue
            before = _position(plan, arrival_tick - 1)
            after = _position(plan, arrival_tick)
            if before == source or after == target or (before == target and after == source):
                return False
        return True

    def _terminal_safe(self, vehicle: int, cell: Cell, tick: int) -> bool:
        for other, plan in self.plans.items():
            if other == vehicle:
                continue
            if _position(plan, tick) == cell:
                return False
            if any(step.tick > tick and step.cell == cell for step in plan.steps):
                return False
        return True

    def _valid_plan(self, plan: Plan) -> bool:
        if plan.start not in self.layout.cells or not _integer(plan.start_tick):
            return False
        cell, tick = plan.start, plan.start_tick
        for step in plan.steps:
            if not _integer(step.tick) or step.tick != tick + 1:
                return False
            if step.cell not in self.layout.cells:
                return False
            if step.action == "move":
                if _distance(cell, step.cell) != 1:
                    return False
            elif step.action in ("wait", "handle"):
                if cell != step.cell:
                    return False
            else:
                return False
            cell, tick = step.cell, step.tick
        return True

    def _combine(self, vehicle: int, plan: Plan, now: int) -> Plan | None:
        if not self._valid_plan(plan) or plan.start_tick < now:
            return None
        old = self.plans[vehicle]
        # Future updates are queue extensions, not edits to unexecuted queue legs.
        if plan.start_tick != now and plan.start_tick < old.end_tick:
            return None
        boundary = plan.start_tick
        if plan.start != _position(old, boundary):
            return None
        if boundary < old.start_tick:
            return None
        prefix = [step for step in old.steps if step.tick <= boundary]
        last_tick = prefix[-1].tick if prefix else old.start_tick
        prefix.extend(RouteStep(t, plan.start, "wait") for t in range(last_tick + 1, boundary + 1))
        initial_events = old.initial_events
        if plan.initial_events:
            if prefix:
                last = prefix[-1]
                prefix[-1] = RouteStep(
                    last.tick, last.cell, last.action, last.completed + plan.initial_events
                )
            else:
                initial_events += plan.initial_events
        return Plan(old.start, old.start_tick, tuple(prefix) + plan.steps, initial_events)

    @staticmethod
    def _compatible(a: Plan, b: Plan, now: int) -> bool:
        if _position(a, now) == _position(b, now):
            return False
        # Positions change only at explicit step ticks. Checking those ticks also
        # covers arbitrary-length terminal holds without scanning an infinite tail.
        for plan in (a, b):
            for step in plan.steps:
                t = step.tick
                if t <= now:
                    continue
                a_after, b_after = _position(a, t), _position(b, t)
                if a_after == b_after:
                    return False
                if _position(a, t - 1) == b_after and _position(b, t - 1) == a_after:
                    return False
        return True

    def commit(self, vehicle: int, plan: Plan, now: int) -> bool:
        """Append at/after availability, or replace the future starting at ``now``.

        The new start must match the existing position. History and events through
        the join tick are retained, and new initial events are merged at that tick.
        Any gap before a queued extension is explicitly reserved as waiting.
        """
        if not _integer(now) or vehicle not in self.plans or plan.start_tick < now:
            return False
        combined = self._combine(vehicle, plan, now)
        if combined is None or any(
            not self._compatible(combined, other_plan, now)
            for other, other_plan in self.plans.items()
            if other != vehicle
        ):
            return False
        # A staging clone may still have conflicts between *other* vehicles.
        # This update validates only the changed vehicle; replace_many performs
        # the global validation before a coordinated replan can be published.
        self.plans = {**self.plans, vehicle: combined}
        return True

    def _replacement(self, vehicle: int, plan: Plan, now: int) -> Plan | None:
        """Accept either a new suffix or a clone's history-preserving full plan."""
        if not self._valid_plan(plan):
            return None
        if plan == self.plans[vehicle]:
            return plan
        prefix = self._combine(vehicle, Plan(self.position(vehicle, now), now, ()), now)
        if prefix is not None and (plan.start, plan.start_tick) == (
            prefix.start,
            prefix.start_tick,
        ):
            length = len(prefix.steps)
            if not length:
                if plan.initial_events[: len(prefix.initial_events)] == prefix.initial_events:
                    return plan
            elif len(plan.steps) >= length and plan.initial_events == prefix.initial_events:
                last, expected = plan.steps[length - 1], prefix.steps[-1]
                if (
                    plan.steps[: length - 1] == prefix.steps[:-1]
                    and (last.tick, last.cell, last.action)
                    == (expected.tick, expected.cell, expected.action)
                    and last.completed[: len(expected.completed)] == expected.completed
                ):
                    return plan
        return self._combine(vehicle, plan, now)

    def replace_many(self, replacements: dict[int, Plan], now: int) -> bool:
        """Atomically validate/publish replacements, including unchanged held plans.

        Accepts new suffixes as in commit, or complete authoritative plans from a
        clone, provided their executed history matches. Passing ``clone.plans``
        therefore works at nonzero ticks without duplicating completion events.
        """
        if not _integer(now) or any(vehicle not in self.plans for vehicle in replacements):
            return False
        candidates = self.plans.copy()
        for vehicle, plan in replacements.items():
            combined = self._replacement(vehicle, plan, now)
            if combined is None:
                return False
            candidates[vehicle] = combined
        if not all(self._valid_plan(plan) for plan in candidates.values()):
            return False
        if any(not self._compatible(a, b, now) for a, b in combinations(candidates.values(), 2)):
            return False
        self.plans = candidates
        return True

    def hold(self, vehicle: int, cell: Cell, now: int) -> None:
        """Stage a stationary future on a clone, preserving history through ``now``.

        The cell must be the vehicle's actual position at now. Other vehicles are
        intentionally NOT checked: stage all affected holds before routing any of
        them. This method is not an atomic, publishable reservation operation.
        """
        if not _integer(now) or cell not in self.layout.cells:
            raise ValueError("Invalid hold cell or tick")
        plan = self._combine(vehicle, Plan(cell, now, ()), now)
        if plan is None:
            raise ValueError("A staged hold must match the vehicle's position at now")
        self.plans[vehicle] = plan

    def search(
        self,
        vehicle: int,
        start: Cell,
        start_tick: int,
        operations: tuple[Operation, ...],
        known_blocked: set[Cell],
        horizon: int,
    ) -> Plan | None:
        """Minimum-tick full-chain A*, with four-neighbor moves, waits and handling.

        Handling is consecutive once begun, but may be deferred before starting.
        Zero-duration services emit initial_events or step.completed upon reaching
        their cell, except the final service is deferred until its terminal hold
        is safe. Final completion always coincides with the route's end tick;
        no actions may follow it. Blocked cells cannot be entered or
        used for waits/handling, but a blocked starting cell may be exited.
        All actions, including waits and handling, use the same reservations.
        """
        if not _integer(start_tick) or not _integer(horizon) or start not in self.layout.cells:
            raise ValueError("Invalid search start or horizon")
        for operation in operations:
            if (
                not _integer(operation.duration)
                or operation.kind not in ("pickup", "dropoff")
                or operation.cell not in self.layout.cells
            ):
                raise ValueError("Invalid operation cell, kind or handling duration")
        if any(
            _position(plan, start_tick) == start
            for other, plan in self.plans.items()
            if other != vehicle
        ):
            return None
        goal = operations[-1].cell if operations else start
        if goal in known_blocked or any(
            plan.end == goal for other, plan in self.plans.items() if other != vehicle
        ):
            return None

        size = len(operations)
        # suffix[i] includes all service and inter-operation travel from op i on.
        suffix = [0] * (size + 1)
        for i in range(size - 1, -1, -1):
            suffix[i] = operations[i].duration + suffix[i + 1]
            if i + 1 < size:
                suffix[i] += _distance(operations[i].cell, operations[i + 1].cell)

        def heuristic(state: State) -> int:
            cell, _, index, remaining = state
            if index == size:
                return _distance(cell, goal)
            if remaining:
                return remaining + suffix[index] - operations[index].duration
            return _distance(cell, operations[index].cell) + suffix[index]

        def zero_events(cell: Cell, index: int, tick: int) -> tuple[int, tuple[Event, ...]]:
            events: list[Event] = []
            while (
                index < size and operations[index].cell == cell and not operations[index].duration
            ):
                if index == size - 1 and not self._terminal_safe(vehicle, cell, tick):
                    break
                op = operations[index]
                events.append((op.task_id, op.kind))
                index += 1
            return index, tuple(events)

        index, initial_events = zero_events(start, 0, start_tick)
        initial: State = (start, start_tick, index, 0)
        deadline = start_tick + horizon
        serial = count()
        heap: list[tuple[int, int, State]] = [
            (start_tick + heuristic(initial), next(serial), initial)
        ]
        parents: dict[State, tuple[State, RouteStep] | None] = {initial: None}
        while heap:
            estimate, _, state = heappop(heap)
            if estimate > deadline:
                break
            cell, tick, index, remaining = state
            if index == size and cell == goal and self._terminal_safe(vehicle, cell, tick):
                steps: list[RouteStep] = []
                cursor = state
                while (parent := parents[cursor]) is not None:
                    cursor, step = parent
                    steps.append(step)
                return Plan(start, start_tick, tuple(reversed(steps)), initial_events)
            if tick == deadline or (operations and index == size):
                continue

            # Active service cannot be interrupted. Before service starts, moving
            # or waiting is a genuine alternative, essential for non-greedy A*.
            actions: list[tuple[Cell, Action, int, int, tuple[Event, ...]]] = []
            if remaining or (
                index < size and cell == operations[index].cell and operations[index].duration > 0
            ):
                op = operations[index]
                left = remaining - 1 if remaining else op.duration - 1
                next_index = index if left else index + 1
                events = () if left else ((op.task_id, op.kind),)
                actions.append((cell, "handle", next_index, left, events))
            if not remaining:
                actions.extend(
                    (target, "move" if target != cell else "wait", index, 0, ())
                    for target in (*self.layout.neighbors(cell), cell)
                )
            for target, action, next_index, left, events in actions:
                if target in known_blocked or not self.can_step(vehicle, cell, target, tick + 1):
                    continue
                if not left:
                    next_index, zero = zero_events(target, next_index, tick + 1)
                    events += zero
                if (
                    operations
                    and next_index == size
                    and not self._terminal_safe(vehicle, target, tick + 1)
                ):
                    # Finishing service is a terminal transition, not permission
                    # to move away and return after a known future conflict.
                    continue
                successor: State = (target, tick + 1, next_index, left)
                if successor in parents or tick + 1 + heuristic(successor) > deadline:
                    continue
                parents[successor] = (
                    state,
                    RouteStep(tick + 1, target, action, events),
                )
                heappush(heap, (tick + 1 + heuristic(successor), next(serial), successor))
        return None
