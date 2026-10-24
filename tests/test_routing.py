"""Reservation safety and optimal full-phase routing, including a small BFS oracle."""

from collections import deque
from random import Random
from typing import Any, Literal, cast

import pytest

from warehouse_sim.domain import Layout, Operation, Plan, RouteStep
from warehouse_sim.routing import ReservationTable


def layout(width=4, height=3, cells=None):
    cells = frozenset(
        cells if cells is not None else ((x, y) for x in range(width) for y in range(height))
    )
    return Layout("test", width, height, cells, (), (), (), ())


def path(start, *cells, tick=0):
    steps = []
    previous = start
    for t, cell in enumerate(cells, tick + 1):
        steps.append(RouteStep(t, cell, "wait" if cell == previous else "move"))
        previous = cell
    return Plan(start, tick, tuple(steps))


def op(cell, duration=0, task=1, kind: Literal["pickup", "dropoff"] = "pickup"):
    return Operation(task, kind, cell, duration)


def test_initial_holds_availability_and_own_reservations():
    table = ReservationTable(layout(), {0: (0, 0), 1: (1, 0)})
    assert table.plans[0] == Plan((0, 0), 0, ())
    assert table.available(0) == ((0, 0), 0)
    assert table.position(1, 10_000) == (1, 0)
    assert not table.can_step(0, (0, 0), (1, 0), 1000)
    assert table.can_step(0, (0, 0), (0, 0), 1)
    assert table.commit(0, path((0, 0), (0, 1), (1, 1)), now=0)
    assert table.available(0) == ((1, 1), 2)
    assert table.position(0, 0) == (0, 0)
    assert table.position(0, 1) == (0, 1)
    assert table.position(0, 500) == (1, 1)
    # The old route is never an obstacle to its own vehicle.
    assert table.can_step(0, (0, 0), (0, 1), 1)
    assert not table.can_step(1, (1, 0), (1, 1), 500)


def test_initial_positions_must_be_valid_and_distinct():
    with pytest.raises(ValueError):
        ReservationTable(layout(), {0: (0, 0), 1: (0, 0)})
    with pytest.raises(ValueError):
        ReservationTable(layout(), {0: (99, 0)})


def test_vertex_reverse_edge_and_following_moves():
    table = ReservationTable(layout(), {0: (0, 1), 1: (1, 0)})
    assert table.commit(1, path((1, 0), (0, 0)), now=0)
    assert not table.can_step(0, (0, 1), (0, 0), 1)  # Vertex.
    assert not table.can_step(0, (0, 0), (1, 0), 1)  # Reverse edge.
    assert table.can_step(0, (1, 1), (1, 0), 1)  # Following into a vacated cell.
    assert not table.can_step(0, (1, 0), (2, 0), 1)  # Occupied departure.
    assert not table.can_step(0, (0, 1), (2, 1), 1)
    assert not table.can_step(0, (0, 1), (0, 0), 0)
    assert not table.can_step(0, (0, 1), (-1, 1), 1)


def test_terminal_hold_checks_later_pass_through_and_existing_terminal():
    table = ReservationTable(layout(), {0: (0, 0), 1: (2, 1)})
    assert table.commit(1, path((2, 1), (2, 0), (1, 0), (1, 1)), 0)
    # Arriving is safe, but staying would collide two ticks later.
    assert table.can_step(0, (0, 0), (1, 0), 1)
    before = table.plans.copy()
    assert not table.commit(0, path((0, 0), (1, 0)), 0)
    assert table.plans == before
    assert not table.commit(0, path((0, 0), (0, 1), (1, 1)), 0)
    assert table.commit(0, path((0, 0), (0, 0), (0, 0), (1, 0)), 0)


def test_terminal_hold_rejects_other_vehicle_already_holding_before_plan_start():
    table = ReservationTable(layout(), {0: (0, 0), 1: (1, 0)})
    before = table.plans.copy()
    assert not table.commit(0, path((0, 0), (1, 0), tick=20), now=0)
    assert table.plans == before


def test_append_preserves_prefix_events_and_reserves_gap():
    table = ReservationTable(layout(), {0: (0, 0)})
    original = Plan(
        (0, 0),
        0,
        (
            RouteStep(1, (1, 0), "move"),
            RouteStep(2, (1, 0), "handle", ((1, "pickup"),)),
        ),
        ((0, "pickup"),),
    )
    assert table.commit(0, original, 0)
    extension = Plan(
        (1, 0), 5, (RouteStep(6, (2, 0), "move", ((2, "dropoff"),)),), ((2, "pickup"),)
    )
    assert table.commit(0, extension, now=1)
    combined = table.plans[0]
    assert combined.steps[:2] == original.steps
    assert combined.initial_events == original.initial_events
    assert [step.action for step in combined.steps[2:5]] == ["wait"] * 3
    assert combined.steps[4].completed == ((2, "pickup"),)
    assert combined.steps[5].completed == ((2, "dropoff"),)
    assert table.available(0) == ((2, 0), 6)
    assert all(table.position(0, t) == (1, 0) for t in (1, 2, 3, 4, 5))


def test_zero_step_append_merges_initial_events_at_join():
    table = ReservationTable(layout(), {0: (0, 0)})
    assert table.commit(0, Plan((0, 0), 0, (), ((1, "pickup"),)), 0)
    assert table.commit(0, Plan((0, 0), 0, (), ((1, "dropoff"),)), 0)
    assert table.plans[0].initial_events == ((1, "pickup"), (1, "dropoff"))


def test_replan_at_now_keeps_executed_history_and_discards_old_future():
    table = ReservationTable(layout(), {0: (0, 0)})
    old = path((0, 0), (1, 0), (2, 0), (3, 0))
    assert table.commit(0, old, 0)
    assert table.commit(0, path((1, 0), (1, 1), tick=1), now=1)
    assert table.plans[0].steps == (old.steps[0], RouteStep(2, (1, 1), "move"))
    assert table.position(0, 0) == (0, 0)
    assert table.position(0, 1) == (1, 0)
    assert table.available(0) == ((1, 1), 2)


@pytest.mark.parametrize(
    "bad",
    [
        Plan((0, 0), 0, (RouteStep(2, (1, 0), "move"),)),
        Plan((0, 0), 0, (RouteStep(1, (2, 0), "move"),)),
        Plan((0, 0), 0, (RouteStep(1, (0, 0), "move"),)),
        Plan((0, 0), 0, (RouteStep(1, (1, 0), "wait"),)),
        Plan((0, 0), 0, (RouteStep(1, (1, 0), "handle"),)),
        Plan((0, 0), 0, (RouteStep(1, (0, 0), cast(Any, "fly")),)),
        Plan((0, 0), 0, (RouteStep(1, (-1, 0), "move"),)),
        Plan((0, 0), -1, ()),
        Plan((0, 0), 0, (RouteStep(cast(Any, 1.0), (0, 0), "wait"),)),
        Plan((1, 0), 0, ()),  # Wrong starting position.
    ],
)
def test_invalid_commit_is_atomic(bad):
    table = ReservationTable(layout(), {0: (0, 0)})
    before = table.plans.copy()
    assert not table.commit(0, bad, 0)
    assert table.plans == before


def test_commit_rejects_past_and_middle_of_queue_without_mutation():
    table = ReservationTable(layout(), {0: (0, 0)})
    assert table.commit(0, path((0, 0), (1, 0), (2, 0), (3, 0)), 0)
    before = table.plans.copy()
    assert not table.commit(0, path((0, 0), (0, 1)), now=1)
    assert not table.commit(0, path((1, 0), (1, 1), tick=1), now=0)
    assert not table.commit(99, Plan((0, 0), 0, ()), now=0)
    assert not table.commit(0, Plan((0, 0), 0, ()), now=-1)
    assert table.plans == before


def test_atomic_replace_many_allows_convoy_but_rejects_swap_and_held_conflicts():
    table = ReservationTable(layout(), {0: (0, 0), 1: (1, 0)})
    a = path((0, 0), (1, 0))
    b = path((1, 0), (2, 0))
    assert not table.commit(0, a, 0)
    assert table.replace_many({0: a, 1: b}, 0)
    assert table.available(0) == ((1, 0), 1)
    assert table.available(1) == ((2, 0), 1)

    swapped = ReservationTable(layout(), {0: (0, 0), 1: (1, 0)})
    before = swapped.plans.copy()
    assert not swapped.replace_many({0: a, 1: path((1, 0), (0, 0))}, 0)
    assert swapped.plans == before

    held = ReservationTable(layout(), {0: (0, 0), 1: (1, 0), 2: (2, 0)})
    before = held.plans.copy()
    assert not held.replace_many({0: a, 1: b}, 0)
    assert held.plans == before
    assert not held.replace_many({0: a, 1: Plan((1, 0), 0, ())}, 0)
    assert held.plans == before


def test_clone_staged_holds_and_atomic_replanning():
    table = ReservationTable(layout(), {0: (0, 0), 1: (1, 0)})
    assert table.replace_many({0: path((0, 0), (1, 0)), 1: path((1, 0), (2, 0))}, 0)
    before = table.plans.copy()
    clone = table.clone()
    assert clone.plans is not table.plans
    clone.hold(1, (1, 0), 0)
    # The staged hold conflicts with an as-yet unreplanned vehicle.
    assert not clone.replace_many({}, 0)
    assert table.plans == before
    clone.hold(0, (0, 0), 0)
    assert clone.replace_many({}, 0)
    a = clone.search(0, (0, 0), 0, (op((0, 1)),), set(), 5)
    assert a is not None and clone.commit(0, a, 0)
    b = clone.search(1, (1, 0), 0, (op((0, 0)),), set(), 5)
    assert b is not None and clone.commit(1, b, 0)
    assert table.plans == before
    assert table.replace_many(clone.plans, 0)
    assert table.plans == clone.plans
    with pytest.raises(ValueError):
        clone.hold(0, (3, 2), 0)


def test_hold_preserves_history_at_nonzero_now():
    table = ReservationTable(layout(), {0: (0, 0)})
    assert table.commit(0, path((0, 0), (1, 0), (2, 0)), 0)
    clone = table.clone()
    clone.hold(0, (1, 0), 1)
    assert clone.plans[0].steps == (RouteStep(1, (1, 0), "move"),)
    assert clone.position(0, 99) == (1, 0)
    assert table.position(0, 99) == (2, 0)


def test_handling_durations_and_completion_events_are_reserved():
    table = ReservationTable(layout(), {0: (0, 0), 1: (0, 2)})
    operations = (op((0, 0), 2), op((2, 0), 3, kind="dropoff"))
    result = table.search(0, (0, 0), 0, operations, set(), 7)
    assert result is not None
    assert result.cost == 7
    assert [step.action for step in result.steps] == [
        "handle",
        "handle",
        "move",
        "move",
        "handle",
        "handle",
        "handle",
    ]
    assert result.steps[1].completed == ((1, "pickup"),)
    assert result.steps[-1].completed == ((1, "dropoff"),)
    assert table.commit(0, result, 0)
    assert not table.can_step(1, (1, 0), (0, 0), 1)
    assert not table.can_step(1, (2, 1), (2, 0), 6)
    assert table.search(0, (0, 0), 0, operations, set(), 6) is None


def test_reservations_force_wait_and_increase_cost():
    table = ReservationTable(layout(3, 2), {0: (0, 0), 1: (1, 1)})
    assert table.commit(1, path((1, 1), (1, 0), (1, 1)), 0)
    result = table.search(0, (0, 0), 0, (op((2, 0)),), set(), 3)
    assert result is not None and result.cost == 3
    assert result.steps[0].action == "wait"
    assert result.steps[1].cell == (1, 0)
    assert table.search(0, (0, 0), 0, (op((2, 0)),), set(), 2) is None
    assert table.commit(0, result, 0)
    assert not table.can_step(1, (0, 1), (0, 0), 1)


def test_search_accounts_for_future_terminal_conflicts_not_just_arrival():
    table = ReservationTable(layout(3, 2), {0: (0, 0), 1: (2, 1)})
    assert table.commit(1, path((2, 1), (2, 0), (2, 0), (1, 0), (1, 1)), 0)
    before = table.plans.copy()
    assert table.search(0, (0, 0), 0, (op((1, 0)),), set(), 3) is None
    result = table.search(0, (0, 0), 0, (op((1, 0)),), set(), 5)
    assert result is not None and result.cost == 4
    assert result.end == (1, 0)
    assert table.plans == before  # Search is speculative.
    assert table.commit(0, result, 0)
    assert table.search(0, (0, 0), 0, (op((1, 1)),), set(), 20) is None


def test_blocked_cells_detour_unreachable_and_blocked_start_exit():
    table = ReservationTable(layout(3, 2), {0: (0, 0)})
    result = table.search(0, (0, 0), 0, (op((2, 0)),), {(1, 0)}, 4)
    assert result is not None and result.cost == 4
    assert all(step.cell != (1, 0) for step in result.steps)
    assert table.search(0, (0, 0), 0, (op((2, 0)),), {(1, 0)}, 3) is None
    assert table.search(0, (0, 0), 0, (op((2, 0)),), {(1, 0), (1, 1)}, 8) is None
    result = table.search(0, (0, 0), 0, (op((1, 0)),), {(0, 0)}, 1)
    assert result is not None and result.cost == 1
    assert table.search(0, (0, 0), 0, (op((1, 0)),), {(1, 0)}, 10) is None


def test_zero_handling_initial_move_and_handle_completion_chains():
    table = ReservationTable(layout(), {0: (0, 0)})
    operations = (
        op((0, 0), 0, 1),
        op((0, 0), 0, 1, "dropoff"),
        op((0, 0), 1, 2),
        op((0, 0), 0, 2, "dropoff"),
        op((1, 0), 0, 3),
        op((1, 0), 0, 3, "dropoff"),
    )
    result = table.search(0, (0, 0), 0, operations, set(), 2)
    assert result is not None and result.cost == 2
    assert result.initial_events == ((1, "pickup"), (1, "dropoff"))
    assert result.steps[0].action == "handle"
    assert result.steps[0].completed == ((2, "pickup"), (2, "dropoff"))
    assert result.steps[1].action == "move"
    assert result.steps[1].completed == ((3, "pickup"), (3, "dropoff"))
    assert table.commit(0, result, 0)


def test_empty_and_all_zero_operations_with_zero_horizon():
    table = ReservationTable(layout(), {0: (0, 0)})
    assert table.search(0, (0, 0), 7, (), set(), 0) == Plan((0, 0), 7, ())
    result = table.search(0, (0, 0), 7, (op((0, 0)),), set(), 0)
    assert result == Plan((0, 0), 7, (), ((1, "pickup"),))
    assert table.search(0, (0, 0), 7, (op((1, 0)),), set(), 0) is None


def test_search_at_availability_uses_relative_horizon_and_ignores_own_plan():
    table = ReservationTable(layout(), {0: (0, 0)})
    assert table.commit(0, path((0, 0), (1, 0), (2, 0)), 0)
    cell, tick = table.available(0)
    result = table.search(0, cell, tick, (op((3, 0), 1),), set(), 2)
    assert result is not None and result.start_tick == 2 and result.end_tick == 4
    assert table.commit(0, result, now=1)
    assert len(table.plans[0].steps) == 4


@pytest.mark.parametrize("duration", [-1, 0.5, True])
def test_invalid_handling_durations_raise(duration):
    table = ReservationTable(layout(), {0: (0, 0)})
    with pytest.raises(ValueError, match="operation"):
        table.search(0, (0, 0), 0, (op((1, 0), duration),), set(), 5)


@pytest.mark.parametrize("operations", [(op((99, 0)),), (op((0, 0), kind=cast(Any, "invalid")),)])
def test_invalid_operation_contracts_raise(operations):
    table = ReservationTable(layout(), {0: (0, 0)})
    with pytest.raises(ValueError, match="operation"):
        table.search(0, (0, 0), 0, operations, set(), 5)


def test_full_phase_search_does_not_commit_to_greedy_first_leg():
    cells = {(0, 0), (1, 0), (1, 1), (2, 0), (3, 0), (3, 1)}
    table = ReservationTable(layout(4, 2, cells), {0: (1, 0), 1: (3, 1), 2: (1, 1)})
    assert table.replace_many(
        {
            0: path((1, 0), (0, 0)),
            1: path((3, 1), (3, 0), (2, 0), (3, 0), (3, 1)),
            2: path((1, 1), (1, 1), (1, 0), (1, 1)),
        },
        0,
    )
    # The shortest first leg reaches (2, 0) at t=1, but every exit at t=2
    # is a vertex conflict or reverse swap. Only delaying that leg can work.
    assert table.can_step(0, (1, 0), (2, 0), 1)
    assert all(not table.can_step(0, (2, 0), cell, 2) for cell in ((1, 0), (2, 0), (3, 0)))
    operations = (op((2, 0)), op((3, 0), task=2))
    result = table.search(0, (1, 0), 0, operations, set(), 7)
    assert result is not None and result.cost == 5
    assert result.steps[0].cell != (2, 0)
    assert [event for step in result.steps for event in step.completed] == [
        (1, "pickup"),
        (2, "pickup"),
    ]
    assert table.commit(0, result, 0)


def brute_force_cost(table, start, start_tick, operations, blocked, horizon):
    """Uninformed exhaustive BFS with elapsed service rather than A*'s remainder."""
    goal = operations[-1].cell if operations else start
    others = [vehicle for vehicle in table.plans if vehicle != 0]

    def terminal_safe(cell, tick):
        last_known_tick = max([tick, *(table.plans[other].end_tick for other in others)])
        return all(
            table.position(other, t) != cell
            for other in others
            for t in range(tick, last_known_tick + 1)
        )

    def finish_zero(cell, done, tick):
        while (
            done < len(operations)
            and operations[done].cell == cell
            and operations[done].duration == 0
        ):
            if done == len(operations) - 1 and not terminal_safe(cell, tick):
                break
            done += 1
        return done

    initial = (start, start_tick, finish_zero(start, 0, start_tick), 0)
    queue = deque([initial])
    seen = {initial}
    while queue:
        cell, tick, done, worked = queue.popleft()
        if any(table.position(other, tick) == cell for other in others):
            continue
        if done == len(operations):
            if cell == goal and cell not in blocked and terminal_safe(cell, tick):
                return tick - start_tick
            if operations:
                # A completed final service cannot be followed by any action.
                continue
        if tick == start_tick + horizon:
            continue
        choices = []
        if not worked:
            choices.extend((p, done, 0) for p in (*table.layout.neighbors(cell), cell))
        if (
            done < len(operations)
            and cell == operations[done].cell
            and operations[done].duration > 0
        ):
            elapsed = worked + 1
            if elapsed == operations[done].duration:
                choices.append((cell, done + 1, 0))
            else:
                choices.append((cell, done, elapsed))
        for target, next_done, elapsed in choices:
            if target in blocked:
                continue
            if any(
                table.position(other, tick + 1) == target
                or (
                    table.position(other, tick) == target
                    and table.position(other, tick + 1) == cell
                )
                for other in others
            ):
                continue
            if not elapsed:
                next_done = finish_zero(target, next_done, tick + 1)
            state = (target, tick + 1, next_done, elapsed)
            if state not in seen:
                seen.add(state)
                queue.append(state)
    return None


def test_multi_operation_astar_matches_small_exhaustive_brute_force():
    rng = Random(38291)
    grid = layout(3, 2)
    cells = sorted(grid.cells)
    successes = failures = 0
    for case in range(120):
        table = ReservationTable(grid, {0: (0, 0), 1: (2, 1)})
        other_cells = []
        cell = (2, 1)
        for _ in range(rng.randrange(6)):
            cell = rng.choice([p for p in (*grid.neighbors(cell), cell) if p != (0, 0)])
            other_cells.append(cell)
        assert table.commit(1, path((2, 1), *other_cells), 0)
        operations = tuple(
            op(rng.choice(cells), rng.randrange(3), task=i, kind="dropoff" if i % 2 else "pickup")
            for i in range(rng.randrange(1, 4))
        )
        blocked = {rng.choice(cells)} if rng.random() < 0.3 else set()
        horizon = rng.randrange(4, 11)
        expected = brute_force_cost(table, (0, 0), 0, operations, blocked, horizon)
        result = table.search(0, (0, 0), 0, operations, blocked, horizon)
        assert (None if result is None else result.cost) == expected, f"case {case}"
        if result is None:
            failures += 1
        else:
            successes += 1
            assert result.end == operations[-1].cell
            events = list(result.initial_events)
            events.extend(event for step in result.steps for event in step.completed)
            assert events == [(operation.task_id, operation.kind) for operation in operations]
            final_event = (operations[-1].task_id, operations[-1].kind)
            if result.steps:
                assert result.steps[-1].completed[-1] == final_event
                assert final_event not in result.initial_events
                assert all(final_event not in step.completed for step in result.steps[:-1])
            else:
                assert result.initial_events[-1] == final_event
            assert table.commit(0, result, 0), f"case {case} failed commitment"
    assert successes >= 15 and failures >= 15


def test_clone_commit_can_progress_while_other_staged_holds_are_unresolved():
    table = ReservationTable(layout(), {0: (0, 0), 1: (1, 0), 2: (2, 1)})
    original = {
        0: path((0, 0), (0, 1)),
        1: path((1, 0), (2, 0), (3, 0)),
        2: path((2, 1), (1, 1), (1, 0), (2, 0)),
    }
    assert table.replace_many(original, 0)
    clone = table.clone()
    clone.hold(0, (0, 0), 0)
    clone.hold(1, (1, 0), 0)
    assert not clone.replace_many({}, 0)  # Vehicle 2 will visit vehicle 1's hold.
    assert clone.commit(0, original[0], 0)  # Independent first replan can progress.
    assert not clone.replace_many({}, 0)
    assert clone.commit(1, original[1], 0)
    assert table.replace_many(clone.plans, 0)


@pytest.mark.parametrize("now", [0, 1, 5])
def test_publish_clone_full_plans_preserves_history_and_events_once(now):
    table = ReservationTable(layout(), {0: (0, 0), 1: (3, 2)})
    assert table.commit(
        0,
        Plan(
            (0, 0),
            0,
            (
                RouteStep(1, (1, 0), "move", ((9, "pickup"),)),
                RouteStep(2, (2, 0), "move"),
                RouteStep(3, (3, 0), "move"),
            ),
            ((8, "pickup"),),
        ),
        0,
    )
    before = table.plans.copy()
    clone = table.clone()
    cell = table.position(0, now)
    clone.hold(0, cell, now)
    route = clone.search(0, cell, now, (op(cell, task=10), op((cell[0], 1), task=11)), set(), 3)
    assert route is not None and clone.commit(0, route, now)
    assert table.plans == before
    assert table.replace_many(clone.plans, now)
    assert table.plans == clone.plans
    events = list(table.plans[0].initial_events)
    events.extend(event for step in table.plans[0].steps for event in step.completed)
    assert events.count((8, "pickup")) == 1
    assert events.count((9, "pickup")) == int(now >= 1)
    assert events.count((10, "pickup")) == events.count((11, "pickup")) == 1
    # A repeat publication of authoritative plans is idempotent.
    assert table.replace_many(clone.plans, now)
    assert table.plans == clone.plans


def test_publish_full_plan_cannot_rewrite_executed_history():
    table = ReservationTable(layout(), {0: (0, 0)})
    original = Plan(
        (0, 0),
        0,
        (
            RouteStep(1, (1, 0), "move", ((1, "pickup"),)),
            RouteStep(2, (2, 0), "move"),
        ),
    )
    assert table.commit(0, original, 0)
    before = table.plans.copy()
    changed_events = Plan(
        (0, 0),
        0,
        (
            RouteStep(1, (1, 0), "move"),
            RouteStep(2, (1, 1), "move"),
        ),
    )
    assert not table.replace_many({0: changed_events}, 1)
    changed_position = path((0, 0), (0, 1), (1, 1))
    assert not table.replace_many({0: changed_position}, 1)
    assert table.plans == before
    assert not table.commit(0, original, now=1)  # commit accepts only new suffixes.


def test_search_result_is_revalidated_against_latest_future_schedule():
    table = ReservationTable(layout(), {0: (0, 0), 1: (2, 1)})
    result = table.search(0, (0, 0), 0, (op((1, 0)),), set(), 4)
    assert result is not None and result.cost == 1
    assert table.commit(1, path((2, 1), (2, 0), (1, 0), (1, 1)), 0)
    before = table.plans.copy()
    assert not table.commit(0, result, 0)
    assert table.plans == before


def test_service_can_be_deferred_but_not_interrupted():
    table = ReservationTable(layout(2, 2), {0: (0, 0), 1: (1, 1)})
    assert table.replace_many(
        {
            0: path((0, 0), (1, 0)),
            1: path((1, 1), (0, 1), (0, 0), (0, 1)),
        },
        0,
    )
    result = table.search(0, (0, 0), 0, (op((0, 0), 2), op((1, 0), kind="dropoff")), set(), 7)
    assert result is not None and result.cost == 6
    handles = [step.tick for step in result.steps if step.action == "handle"]
    assert handles == [4, 5]
    assert result.steps[0].action != "handle"
    assert result.steps[1].cell != (0, 0)
    assert table.commit(0, result, 0)


def test_empty_and_zero_operations_still_need_a_safe_terminal_forever():
    table = ReservationTable(layout(2, 2), {0: (0, 0), 1: (1, 1)})
    assert table.replace_many(
        {
            0: path((0, 0), (1, 0)),
            1: path((1, 1), (0, 1), (0, 0), (0, 1)),
        },
        0,
    )
    for operations in ((), (op((0, 0)),)):
        assert table.search(0, (0, 0), 0, operations, set(), 2) is None
        result = table.search(0, (0, 0), 0, operations, set(), 4)
        assert result is not None and result.cost == 3
        assert result.end == (0, 0)
        assert result.initial_events == ()
        if operations:
            assert result.steps[-1].completed == ((1, "pickup"),)
            assert all(not step.completed for step in result.steps[:-1])
        assert table.clone().commit(0, result, 0)


@pytest.mark.parametrize("duration", [0, 1, 2, 3])
def test_final_completion_is_last_step_after_future_terminal_visit(duration):
    table = ReservationTable(layout(3, 2), {0: (0, 0), 1: (2, 1)})
    assert table.commit(1, path((2, 1), (2, 0), (2, 0), (2, 0), (2, 0), (1, 0), (1, 1)), 0)
    operations = (op((0, 0)), op((1, 0), duration, kind="dropoff"))
    expected_cost = 6 + duration
    assert brute_force_cost(table, (0, 0), 0, operations, set(), expected_cost) == expected_cost
    assert table.search(0, (0, 0), 0, operations, set(), expected_cost - 1) is None
    result = table.search(0, (0, 0), 0, operations, set(), expected_cost)
    assert result is not None and result.cost == expected_cost
    assert result.initial_events == ((1, "pickup"),)
    assert result.steps[-1].completed == ((1, "dropoff"),)
    assert all(not step.completed for step in result.steps[:-1])
    assert sum(step.action == "handle" for step in result.steps) == duration
    assert table.commit(0, result, 0)


@pytest.mark.parametrize("pickup_duration", [0, 1])
def test_final_zero_at_start_is_deferred_but_earlier_service_can_complete(pickup_duration):
    table = ReservationTable(layout(2, 2), {0: (0, 0), 1: (1, 1)})
    assert table.replace_many(
        {
            0: path((0, 0), (1, 0)),
            1: path((1, 1), (0, 1), (0, 0), (0, 1)),
        },
        0,
    )
    operations = (op((0, 0), pickup_duration), op((0, 0), kind="dropoff"))
    assert brute_force_cost(table, (0, 0), 0, operations, set(), 4) == 3
    result = table.search(0, (0, 0), 0, operations, set(), 4)
    assert result is not None and result.cost == 3
    assert result.initial_events == (() if pickup_duration else ((1, "pickup"),))
    if pickup_duration:
        assert result.steps[0].action == "handle"
        assert result.steps[0].completed == ((1, "pickup"),)
    assert result.steps[-1].completed == ((1, "dropoff"),)
    assert all((1, "dropoff") not in step.completed for step in result.steps[:-1])
    assert table.clone().commit(0, result, 0)
