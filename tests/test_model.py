"""Integration contracts for the real Mesa model, agents, protocol, and reservations.

The production layouts exercise the complete strategy design. Tiny deterministic
warehouses make disruption and service boundaries observable without long runs or
replacing any agent, routing search, message delivery, or environment implementation.
"""

from collections import Counter
from dataclasses import asdict
from itertools import combinations
from statistics import mean, pstdev
from typing import Any

import mesa
import pytest
from mesa.agentset import AbstractAgentSet
from mesa.datacollection import DataCollector

from warehouse_sim.agents import AllocatorAgent, CustomerAgent, RoutePlannerAgent, VehicleAgent
from warehouse_sim.domain import STRATEGIES, Config, Layout, Plan, TaskRecord, TaskSpec
from warehouse_sim.messages import ProtocolAgent
from warehouse_sim.model import WarehouseModel
from warehouse_sim.routing import ReservationTable


@pytest.fixture
def tiny_model():
    """Install a fully specified world before the first genuine model tick."""

    def make(*, tasks=None, starts=None, layout=None, strategy="S5", scenario="D1", **options):
        starts = {0: (0, 2), 1: (8, 4)} if starts is None else starts
        tasks = [TaskSpec(0, 0, (1, 2), (6, 2))] if tasks is None else tasks
        if layout is None:
            cells = frozenset((x, y) for x in range(9) for y in range(5))
            layout = Layout(
                "tiny",
                9,
                5,
                cells,
                tuple(dict.fromkeys(t.pickup for t in tasks)),
                tuple(dict.fromkeys(t.dropoff for t in tasks)),
                tuple(starts.values()),
                tuple(sorted(cells)),
            )
        config = Config(
            vehicles=len(starts),
            total_tasks=len(tasks),
            max_ticks=100,
            planning_horizon=64,
            **options,
        )
        model = WarehouseModel(config, strategy=strategy, scenario=scenario, seed=17, trace=True)
        model.layout = layout
        model.tasks = {task.id: TaskRecord(task) for task in tasks}
        model.customer.tasks = list(tasks)
        model.vehicle_roster = {v: layout.zone(cell) for v, cell in starts.items()}
        model.planner.table = ReservationTable(layout, starts)
        for vehicle in model.vehicles:
            vehicle.state.cell = starts[vehicle.state.id]
            vehicle.state.zone = layout.zone(vehicle.state.cell)
            vehicle.plan = Plan(vehicle.state.cell, 0, ())
        model.event_proposals = {}
        model.trajectory.clear()
        model._snapshot()
        return model

    return make


@pytest.fixture
def corridor_model(tiny_model):
    """A held AGV in the corridor prevents its follower from reaching the left exit."""

    def make():
        cells = frozenset({(x, 1) for x in range(7)} | {(2, 0), (4, 0)})
        starts = {0: (2, 1), 1: (4, 1)}
        tasks = [TaskSpec(0, 0, (2, 1), (2, 0)), TaskSpec(1, 0, (4, 1), (0, 1))]
        layout = Layout(
            "corridor",
            7,
            2,
            cells,
            ((2, 1), (4, 1)),
            ((2, 0), (0, 1)),
            tuple(starts.values()),
            tuple(sorted(cells)),
        )
        return tiny_model(tasks=tasks, starts=starts, layout=layout)

    return make


@pytest.fixture
def wiretap(monkeypatch):
    """Observe actual bus delivery, including replies, without substituting endpoints."""

    def attach(model):
        messages = []
        record = model.bus._record

        def observe(message):
            messages.append(message)
            record(message)

        monkeypatch.setattr(model.bus, "_record", observe)
        return messages

    return attach


def assert_reservations_safe(model):
    table = model.planner.table
    end = max(plan.end_tick for plan in table.plans.values())
    for tick in range(model.tick, max(model.tick, end) + 2):
        positions = {v: table.position(v, tick) for v in table.plans}
        assert len(set(positions.values())) == len(positions), (tick, positions)
        if tick == model.tick:
            assert positions == {v.state.id: v.state.cell for v in model.vehicles}
        for first, second in combinations(positions, 2):
            assert not (
                table.position(first, tick + 1) == positions[second]
                and table.position(second, tick + 1) == positions[first]
            ), (tick, first, second)


def checked_step(model):
    """Check physical safety, immutable ownership, FIFO queues, and future reservations."""
    before = {v.state.id: (v.state.cell, tuple(v.state.queue)) for v in model.vehicles}
    owners = {
        task.spec.id: (task.assigned_to, task.assigned_tick)
        for task in model.tasks.values()
        if task.assigned_to is not None
    }
    model.step()
    assert model.tick == model.steps
    for task_id, owner in owners.items():
        task = model.tasks[task_id]
        assert (task.assigned_to, task.assigned_tick) == owner
    for vehicle in model.vehicles:
        state = vehicle.state
        previous_cell, previous_queue = before[state.id]
        assert abs(previous_cell[0] - state.cell[0]) + abs(previous_cell[1] - state.cell[1]) <= 1
        assert state.cell not in model.active_blockages
        assert len(state.queue) <= model.config.queue_max
        survivors = [t for t in previous_queue if model.tasks[t].dropoff_tick is None]
        assert state.queue[: len(survivors)] == survivors
        for task_id in state.queue:
            assert model.tasks[task_id].assigned_to == state.id
    assert_reservations_safe(model)


def assert_trace_consistent(model, result):
    ticks = result["metrics"]["simulation_ticks"]
    trace = result["trace"]
    assert [snapshot["tick"] for snapshot in trace] == list(range(ticks + 1))
    for snapshot in trace:
        cells = [v["cell"] for v in snapshot["vehicles"]]
        assert len(set(cells)) == len(cells)
        assert not set(cells) & set(snapshot["blocked"])
        for vehicle in snapshot["vehicles"]:
            assert vehicle["cell"] in model.layout.cells
            assert vehicle["cell"] == model.planner.table.position(vehicle["id"], snapshot["tick"])
            assert len(vehicle["queue"]) <= model.config.queue_max
    for before, after in zip(trace, trace[1:], strict=False):
        for first, second in combinations(range(len(model.vehicles)), 2):
            assert not (
                before["vehicles"][first]["cell"] == after["vehicles"][second]["cell"]
                and before["vehicles"][second]["cell"] == after["vehicles"][first]["cell"]
            )
        for old, new in zip(before["vehicles"], after["vehicles"], strict=True):
            distance = sum(abs(a - b) for a, b in zip(old["cell"], new["cell"], strict=True))
            assert distance <= 1
    for vehicle in model.vehicles:
        assert vehicle.state.cell == trace[-1]["vehicles"][vehicle.state.id]["cell"]
        moves = sum(
            a["vehicles"][vehicle.state.id]["cell"] != b["vehicles"][vehicle.state.id]["cell"]
            for a, b in zip(trace, trace[1:], strict=False)
        )
        assert moves == vehicle.state.moves
        empty_moves = 0
        for before, after in zip(trace, trace[1:], strict=False):
            if (
                before["vehicles"][vehicle.state.id]["cell"]
                == after["vehicles"][vehicle.state.id]["cell"]
            ):
                continue
            carrying = any(
                t["assigned_to"] == vehicle.state.id
                and t["pickup_tick"] is not None
                and t["pickup_tick"] <= before["tick"]
                and (t["dropoff_tick"] is None or t["dropoff_tick"] >= after["tick"])
                for t in result["tasks"]
            )
            empty_moves += not carrying
        assert empty_moves == vehicle.state.empty_moves

    messages = result["message_trace"]
    assert [message["tick"] for message in messages] == sorted(m["tick"] for m in messages)
    assert all(0 <= message["tick"] <= ticks for message in messages)
    assert Counter(m["performative"] for m in messages) == result["messages"]["by_performative"]
    assert Counter(m["sender"] for m in messages) == result["messages"]["by_sender"]
    for task in result["tasks"]:
        conversation = f"task:{task['id']}"
        releases = [
            m for m in messages if m["sender"] == "customer" and m["conversation"] == conversation
        ]
        assert [m["tick"] for m in releases] == (
            [task["created"]] if task["created"] < ticks else []
        )
        completions = [
            m
            for m in messages
            if m["receiver"] == "allocator"
            and m["sender"].startswith("agv:")
            and m["conversation"] == conversation
        ]
        if task["dropoff_tick"] is None:
            assert not completions
        else:
            assert len(completions) == 1
            assert completions[0]["tick"] == task["dropoff_tick"]
            assert completions[0]["sender"] == f"agv:{task['assigned_to']}"

    series = result["timeseries"]
    assert set(series) == {
        "completed_tasks",
        "pending_tasks",
        "active_disruptions",
        "communication_volume",
    }
    assert all(len(values) == ticks for values in series.values())
    assert series["completed_tasks"] == [
        sum(t["dropoff_tick"] is not None and t["dropoff_tick"] <= tick for t in result["tasks"])
        for tick in range(1, ticks + 1)
    ]
    assert series["active_disruptions"] == [len(s["blocked"]) for s in trace[1:]]
    assert series["communication_volume"] == sorted(series["communication_volume"])
    assert series["communication_volume"][-1] == result["metrics"]["communication_volume"]
    assert series["pending_tasks"][-1] == len(model.allocator.pending)


def assert_accounting(model, result):
    metrics = result["metrics"]
    vehicles = result["vehicles"]
    tasks = result["tasks"]
    completed = [t for t in tasks if t["dropoff_tick"] is not None]
    done, ticks = len(completed), model.tick
    for vehicle in vehicles:
        assert vehicle["busy"] + vehicle["idle"] == ticks
        assert vehicle["moves"] + vehicle["waits"] + vehicle["handles"] == vehicle["busy"]
        assert 0 <= vehicle["empty_moves"] <= vehicle["moves"]
        assert vehicle["completed"] == sum(t["assigned_to"] == vehicle["id"] for t in completed)
    assert sum(v["completed"] for v in vehicles) == done
    assert metrics["completed_tasks"] == done
    assert metrics["unfinished_tasks"] == len(tasks) - done
    assert metrics["execution_failures"] == int(done != len(tasks))
    assert metrics["simulation_ticks"] == ticks
    assert metrics["throughput"] == pytest.approx(done / ticks)
    busy = sum(v["busy"] for v in vehicles)
    assert metrics["utilization"] == pytest.approx(busy / (len(vehicles) * ticks))
    assert metrics["idle_time"] == len(vehicles) * ticks - busy
    assert metrics["sum_of_costs"] == busy
    moves = sum(v["moves"] for v in vehicles)
    assert metrics["sum_of_moves"] == moves
    empty_ratio = sum(v["empty_moves"] for v in vehicles) / moves if moves else None
    assert metrics["empty_travel_ratio"] == pytest.approx(empty_ratio)
    counts = [v["completed"] for v in vehicles]
    cv = pstdev(counts) / mean(counts) if done else None
    assert metrics["load_balance_cv"] == pytest.approx(cv)
    completion_time = mean(t["dropoff_tick"] - t["created"] for t in completed) if done else None
    pickup_delay = mean(t["pickup_tick"] - t["created"] for t in completed) if done else None
    assert metrics["mean_completion_time"] == pytest.approx(completion_time)
    assert metrics["pickup_delay"] == pytest.approx(pickup_delay)
    makespan = (
        max(t["dropoff_tick"] for t in completed) - min(t["created"] for t in tasks)
        if done == len(tasks)
        else None
    )
    assert metrics["makespan"] == makespan
    assert metrics["communication_volume"] == sum(result["messages"]["by_performative"].values())
    assert metrics["communication_volume"] == sum(result["messages"]["by_sender"].values())
    events = result["disruptions"]
    assert metrics["disruption_proposals"] == len(events)
    assert metrics["accepted_disruptions"] == sum(e["accepted"] for e in events)
    assert metrics["skipped_disruptions"] == sum(not e["accepted"] for e in events)
    assert metrics["detected_disruptions"] == sum(e["observed_tick"] is not None for e in events)
    for task in tasks:
        if task["assigned_to"] is not None:
            assert task["created"] <= task["assigned_tick"] <= ticks
            if model.strategy.scope == "zone":
                assert (
                    model.layout.zone(task["pickup"]) == model.vehicle_roster[task["assigned_to"]]
                )
        if task["pickup_tick"] is not None:
            assert task["assigned_tick"] <= task["pickup_tick"] <= ticks
        if task["dropoff_tick"] is not None:
            assert task["pickup_tick"] <= task["dropoff_tick"] <= ticks
    if done == len(tasks):
        assert sum(v["handles"] for v in vehicles) == len(tasks) * (
            model.config.pickup_ticks + model.config.dropoff_ticks
        )
        assert all(not v["queue"] and v["carrying"] is None for v in vehicles)


@pytest.mark.parametrize("total_tasks", [6, 12])
@pytest.mark.parametrize("layout", ["A", "B"])
@pytest.mark.parametrize("scenario", ["D0", "D4"])
@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_strategy_design_completes_reproducibly_with_exact_accounting(
    total_tasks,
    layout,
    scenario,
    strategy,
):
    config = Config(total_tasks=total_tasks, max_ticks=350, planning_horizon=80, arrival_interval=2)
    options: dict[str, Any] = dict(
        strategy=strategy, scenario=scenario, layout=layout, seed=17, trace=True
    )
    model = WarehouseModel(config, **options)
    result = model.run()
    assert result == WarehouseModel(config, **options).run()
    assert (result["strategy"], result["scenario"], result["layout"], result["seed"]) == (
        strategy,
        scenario,
        layout,
        17,
    )
    assert result["config"] == asdict(config)
    assert result["metrics"]["unfinished_tasks"] == 0
    assert not model.running
    assert model.steps == model.tick
    assert_accounting(model, result)
    assert_trace_consistent(model, result)
    assert_reservations_safe(model)


@pytest.mark.parametrize("layout", ["A", "B"])
@pytest.mark.parametrize("strategy", list(STRATEGIES))
def test_each_strategy_handles_a_forced_disruption_on_its_actual_route(layout, strategy, wiretap):
    config = Config(total_tasks=6, max_ticks=350, planning_horizon=80, arrival_interval=2)
    model = WarehouseModel(
        config,
        strategy=strategy,
        scenario="D4",
        layout=layout,
        seed=17,
        trace=True,
    )
    messages = wiretap(model)
    model.event_proposals = {}
    checked_step(model)
    occupied = {v.state.cell for v in model.vehicles}
    candidates = [
        step
        for vehicle in model.vehicles
        for step in vehicle.plan.steps
        if step.tick > model.tick
        and step.cell in model.layout.disruption_cells
        and step.cell not in occupied
    ]
    assert candidates, "The fixture must obstruct a real future route, not an unrelated aisle"
    blocked = min(candidates, key=lambda step: (step.tick, step.cell)).cell
    model.event_proposals = {model.tick: blocked}
    while model.running:
        checked_step(model)
    result = model.result()
    assert len(result["disruptions"]) == 1
    event = result["disruptions"][0]
    assert event["accepted"] and event["observed_tick"] is not None
    discoveries = [
        m
        for m in messages
        if blocked in m.content.get("blocked", ()) and m.content.get("kind") == "observation"
    ]
    assert len(discoveries) == 1
    discovery = discoveries[0]
    assert discovery.tick == event["observed_tick"]
    observer_id = int(discovery.sender.split(":")[1])
    observed_cell = result["trace"][discovery.tick]["vehicles"][observer_id]["cell"]
    assert sum(abs(a - b) for a, b in zip(observed_cell, blocked, strict=True)) == 1
    assert all(
        snapshot["vehicles"][observer_id]["cell"] == observed_cell
        for snapshot in result["trace"][discovery.tick : event["end"] + 1]
    )
    assert result["metrics"]["unfinished_tasks"] == 0
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


def test_real_mesa_registration_activation_and_datacollection():
    model = WarehouseModel(Config(total_tasks=6, max_ticks=200), trace=True)
    assert isinstance(model, mesa.Model)
    assert isinstance(model.agents, AbstractAgentSet)
    assert isinstance(model.datacollector, DataCollector)
    assert len(model.agents) == model.config.vehicles + 3
    expected = {
        AllocatorAgent: [model.allocator],
        RoutePlannerAgent: [model.planner],
        CustomerAgent: [model.customer],
        VehicleAgent: model.vehicles,
    }
    for agent_type, agents in expected.items():
        assert isinstance(model.agents_by_type[agent_type], AbstractAgentSet)
        assert list(model.agents_by_type[agent_type]) == agents
        for agent in agents:
            assert isinstance(agent, mesa.Agent)
            assert isinstance(agent, ProtocolAgent)
            assert agent.model is model
            assert model.bus.endpoints[agent.name] is agent
    assert len({a.unique_id for a in model.agents}) == len(model.agents)
    checked_step(model)
    assert len(model.datacollector.model_vars["completed_tasks"]) == 1
    assert model.datacollector.get_model_vars_dataframe().shape == (1, 4)


@pytest.mark.parametrize("layout", ["A", "B"])
def test_initial_workload_and_positions_are_matched_across_entire_design(layout):
    config = Config(total_tasks=12, max_ticks=200)
    reference = WarehouseModel(config, layout=layout, seed=29)
    specs = [t.spec for t in reference.tasks.values()]
    starts = {v.state.id: v.state.cell for v in reference.vehicles}
    for strategy in STRATEGIES:
        for scenario in ("D0", "D1", "D2", "D3", "D4"):
            model = WarehouseModel(
                config, strategy=strategy, scenario=scenario, layout=layout, seed=29
            )
            assert [t.spec for t in model.tasks.values()] == specs
            assert {v.state.id: v.state.cell for v in model.vehicles} == starts
            assert model.vehicle_roster == reference.vehicle_roster
    different = WarehouseModel(config, layout=layout, seed=30)
    assert [t.spec for t in different.tasks.values()] != specs


@pytest.mark.parametrize("layout", ["A", "B"])
def test_proposal_process_is_common_nested_and_independent_of_strategy(layout):
    config = Config(total_tasks=6, max_ticks=3000)
    models = {
        scenario: WarehouseModel(config, scenario=scenario, layout=layout, seed=23)
        for scenario in ("D0", "D1", "D2", "D3", "D4")
    }
    proposals = {scenario: m.event_proposals for scenario, m in models.items()}
    assert not proposals["D0"]
    assert proposals["D1"] == proposals["D2"]
    assert proposals["D3"] == proposals["D4"]
    assert proposals["D1"]
    assert proposals["D1"].items() < proposals["D3"].items()
    for strategy in STRATEGIES:
        other = WarehouseModel(config, strategy=strategy, scenario="D4", layout=layout, seed=23)
        assert other.event_proposals == proposals["D4"]
    for model in models.values():
        assert all(0 <= tick < config.max_ticks for tick in model.event_proposals)
        assert set(model.event_proposals.values()) <= set(model.layout.disruption_cells)


@pytest.mark.parametrize("layout", ["A", "B"])
def test_real_default_workload_and_unique_output_capacity(layout):
    assert Config().total_tasks == 48
    default = WarehouseModel(layout=layout, seed=2)
    assert len(default.tasks) == 48
    assert len(default.layout.dropoffs) == 68
    assert len({t.spec.dropoff for t in default.tasks.values()}) == 48
    full = WarehouseModel(Config(total_tasks=68), layout=layout, seed=2)
    assert len({t.spec.dropoff for t in full.tasks.values()}) == 68
    assert set(full.vehicle_roster.values()) == {0, 1}
    with pytest.raises(ValueError, match="unique output slots"):
        WarehouseModel(Config(total_tasks=69), layout=layout)


@pytest.mark.parametrize("scenario", ["D1", "D4"])
def test_local_discovery_and_first_observer_holds_until_actual_clearance(
    tiny_model,
    wiretap,
    scenario,
):
    model = tiny_model(scenario=scenario)
    messages = wiretap(model)
    blocked = (3, 2)
    model.event_proposals = {0: blocked}
    checked_step(model)
    event = model.disruptions[0]
    assert event["accepted"]
    assert not model.planner.known_blocked
    assert all(not v.known_blocked for v in model.vehicles)
    assert event["observed_tick"] is None
    # An unobserved environment event must not silently influence route estimates.
    assert blocked in [step.cell for step in model.vehicles[0].plan.steps]
    while not model.planner.monitors:
        checked_step(model)
    assert event["observed_tick"] == 3
    assert model.planner.monitors == {blocked: 0}
    observer = model.vehicles[0]
    held_cell, queue = observer.state.cell, tuple(observer.state.queue)
    assert held_cell == (2, 2)
    assert observer.holding and 0 in model.planner.held
    assert observer.state.carrying == 0
    observations = [m for m in messages if m.content.get("kind") == "observation"]
    assert len(observations) == 1
    assert observations[0].sender == "agv:0"
    assert observations[0].tick == event["observed_tick"]
    assert observations[0].content == {
        "kind": "observation",
        "blocked": {blocked},
        "cleared": set(),
    }
    maps = [m for m in messages if m.content.get("kind") == "map"]
    assert {m.receiver for m in maps} == {v.name for v in model.vehicles}
    assert all(m.performative == "INFORM" and m.tick == 3 for m in maps)
    assert all(v.known_blocked == {blocked} for v in model.vehicles)
    # Extending the environment lifetime must not expose an expiry timer to an AGV.
    original_end = event["end"]
    event["end"] += 3
    while model.tick < event["end"]:
        checked_step(model)
        assert observer.state.cell == held_cell
        assert tuple(observer.state.queue) == queue
        assert observer.holding
        assert model.planner.monitors == {blocked: 0}
        assert model.planner.known_blocked == {blocked}
    assert model.tick > original_end
    checked_step(model)  # Expiry is sensed locally before this tick's movement.
    assert not model.planner.monitors
    assert not model.planner.known_blocked
    assert all(not v.known_blocked for v in model.vehicles)
    assert not observer.holding
    assert observer.state.cell != held_cell
    clearances = [m for m in messages if m.content.get("cleared")]
    assert len(clearances) == 1
    assert clearances[0].sender == "agv:0"
    assert clearances[0].tick == event["end"]
    result = model.run()
    assert result["metrics"]["unfinished_tasks"] == 0
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


def test_environment_skips_occupied_and_overlapping_proposals_without_relocation(tiny_model):
    model = tiny_model(tasks=[TaskSpec(0, 90, (1, 2), (6, 2))])
    occupied, remote = model.vehicles[0].state.cell, (4, 0)
    model.event_proposals = {0: occupied, 1: remote, 2: remote, 11: remote}
    for _ in range(12):
        checked_step(model)
    events = model.disruptions
    assert [e["cell"] for e in events] == [occupied, remote, remote, remote]
    assert [e["accepted"] for e in events] == [False, True, False, True]
    assert [e["skip_reason"] for e in events] == ["occupied", None, "overlap", None]
    assert [(e["start"], e["end"], e["duration"]) for e in events] == [
        (0, 10, 10),
        (1, 11, 10),
        (2, 12, 10),
        (11, 21, 10),
    ]
    assert model.active_blockages == {remote: events[-1]}
    assert all(e["observed_tick"] is None for e in events)
    assert not model.planner.known_blocked
    assert_accounting(model, model.result())


@pytest.mark.parametrize("blocked", [(3, 1), (2, 0)], ids=["direct-route", "hold-closure"])
@pytest.mark.parametrize("extra_change", [False, True], ids=["interval-retry", "map-change-retry"])
def test_coordinated_closure_preserves_queues_and_retries_nonmonitors(
    corridor_model,
    wiretap,
    monkeypatch,
    blocked,
    extra_change,
):
    model = corridor_model()
    messages = wiretap(model)
    table = model.planner.table
    replace_many = table.replace_many
    publications = []

    def publish(replacements, now):
        old = table.plans.copy()
        assert replace_many(replacements, now)
        # Publication may not truncate/rewrite executed movements or completion events.
        for vehicle, plan in old.items():
            history = tuple(s for s in plan.steps if s.tick <= now)
            assert table.plans[vehicle].steps[: len(history)] == history
        assert_reservations_safe(model)
        publications.append(now)
        return True

    monkeypatch.setattr(table, "replace_many", publish)
    model.event_proposals = {1: blocked}
    if extra_change:
        model.event_proposals[2] = (1, 1)
    checked_step(model)
    queues = [tuple(v.state.queue) for v in model.vehicles]
    assert queues == [(0,), (1,)]
    assert (2, 1) in [s.cell for s in model.vehicles[1].plan.steps if s.tick > model.tick]
    if blocked == (2, 0):
        assert blocked not in [s.cell for s in model.vehicles[1].plan.steps]
    checked_step(model)
    assert model.planner.monitors[blocked] == 0
    assert model.planner.held == {0, 1}
    assert all(v.holding for v in model.vehicles)
    assert [tuple(v.state.queue) for v in model.vehicles] == queues
    assert publications == [1]
    held_cells = [v.state.cell for v in model.vehicles]
    while model.tick < 11:
        tick = model.tick
        begin = len(messages)
        checked_step(model)
        requests = [
            m.receiver
            for m in messages[begin:]
            if m.performative == "REQUEST" and m.content.get("kind") == "remaining"
        ]
        changed = extra_change and tick == 2
        assert requests.count("agv:1") == int(changed or tick % model.config.auction_interval == 0)
        assert requests.count("agv:0") == int(changed)
        assert [v.state.cell for v in model.vehicles] == held_cells
        assert [tuple(v.state.queue) for v in model.vehicles] == queues
    while model.running:
        checked_step(model)
    result = model.result()
    assert result["metrics"]["unfinished_tasks"] == 0
    assert len(publications) >= 4
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


def test_failed_reservation_publication_is_atomic_in_live_model(corridor_model):
    model = corridor_model()
    checked_step(model)
    table = model.planner.table
    before = table.plans.copy()
    # Both holds have valid starts/history, but jointly holding the leader blocks
    # the follower's already committed future. No part of this transaction may leak.
    leader_hold = Plan(model.vehicles[0].state.cell, model.tick, ())
    assert not table.replace_many({0: leader_hold}, model.tick)
    assert table.plans == before
    assert all(v.plan == before[v.state.id] for v in model.vehicles)
    while model.running:
        checked_step(model)
    assert model.result()["metrics"]["unfinished_tasks"] == 0


@pytest.mark.parametrize("pickup_ticks,dropoff_ticks", [(0, 0), (0, 1), (1, 0)])
def test_zero_duration_services_deliver_each_event_once_with_exact_timestamps(
    tiny_model,
    wiretap,
    pickup_ticks,
    dropoff_ticks,
):
    tasks = [TaskSpec(0, 0, (0, 2), (0, 2)), TaskSpec(1, 0, (0, 2), (2, 2))]
    model = tiny_model(tasks=tasks, pickup_ticks=pickup_ticks, dropoff_ticks=dropoff_ticks)
    messages = wiretap(model)
    while model.running:
        checked_step(model)
    result = model.result()
    first, second = result["tasks"]
    assert first["assigned_tick"] == second["assigned_tick"] == 0
    assert first["pickup_tick"] == pickup_ticks
    assert first["dropoff_tick"] == pickup_ticks + dropoff_ticks
    assert second["pickup_tick"] == 2 * pickup_ticks + dropoff_ticks
    assert second["dropoff_tick"] == 2 * (pickup_ticks + dropoff_ticks) + 2
    completions = [m for m in messages if "completion_tick" in m.content]
    assert [(m.content["done"], m.tick, m.content["completion_tick"]) for m in completions] == [
        (t["id"], t["dropoff_tick"], t["dropoff_tick"]) for t in result["tasks"]
    ]
    assert all(t["assigned_to"] == 0 for t in result["tasks"])
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


def test_downstream_block_preserves_partially_executed_three_tick_handling(tiny_model, wiretap):
    model = tiny_model(
        tasks=[TaskSpec(0, 0, (0, 2), (4, 2))],
        pickup_ticks=3,
        dropoff_ticks=3,
    )
    messages = wiretap(model)
    model.event_proposals = {1: (1, 2)}
    checked_step(model)
    vehicle = model.vehicles[0]
    assert vehicle.handling_done == 1
    assert model.tasks[0].pickup_tick is None
    checked_step(model)
    remaining = [m for m in messages if m.sender == "agv:0" and "active_handling" in m.content]
    assert len(remaining) == 1
    assert remaining[0].content["active_handling"]
    assert remaining[0].content["operations"][0].duration == 2
    assert vehicle.handling_done == 2
    assert model.tasks[0].pickup_tick is None
    checked_step(model)
    assert model.tasks[0].pickup_tick == 3
    assert vehicle.state.carrying == 0
    assert vehicle.handling_done == 0
    assert [(s.tick, s.action, s.cell) for s in vehicle.plan.steps[:3]] == [
        (1, "handle", (0, 2)),
        (2, "handle", (0, 2)),
        (3, "handle", (0, 2)),
    ]
    assert [s.completed for s in vehicle.plan.steps[:3]] == [(), (), ((0, "pickup"),)]
    while model.running:
        checked_step(model)
    result = model.result()
    assert result["tasks"][0]["dropoff_tick"] == 18
    assert vehicle.state.handles == 6
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


@pytest.mark.parametrize("strategy,capacity", [("S2", 1), ("S5", 3), ("S9", 3)])
def test_idle_queue_and_bundle_commitment_enforce_capacity_and_fifo(tiny_model, strategy, capacity):
    tasks = [
        TaskSpec(i, 0, (1, 2), dropoff)
        for i, dropoff in enumerate(((6, 2), (6, 1), (6, 3), (7, 2)))
    ]
    model = tiny_model(tasks=tasks, strategy=strategy)
    checked_step(model)
    assert len(model.vehicles[0].state.queue) == capacity
    assert not model.vehicles[1].state.queue
    assert len(model.allocator.pending) == len(tasks) - capacity
    first_queue = tuple(model.vehicles[0].state.queue)
    # A full queue (or a busy idle-only AGV) must refuse even a feasible local task.
    bids = model.bus.send("CFP", "allocator", "agv:0", {"tasks": [tasks[-1]]})
    assert bids == {"bids": []}
    while model.running:
        checked_step(model)
    result = model.result()
    completion_order = sorted(tasks, key=lambda t: model.tasks[t.id].dropoff_tick)
    assert tuple(t.id for t in completion_order[:capacity]) == first_queue
    assert all(t["assigned_to"] == 0 for t in result["tasks"])
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


@pytest.mark.parametrize("strategy", ["S5", "S9"])
def test_disruption_replanning_preserves_a_full_three_task_queue(tiny_model, strategy):
    tasks = [TaskSpec(i, 0, (1, 2), dropoff) for i, dropoff in enumerate(((6, 2), (6, 1), (6, 3)))]
    model = tiny_model(tasks=tasks, strategy=strategy)
    model.event_proposals = {0: (3, 2)}
    checked_step(model)
    vehicle = model.vehicles[0]
    committed = tuple(vehicle.state.queue)
    assert len(committed) == 3
    assert not model.allocator.pending
    while not vehicle.holding:
        checked_step(model)
    assert model.planner.monitors == {(3, 2): 0}
    assert tuple(vehicle.state.queue) == committed
    assert all(model.tasks[t].dropoff_tick is None for t in committed)
    event = model.disruptions[0]
    while model.tick < event["end"]:
        checked_step(model)
        assert tuple(vehicle.state.queue) == committed
    while model.running:
        checked_step(model)
    assert tuple(sorted(committed, key=lambda t: model.tasks[t].dropoff_tick)) == committed
    for previous, following in zip(committed, committed[1:], strict=False):
        assert model.tasks[previous].dropoff_tick <= model.tasks[following].pickup_tick
    result = model.result()
    assert result["metrics"]["unfinished_tasks"] == 0
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


@pytest.mark.parametrize("strategy", [s.id for s in STRATEGIES.values() if s.scope == "zone"])
def test_zone_filter_applies_to_each_bid_not_just_allocator_shortlisting(tiny_model, strategy):
    tasks = [TaskSpec(0, 0, (1, 2), (3, 2)), TaskSpec(1, 0, (7, 2), (6, 2))]
    model = tiny_model(tasks=tasks, strategy=strategy)
    for vehicle in model.vehicles:
        response = model.bus.send("CFP", "allocator", vehicle.name, {"tasks": tasks})
        assert response["bids"]
        eligible = {t.id for t in tasks if model.layout.zone(t.pickup) == vehicle.state.zone}
        assert all(set(bid["tasks"]) <= eligible for bid in response["bids"])
        wrong_zone = [t for t in tasks if t.id not in eligible]
        assert model.bus.send("CFP", "allocator", vehicle.name, {"tasks": wrong_zone}) == {
            "bids": []
        }
    result = model.run()
    assert result["metrics"]["unfinished_tasks"] == 0
    assert [t["assigned_to"] for t in result["tasks"]] == [0, 1]
    assert_accounting(model, result)


@pytest.mark.parametrize(
    "route,availability", [("S3", "S4"), ("S5", "S6"), ("S7", "S8"), ("S9", "S10")]
)
def test_availability_utility_uses_bidders_own_committed_tail_without_mutating_it(
    tiny_model,
    route,
    availability,
):
    tasks = [TaskSpec(0, 0, (1, 2), (6, 2)), TaskSpec(1, 80, (2, 2), (5, 3))]
    model = tiny_model(tasks=tasks, strategy=route)
    checked_step(model)
    assert model.vehicles[0].state.queue == [0]
    assert not model.vehicles[1].state.queue
    plans = model.planner.table.plans.copy()
    queues = [tuple(v.state.queue) for v in model.vehicles]
    for vehicle in model.vehicles:
        model.strategy = STRATEGIES[route]
        route_bid = vehicle._bid([tasks[1]])
        model.strategy = STRATEGIES[availability]
        availability_bid = vehicle._bid([tasks[1]])
        assert route_bid is not None and availability_bid is not None
        plan = model.planner.estimates[availability_bid["estimate_id"]].plan
        assert plan.start == vehicle.plan.end
        assert plan.start_tick == max(model.tick, vehicle.plan.end_tick)
        delay = max(0, vehicle.plan.end_tick - model.tick)
        assert route_bid["utility"] == -plan.cost
        assert availability_bid["utility"] == -plan.cost - delay
        assert route_bid["utility"] - availability_bid["utility"] == delay
        assert bool(delay) == (vehicle.state.id == 0)
    assert model.planner.table.plans == plans
    assert [tuple(v.state.queue) for v in model.vehicles] == queues


@pytest.mark.parametrize(
    "some_completed", [False, True], ids=["no-completion", "partial-completion"]
)
def test_max_ticks_failure_keeps_unfinished_tasks_and_missing_completion_metrics(
    tiny_model, some_completed
):
    tasks = [TaskSpec(0, 0, (0, 2), (1, 2)), TaskSpec(1, 90, (1, 2), (6, 2))]
    model = tiny_model(tasks=tasks)
    # The factory has already built real agents; changing this limit does not alter demand.
    limit = 4 if some_completed else 1
    model.config = Config(vehicles=2, total_tasks=2, max_ticks=limit, planning_horizon=64)
    result = model.run()
    metrics = result["metrics"]
    assert not model.running
    assert model.tick == limit
    assert metrics["execution_failures"] == 1
    assert metrics["unfinished_tasks"] == (1 if some_completed else 2)
    assert metrics["completed_tasks"] == int(some_completed)
    assert metrics["makespan"] is None
    assert result["tasks"][1]["assigned_to"] is None
    assert result["tasks"][1]["pickup_tick"] is None
    assert result["tasks"][1]["dropoff_tick"] is None
    if some_completed:
        assert metrics["mean_completion_time"] == 3
        assert metrics["pickup_delay"] == 1
    else:
        for key in ("mean_completion_time", "pickup_delay", "load_balance_cv"):
            assert metrics[key] is None
    assert_accounting(model, result)
    assert_trace_consistent(model, result)


def test_trace_disabled_preserves_behavior_and_communication_accounting():
    config = Config(total_tasks=6, max_ticks=250, planning_horizon=80)
    plain = WarehouseModel(config, strategy="S8", scenario="D4", layout="B", seed=17).run()
    traced = WarehouseModel(
        config, strategy="S8", scenario="D4", layout="B", seed=17, trace=True
    ).run()
    assert plain["trace"] == plain["message_trace"] == []
    assert plain["timeseries"] == {}
    for key in ("metrics", "tasks", "vehicles", "disruptions", "messages"):
        assert plain[key] == traced[key]
    assert plain["metrics"]["communication_volume"] > 0
