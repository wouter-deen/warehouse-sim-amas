"""Mesa warehouse model with deterministic protocol phases and simultaneous movement."""

import math
import random
import statistics
from dataclasses import asdict

import mesa
from mesa.datacollection import DataCollector

from warehouse_sim.agents import AllocatorAgent, CustomerAgent, RoutePlannerAgent, VehicleAgent
from warehouse_sim.domain import SCENARIOS, STRATEGIES, Config, TaskRecord
from warehouse_sim.layouts import get_layout, workload
from warehouse_sim.messages import MessageBus


class WarehouseModel(mesa.Model):
    """Each Mesa step is one physical tick; no movement occurs during auctions/replanning.

    The environment alone knows disruption lifetimes. Planners learn cells only through
    AGV sensor reports. The event proposal stream is independent of allocation behavior;
    occupied-cell and overlapping proposals are logged and skipped rather than relocated.
    """

    def __init__(self, config=None, strategy="S1", scenario="D0", layout="A", seed=0, trace=False):
        super().__init__(rng=seed)
        config = Config() if config is None else config
        self.config = config
        self.strategy = STRATEGIES[strategy]
        self.disruption_scenario = SCENARIOS[scenario]
        self.layout = get_layout(layout)
        self.scenario_seed = seed
        self.record_trace = trace
        self.tick = 0
        self.bus = MessageBus(self)
        specs, starts = workload(self.layout, config, seed)
        self.tasks = {t.id: TaskRecord(t) for t in specs}
        self.vehicle_roster = {v: self.layout.zone(c) for v, c in starts.items()}
        self.allocator = AllocatorAgent(self)
        self.planner = RoutePlannerAgent(self, starts)
        self.customer = CustomerAgent(self, specs)
        self.vehicles = [VehicleAgent(self, v, starts[v]) for v in sorted(starts)]
        self.active_blockages = {}
        self.disruptions = []
        self.event_proposals = self._generate_proposals()
        self.deadlock_episodes = 0
        self.no_progress = 0
        self.in_stall = False
        self.trajectory = []
        self.datacollector = DataCollector(
            model_reporters={
                "completed_tasks": lambda m: sum(
                    t.dropoff_tick is not None for t in m.tasks.values()
                ),
                "pending_tasks": lambda m: len(m.allocator.pending),
                "active_disruptions": lambda m: len(m.active_blockages),
                "communication_volume": lambda m: sum(m.bus.counts.values()),
            }
        )
        if trace:
            self._snapshot()

    def _generate_proposals(self):
        rng = random.Random(f"disruptions:{self.scenario_seed}")
        proposals = {}
        # Thinning a common marked process gives D1/D2 matching starts/cells and
        # D3/D4 matching starts/cells, with rare proposals nested in frequent ones.
        for tick in range(self.config.max_ticks):
            mark = rng.random()
            cell = rng.choice(self.layout.disruption_cells)
            if mark < self.disruption_scenario.frequency / 1000:
                proposals[tick] = cell
        return proposals

    def _environment(self):
        self.active_blockages = {
            c: e for c, e in self.active_blockages.items() if self.tick < e["end"]
        }
        if self.tick not in self.event_proposals:
            return
        cell = self.event_proposals[self.tick]
        occupied = {vehicle.state.cell for vehicle in self.vehicles}
        reason = (
            "occupied" if cell in occupied else "overlap" if cell in self.active_blockages else None
        )
        event = {
            "cell": cell,
            "start": self.tick,
            "end": self.tick + self.disruption_scenario.duration,
            "duration": self.disruption_scenario.duration,
            "accepted": reason is None,
            "skip_reason": reason,
            "observed_tick": None,
        }
        self.disruptions.append(event)
        if reason is None:
            self.active_blockages[cell] = event

    def mark_observation(self, cell):
        event = self.active_blockages.get(cell)
        if event is not None and event["observed_tick"] is None:
            event["observed_tick"] = self.tick

    def _snapshot(self):
        self.trajectory.append(
            {
                "tick": self.tick,
                "vehicles": [
                    {
                        "id": v.state.id,
                        "cell": v.state.cell,
                        "carrying": v.state.carrying,
                        "queue": list(v.state.queue),
                    }
                    for v in self.vehicles
                ],
                "blocked": sorted(self.active_blockages),
                "known_blocked": sorted(self.planner.known_blocked),
            }
        )

    def step(self):
        self._environment()
        self.customer.step()
        # Explicit Mesa AgentSet activation keeps observation and physical movement
        # separate. The latter is safe to apply sequentially because routes were
        # checked against the same pre-movement reservation snapshot.
        self.agents_by_type[VehicleAgent].do("observe")
        self.planner.replan()
        self.allocator.step()
        self.planner.estimates.clear()
        before = {v.state.id: v.state.cell for v in self.vehicles}
        progress_before = sum(v.state.moves + v.state.handles for v in self.vehicles)
        self.agents_by_type[VehicleAgent].do("step")
        after = {v.state.id: v.state.cell for v in self.vehicles}
        if len(set(after.values())) != len(after):
            raise AssertionError(f"Vertex collision at {self.tick + 1}")
        for first in after:
            if after[first] in self.active_blockages:
                raise AssertionError(f"AGV entered an obstruction at {self.tick + 1}")
            if after[first] != self.planner.table.position(first, self.tick + 1):
                raise AssertionError("Physical state diverged from the committed reservation")
            for second in after:
                if (
                    first < second
                    and before[first] == after[second]
                    and before[second] == after[first]
                ):
                    raise AssertionError("Opposite-edge collision")
        progress_after = sum(v.state.moves + v.state.handles for v in self.vehicles)
        has_work = any(v.state.queue for v in self.vehicles) or bool(self.allocator.pending)
        if has_work and progress_after == progress_before:
            self.no_progress += 1
            if self.no_progress >= self.config.deadlock_window and not self.in_stall:
                self.deadlock_episodes += 1
                self.in_stall = True
        else:
            self.no_progress = 0
            self.in_stall = False
        self.tick += 1
        self.datacollector.collect(self)
        if self.record_trace:
            self._snapshot()
        if (
            all(t.dropoff_tick is not None for t in self.tasks.values())
            or self.tick >= self.config.max_ticks
        ):
            self.running = False

    def run(self):
        while self.running:
            self.step()
        return self.result()

    def result(self):
        completed = [t for t in self.tasks.values() if t.dropoff_tick is not None]
        vehicles = [v.state for v in self.vehicles]
        done = len(completed)
        ticks = self.tick
        moves = sum(v.moves for v in vehicles)
        costs = sum(v.moves + v.waits + v.handles for v in vehicles)
        counts = [v.completed for v in vehicles]
        load_cv = statistics.pstdev(counts) / statistics.mean(counts) if done else None
        completion_times, pickup_delays, dropoff_ticks = [], [], []
        for task in completed:
            assert task.dropoff_tick is not None and task.pickup_tick is not None
            completion_times.append(task.dropoff_tick - task.spec.created)
            pickup_delays.append(task.pickup_tick - task.spec.created)
            dropoff_ticks.append(task.dropoff_tick)
        mean = statistics.mean
        metrics = {
            "mean_completion_time": mean(completion_times) if done else None,
            "throughput": done / ticks if ticks else 0,
            "unfinished_tasks": len(self.tasks) - done,
            "utilization": sum(v.busy for v in vehicles) / (len(vehicles) * ticks) if ticks else 0,
            "sum_of_costs": costs,
            "communication_volume": sum(self.bus.counts.values()),
            "pickup_delay": mean(pickup_delays) if done else None,
            "makespan": max(dropoff_ticks) - min(t.spec.created for t in self.tasks.values())
            if done == len(self.tasks)
            else None,
            "sum_of_moves": moves,
            "empty_travel_ratio": sum(v.empty_moves for v in vehicles) / moves if moves else None,
            "idle_time": sum(v.idle for v in vehicles),
            "load_balance_cv": load_cv,
            "execution_failures": int(done != len(self.tasks)),
            "deadlock_episodes": self.deadlock_episodes,
            "detected_disruptions": sum(e["observed_tick"] is not None for e in self.disruptions),
            "completed_tasks": done,
            "simulation_ticks": ticks,
            "disruption_proposals": len(self.disruptions),
            "accepted_disruptions": sum(e["accepted"] for e in self.disruptions),
            "skipped_disruptions": sum(not e["accepted"] for e in self.disruptions),
            "replans": self.planner.replans,
            "failed_replans": self.planner.failed_replans,
        }
        if not all(value is None or math.isfinite(value) for value in metrics.values()):
            raise AssertionError("Non-finite experiment metric")
        return {
            "strategy": self.strategy.id,
            "scenario": self.disruption_scenario.id,
            "layout": self.layout.id,
            "seed": self.scenario_seed,
            "config": asdict(self.config),
            "metrics": metrics,
            "tasks": [
                {
                    **asdict(t.spec),
                    "assigned_to": t.assigned_to,
                    "assigned_tick": t.assigned_tick,
                    "pickup_tick": t.pickup_tick,
                    "dropoff_tick": t.dropoff_tick,
                }
                for t in self.tasks.values()
            ],
            "vehicles": [asdict(v) for v in vehicles],
            "disruptions": self.disruptions,
            "messages": {
                "by_performative": dict(self.bus.counts),
                "by_sender": dict(self.bus.by_sender),
            },
            "trace": self.trajectory,
            "message_trace": self.bus.trace,
            "timeseries": self.datacollector.model_vars if self.record_trace else {},
        }
