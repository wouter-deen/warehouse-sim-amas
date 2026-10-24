"""Mesa agents: local valuation, contract-net allocation, and authoritative routing."""

from dataclasses import dataclass
from itertools import combinations

from warehouse_sim.domain import Operation, Plan, RouteStep, TaskSpec, VehicleState, operations_for
from warehouse_sim.messages import Message, ProtocolAgent, Reply
from warehouse_sim.routing import ReservationTable


@dataclass(frozen=True)
class Estimate:
    vehicle: int
    plan: Plan
    operations: tuple[Operation, ...]
    version: int


class CustomerAgent(ProtocolAgent):
    def __init__(self, model, tasks):
        super().__init__(model, "customer")
        self.tasks = tasks
        self.cursor = 0

    def step(self):
        while self.cursor < len(self.tasks) and self.tasks[self.cursor].created <= self.model.tick:
            task = self.tasks[self.cursor]
            self.send("INFORM", "allocator", {"task": task}, f"task:{task.id}")
            self.cursor += 1

    def receive(self, message):
        return None


class VehicleAgent(ProtocolAgent):
    def __init__(self, model, vehicle, cell):
        super().__init__(model, f"agv:{vehicle}")
        self.state = VehicleState(vehicle, cell, model.layout.zone(cell))
        self.plan = Plan(cell, 0, ())
        self.holding = False
        self.handling_done = 0
        self.known_blocked = set()

    def remaining(self):
        result = []
        for task_id in self.state.queue:
            task = self.model.tasks[task_id]
            if task.pickup_tick is None:
                result.append(
                    Operation(task_id, "pickup", task.spec.pickup, self.model.config.pickup_ticks)
                )
            result.append(
                Operation(task_id, "dropoff", task.spec.dropoff, self.model.config.dropoff_ticks)
            )
        if result and self.handling_done:
            first = result[0]
            result[0] = Operation(
                first.task_id, first.kind, first.cell, first.duration - self.handling_done
            )
        return tuple(result)

    def _estimate(self, operations, start=None, start_tick=None):
        if start is None:
            start = self.plan.end
        if start_tick is None:
            start_tick = max(self.model.tick, self.plan.end_tick)
        return self.send(
            "REQUEST",
            "planner",
            {
                "kind": "estimate",
                "vehicle": self.state.id,
                "start": start,
                "start_tick": start_tick,
                "operations": operations,
                "version": self.state.schedule_version,
            },
        )

    def _bid(self, tasks):
        ordered = []
        remaining = list(tasks)
        endpoint = self.plan.end
        available = max(self.model.tick, self.plan.end_tick)
        estimate = None
        # Internal greedy sequencing updates both endpoint and time after every task.
        while remaining:
            options = []
            for task in remaining:
                estimate = self._estimate(
                    operations_for([task], self.model.config), endpoint, available
                )
                if estimate["feasible"]:
                    options.append((estimate["cost"], task.id, task, estimate))
            if not options:
                return None
            _, _, task, estimate = min(options, key=lambda item: item[:2])
            ordered.append(task)
            remaining.remove(task)
            endpoint, available = estimate["end"], estimate["end_tick"]
        # Single tasks already have the required continuous estimate.
        if len(ordered) > 1:
            estimate = self._estimate(operations_for(ordered, self.model.config))
            if not estimate["feasible"]:
                return None
        if estimate is None:
            return None
        delay = max(0, self.plan.end_tick - self.model.tick)
        utility = -estimate["cost"]
        if self.model.strategy.utility == "availability":
            utility -= delay
        return {
            "vehicle": self.state.id,
            "tasks": tuple(t.id for t in ordered),
            "utility": utility,
            "estimate_id": estimate["estimate_id"],
        }

    def receive(self, message: Message):
        content = message.content
        if message.performative == "CFP":
            config, strategy = self.model.config, self.model.strategy
            capacity = config.queue_max - len(self.state.queue)
            if (
                (strategy.commitment == "idle" and self.state.queue)
                or self.holding
                or capacity <= 0
            ):
                return Reply("REFUSE", {"bids": []})
            tasks = [
                t
                for t in content["tasks"]
                if strategy.scope == "global" or self.model.layout.zone(t.pickup) == self.state.zone
            ]
            bids = []
            limit = min(
                capacity, len(tasks), config.bundle_batch if strategy.commitment == "bundle" else 1
            )
            for size in range(1, limit + 1):
                for subset in combinations(tasks, size):
                    bid = self._bid(subset)
                    if bid is not None:
                        bids.append(bid)
            return Reply("PROPOSE" if bids else "REFUSE", {"bids": bids})
        if message.performative == "ACCEPT_PROPOSAL":
            bid = content["bid"]
            confirmation = self.send(
                "REQUEST",
                "planner",
                {
                    "kind": "commit",
                    "estimate_id": bid["estimate_id"],
                    "vehicle": self.state.id,
                    "version": self.state.schedule_version,
                },
                message.conversation,
            )
            if not confirmation["success"]:
                return Reply("FAILURE", {"success": False})
            self.state.queue.extend(bid["tasks"])
            self.plan = confirmation["plan"]
            self.state.schedule_version += 1
            self.holding = False
            # Initial zero-duration services may complete at this tick.
            for event in confirmation["initial_events"]:
                self.complete(event, self.model.tick)
            return Reply("INFORM", {"success": True})
        if message.performative == "REQUEST" and content["kind"] == "remaining":
            return Reply(
                "INFORM",
                {
                    "operations": self.remaining(),
                    "cell": self.state.cell,
                    "queue": tuple(self.state.queue),
                    "active_handling": self.handling_done > 0,
                },
            )
        if message.performative == "INFORM" and content["kind"] == "map":
            self.known_blocked = set(content["blocked"])
        if message.performative == "INFORM" and content["kind"] == "schedule":
            self.plan = content["plan"]
            self.holding = content["holding"]
            self.state.schedule_version += 1
            for event in content.get("initial_events", ()):
                self.complete(event, self.model.tick)
        return None

    def observe(self):
        nearby = {
            c
            for c in self.model.layout.disruption_cells
            if abs(c[0] - self.state.cell[0]) + abs(c[1] - self.state.cell[1]) <= 1
        }
        active = set(self.model.active_blockages)
        discovered = (nearby & active) - self.known_blocked
        cleared = (nearby & self.known_blocked) - active
        if discovered or cleared:
            self.send(
                "INFORM",
                "planner",
                {"kind": "observation", "blocked": discovered, "cleared": cleared},
            )

    def complete(self, event, tick):
        task_id, kind = event
        record = self.model.tasks[task_id]
        self.handling_done = 0
        if kind == "pickup":
            if record.pickup_tick is not None:
                return
            record.pickup_tick = tick
            self.state.carrying = task_id
        else:
            if record.dropoff_tick is not None:
                return
            record.dropoff_tick = tick
            self.state.carrying = None
            self.state.queue.remove(task_id)
            self.state.completed += 1
            self.send(
                "INFORM",
                "allocator",
                {"done": task_id, "completion_tick": tick},
                f"task:{task_id}",
                tick=tick,
            )

    def step(self):
        now = self.model.tick
        index = now - self.plan.start_tick
        step = self.plan.steps[index] if 0 <= index < len(self.plan.steps) else None
        busy = bool(self.state.queue)
        if busy:
            self.state.busy += 1
        else:
            self.state.idle += 1
        if step is None:
            if busy:
                self.state.waits += 1
            return
        if step.action == "move":
            self.state.moves += 1
            if self.state.carrying is None:
                self.state.empty_moves += 1
        elif step.action == "handle":
            self.state.handles += 1
            self.handling_done += 1
        elif busy:
            self.state.waits += 1
        self.state.cell = step.cell
        for event in step.completed:
            self.complete(event, now + 1)


class RoutePlannerAgent(ProtocolAgent):
    def __init__(self, model, starts):
        super().__init__(model, "planner")
        self.table = ReservationTable(model.layout, starts)
        self.known_blocked = set()
        self.estimates = {}
        self.next_estimate = 0
        self.changed = False
        self.replans = 0
        self.failed_replans = 0
        self.held = set()
        self.monitors = {}

    def receive(self, message):
        content = message.content
        if message.performative == "INFORM" and content["kind"] == "observation":
            for cell in content["blocked"]:
                self.model.mark_observation(cell)
            for cell in content["blocked"]:
                self.monitors.setdefault(cell, int(message.sender.split(":")[1]))
            for cell in content["cleared"]:
                self.monitors.pop(cell, None)
            self.known_blocked.update(content["blocked"])
            self.known_blocked.difference_update(content["cleared"])
            for vehicle in sorted(self.model.vehicle_roster):
                self.send(
                    "INFORM",
                    f"agv:{vehicle}",
                    {"kind": "map", "blocked": frozenset(self.known_blocked)},
                )
            self.changed = True
            return None
        if content["kind"] == "estimate":
            plan = self.table.search(
                content["vehicle"],
                content["start"],
                content["start_tick"],
                content["operations"],
                self.known_blocked,
                self.model.config.planning_horizon,
            )
            if plan is None:
                return Reply("FAILURE", {"feasible": False})
            key = self.next_estimate
            self.next_estimate += 1
            self.estimates[key] = Estimate(
                content["vehicle"], plan, content["operations"], content["version"]
            )
            return Reply(
                "INFORM",
                {
                    "feasible": True,
                    "cost": plan.cost,
                    "estimate_id": key,
                    "end": plan.end,
                    "end_tick": plan.end_tick,
                },
            )
        if content["kind"] == "commit":
            estimate = self.estimates.get(content["estimate_id"])
            vehicle, now = content["vehicle"], self.model.tick
            if (
                estimate is None
                or estimate.vehicle != vehicle
                or estimate.version != content["version"]
            ):
                return Reply("FAILURE", {"success": False})
            end, available = self.table.available(vehicle)
            available = max(now, available)
            plan = estimate.plan
            if plan.start != end or plan.start_tick != available:
                return Reply("FAILURE", {"success": False})
            if any(step.cell in self.known_blocked for step in plan.steps) or not self.table.commit(
                vehicle, plan, now
            ):
                plan = self.table.search(
                    vehicle,
                    end,
                    available,
                    estimate.operations,
                    self.known_blocked,
                    self.model.config.planning_horizon,
                )
                if plan is None or not self.table.commit(vehicle, plan, now):
                    return Reply("FAILURE", {"success": False})
            self.changed = True
            return Reply(
                "INFORM",
                {
                    "success": True,
                    "plan": self.table.plans[vehicle],
                    "initial_events": plan.initial_events if available == now else (),
                },
            )
        raise ValueError(f"Unexpected planner message {message}")

    def replan(self):
        now = self.model.tick
        # Known obstructions affect entire future queues, not only the next edge.
        affected = {
            v
            for v, plan in self.table.plans.items()
            if any(s.tick > now and s.cell in self.known_blocked for s in plan.steps)
        }
        watching = set(self.monitors.values())
        if self.changed or now % self.model.config.auction_interval == 0:
            affected.update(self.held - watching)
        if self.changed:
            affected.update(watching)
        affected.update(watching - self.held)
        if not affected:
            self.changed = False
            return
        # Holding a disrupted vehicle may invalidate an otherwise unaffected route.
        while True:
            cells = {self.table.position(v, now) for v in affected}
            extra = {
                v
                for v, plan in self.table.plans.items()
                if v not in affected and any(s.tick > now and s.cell in cells for s in plan.steps)
            }
            if not extra:
                break
            affected.update(extra)
        staged = self.table.clone()
        for vehicle in sorted(affected):
            staged.hold(vehicle, self.table.position(vehicle, now), now)
        messages = {}
        for vehicle in sorted(affected):
            state = self.send("REQUEST", f"agv:{vehicle}", {"kind": "remaining"})
            self.send("INFORM", f"agv:{vehicle}", {"kind": "hold"})
            operations = state["operations"]
            start, start_tick = state["cell"], now
            # A downstream closure cannot interrupt already-started handling.
            if state["active_handling"] and operations:
                operation = operations[0]
                prefix = Plan(
                    start,
                    now,
                    tuple(
                        RouteStep(
                            now + i + 1,
                            start,
                            "handle",
                            ((operation.task_id, operation.kind),)
                            if i + 1 == operation.duration
                            else (),
                        )
                        for i in range(operation.duration)
                    ),
                )
                if not staged.commit(vehicle, prefix, now):
                    raise AssertionError("Mandatory handling prefix was not reservable")
                start_tick = prefix.end_tick
                operations = operations[1:]
            plan = (
                None
                if vehicle in watching
                else staged.search(
                    vehicle,
                    start,
                    start_tick,
                    operations,
                    self.known_blocked,
                    self.model.config.planning_horizon,
                )
            )
            holding = plan is None
            if plan is not None and not staged.commit(vehicle, plan, now):
                raise AssertionError("A route conflicted with its own staged reservation snapshot")
            self.replans += 1
            self.failed_replans += int(holding and vehicle not in watching)
            if holding:
                self.held.add(vehicle)
            else:
                self.held.discard(vehicle)
            messages[vehicle] = (holding, plan.initial_events if plan and start_tick == now else ())
        if not self.table.replace_many(staged.plans, now):
            raise AssertionError("Coordinated reservation update was not collision safe")
        for vehicle, (holding, initial_events) in messages.items():
            self.send(
                "INFORM",
                f"agv:{vehicle}",
                {
                    "kind": "schedule",
                    "plan": self.table.plans[vehicle],
                    "holding": holding,
                    "initial_events": initial_events,
                    "activation_tick": now,
                },
            )
        self.changed = False


class AllocatorAgent(ProtocolAgent):
    def __init__(self, model):
        super().__init__(model, "allocator")
        self.pending: dict[int, TaskSpec] = {}

    def receive(self, message):
        if "task" in message.content:
            task = message.content["task"]
            self.pending[task.id] = task
        return None

    def _auction(self, tasks):
        all_bids = {}
        strategy = self.model.strategy
        for vehicle, zone in sorted(self.model.vehicle_roster.items()):
            if strategy.scope == "zone" and not any(
                self.model.layout.zone(t.pickup) == zone for t in tasks
            ):
                continue
            result = self.send(
                "CFP",
                f"agv:{vehicle}",
                {"tasks": tasks},
                "auction:" + ",".join(str(t.id) for t in tasks),
            )
            if result["bids"]:
                all_bids[vehicle] = result["bids"]
        if not all_bids:
            return 0
        # Small-batch exact set packing over submitted (greedily sequenced) bids.
        # Maximal coverage precedes utility, so negative costs cannot prefer no work.
        best_key, best_assignment = None, []
        vehicles = sorted(all_bids)

        def visit(index, covered, utility, assignment):
            nonlocal best_key, best_assignment
            if index == len(vehicles):
                tie = tuple((b["vehicle"], tuple(sorted(b["tasks"]))) for b in assignment)
                key = (-len(covered), -utility, tie)
                if best_key is None or key < best_key:
                    best_key, best_assignment = key, list(assignment)
                return
            visit(index + 1, covered, utility, assignment)
            for bid in all_bids[vehicles[index]]:
                ids = set(bid["tasks"])
                if not covered & ids:
                    visit(index + 1, covered | ids, utility + bid["utility"], assignment + [bid])

        visit(0, set(), 0, [])
        winners = {b["vehicle"]: b for b in best_assignment}
        for vehicle, bids in all_bids.items():
            if vehicle not in winners:
                self.send("REJECT_PROPOSAL", f"agv:{vehicle}", {"bids": bids})
        assigned = 0
        for bid in best_assignment:
            # Set assignment metadata before zero-duration completion events are delivered.
            for task_id in bid["tasks"]:
                record = self.model.tasks[task_id]
                record.assigned_to, record.assigned_tick = bid["vehicle"], self.model.tick
            response = self.send("ACCEPT_PROPOSAL", f"agv:{bid['vehicle']}", {"bid": bid})
            if response["success"]:
                for task_id in bid["tasks"]:
                    self.pending.pop(task_id)
                    assigned += 1
            else:
                for task_id in bid["tasks"]:
                    record = self.model.tasks[task_id]
                    record.assigned_to = record.assigned_tick = None
        return assigned

    def step(self):
        if self.model.tick % self.model.config.auction_interval:
            return
        pending = sorted(self.pending.values(), key=lambda t: (t.created, t.id))
        size = self.model.config.bundle_batch if self.model.strategy.commitment == "bundle" else 1
        for offset in range(0, len(pending), size):
            self._auction(pending[offset : offset + size])
