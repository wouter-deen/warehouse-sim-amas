"""Interactive visualization of one warehouse simulation run.

Run from the repository root with ``uv run python demo.py``. Change the study
parameters below to explore different strategies, disruption scenarios, and layouts.
"""

from __future__ import annotations

from math import atan2, degrees

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
from matplotlib.widgets import Button

from warehouse_sim.domain import Config
from warehouse_sim.model import WarehouseModel


class CanvasSafeAnimation(FuncAnimation):
    """Stop a stale blitted timer if its GUI canvas has already been replaced."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._blit_canvas = self._fig.canvas # type: ignore

    def _step(self, *args) -> bool:
        canvas = self._fig.canvas # type: ignore
        blit_methods = ("restore_region", "copy_from_bbox", "blit")
        if self._blit and (
            canvas is not self._blit_canvas
            or not all(callable(getattr(canvas, name, None)) for name in blit_methods)
        ):
            self._blit = False
            if self.event_source is not None:
                self.event_source.stop()
            return False
        return super()._step(*args) # type: ignore


# Experiment dimensions
STRATEGY = "S8"
SCENARIO = "D4"
LAYOUT = "A"
SEED = 0

# Simulation configuration (same fields as configs/study.json)
VEHICLES = 4
TOTAL_TASKS = 48
ARRIVAL_INTERVAL = 8.0
MAX_TICKS = 1600
PLANNING_HORIZON = 192
QUEUE_MAX = 3
PICKUP_TICKS = 1
DROPOFF_TICKS = 1
AUCTION_INTERVAL = 4
BUNDLE_BATCH = 3
DEADLOCK_WINDOW = 100

# Display configuration
FRAME_INTERVAL_MS = 100
START_PAUSED = False


def make_config() -> Config:
    return Config(
        vehicles=VEHICLES,
        total_tasks=TOTAL_TASKS,
        arrival_interval=ARRIVAL_INTERVAL,
        max_ticks=MAX_TICKS,
        planning_horizon=PLANNING_HORIZON,
        queue_max=QUEUE_MAX,
        pickup_ticks=PICKUP_TICKS,
        dropoff_ticks=DROPOFF_TICKS,
        auction_interval=AUCTION_INTERVAL,
        bundle_batch=BUNDLE_BATCH,
        deadlock_window=DEADLOCK_WINDOW,
    )


class WarehouseDemo:
    def __init__(self) -> None:
        self.config = make_config()
        self.playing = not START_PAUSED
        self.model = self._new_model()
        self.layout = self.model.layout
        self.colors = plt.get_cmap("tab10").colors # type: ignore

        self.figure, self.axes = plt.subplots(figsize=(10, 8))
        self.figure.subplots_adjust(bottom=0.17, right=0.82)
        self.routes_visible = False
        self._draw_layout()
        self._draw_dynamic_state()
        self._add_controls()
        self.animation = CanvasSafeAnimation(
            self.figure,
            self._update,
            init_func=self._init_animation,
            interval=FRAME_INTERVAL_MS,
            blit=True,
            cache_frame_data=False,
        )

    def _new_model(self) -> WarehouseModel:
        return WarehouseModel(
            config=self.config,
            strategy=STRATEGY,
            scenario=SCENARIO,
            layout=LAYOUT,
            seed=SEED,
        )

    def _draw_layout(self) -> None:
        grid = [
            [0 if (x, y) in self.layout.cells else 1 for x in range(self.layout.width)]
            for y in range(self.layout.height)
        ]
        self.axes.imshow(
            grid,
            origin="lower",
            cmap=ListedColormap(("#f4f4ed", "#555b63")),
            vmin=0,
            vmax=1,
            interpolation="none",
        )
        for x, y in self.layout.pickups:
            self.axes.add_patch(
                Rectangle(
                    (x - 0.43, y - 0.43),
                    0.86,
                    0.86,
                    facecolor="#e8f5e9",
                    edgecolor="#388e3c",
                    linewidth=1.2,
                    zorder=3,
                )
            )
            self.axes.add_patch(
                Rectangle(
                    (x - 0.33, y - 0.27),
                    0.66,
                    0.54,
                    facecolor="#a5d6a7",
                    edgecolor="#355e3b",
                    linewidth=1.5,
                    zorder=4,
                )
            )
            for bay_x in (x - 0.22, x, x + 0.22):
                self.axes.add_patch(
                    Rectangle(
                        (bay_x - 0.08, y - 0.17),
                        0.16,
                        0.34,
                        facecolor="#dcedc8",
                        edgecolor="#558b2f",
                        linewidth=1,
                        zorder=5,
                    )
                )
                self.axes.add_patch(
                    Rectangle(
                        (bay_x - 0.04, y + 0.08),
                        0.08,
                        0.05,
                        facecolor="#fff8e1",
                        edgecolor="#8d6e63",
                        linewidth=0.6,
                        zorder=6,
                    )
                )

        for x, y in self.layout.dropoffs:
            self.axes.add_patch(
                Rectangle(
                    (x - 0.43, y - 0.43),
                    0.86,
                    0.86,
                    facecolor="#e3f2fd",
                    edgecolor="#1565c0",
                    linewidth=1.5,
                    zorder=3,
                )
            )
            self.axes.add_patch(
                Rectangle(
                    (x - 0.27, y - 0.27),
                    0.54,
                    0.54,
                    facecolor="#bbdefb",
                    edgecolor="#1976d2",
                    linewidth=1.2,
                    zorder=4,
                )
            )
            self.axes.add_patch(
                Polygon(
                    ((x, y - 0.14), (x - 0.11, y + 0.02), (x + 0.11, y + 0.02)),
                    closed=True,
                    facecolor="#1565c0",
                    edgecolor="#1565c0",
                    zorder=5,
                )
            )
            self.axes.add_patch(
                Rectangle(
                    (x - 0.045, y + 0.02),
                    0.09,
                    0.18,
                    facecolor="#1565c0",
                    edgecolor="#1565c0",
                    zorder=5,
                )
            )
        self.axes.set(
            xlim=(-0.5, self.layout.width - 0.5),
            ylim=(-0.5, self.layout.height - 0.5),
            xticks=range(self.layout.width),
            yticks=range(self.layout.height),
            xlabel="Warehouse X",
            ylabel="Warehouse Y",
            title=f"Layout {LAYOUT} | {STRATEGY} | {SCENARIO} | seed {SEED}",
            aspect="equal",
        )
        self.axes.set_xticks([x - 0.5 for x in range(self.layout.width + 1)], minor=True)
        self.axes.set_yticks([y - 0.5 for y in range(self.layout.height + 1)], minor=True)
        self.axes.grid(which="minor", color="#a6a6a0", linewidth=0.5)
        self.axes.tick_params(which="minor", bottom=False, left=False)

        self.blockage_artist = self.axes.scatter(
            [],
            [],
            marker="X",
            color="#ef6c00",
            s=130,
            label="Active blockage",
            zorder=5,
        )
        self.vehicle_artists: list[
            tuple[Rectangle, tuple[Rectangle, ...], Rectangle, Rectangle]
        ] = []
        for vehicle_id in range(self.config.vehicles):
            color = self.colors[vehicle_id % len(self.colors)]
            body = Rectangle(
                (-0.29, -0.18),
                0.58,
                0.36,
                facecolor=color,
                edgecolor="#263238",
                linewidth=1.2,
                zorder=7,
            )
            wheels = tuple(
                Rectangle(
                    (wheel_x, wheel_y),
                    0.13,
                    0.1,
                    facecolor="#263238",
                    edgecolor="#111820",
                    linewidth=0.5,
                    zorder=6,
                )
                for wheel_x, wheel_y in (
                    (-0.25, -0.24),
                    (-0.25, 0.14),
                    (0.12, -0.24),
                    (0.12, 0.14),
                )
            )
            nose = Rectangle(
                (0.22, -0.065),
                0.1,
                0.13,
                facecolor="#fffde7",
                edgecolor="#263238",
                linewidth=0.6,
                zorder=8,
            )
            cargo = Rectangle(
                (-0.11, -0.1),
                0.22,
                0.2,
                facecolor="#ffca28",
                edgecolor="#5d4037",
                linewidth=1,
                zorder=9,
                visible=False,
            )
            for artist in (*wheels, body, nose, cargo):
                self.axes.add_patch(artist)
            self.vehicle_artists.append((body, wheels, nose, cargo))
        self.route_artists = [
            self.axes.plot(
                [],
                [],
                color=self.colors[vehicle_id % len(self.colors)],
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                label=f"AGV {vehicle_id} route",
                zorder=4,
            )[0]
            for vehicle_id in range(self.config.vehicles)
        ]
        self.vehicle_labels = [
            self.axes.text(0, 0, "", ha="center", va="center", fontsize=8, zorder=7)
            for _ in range(self.config.vehicles)
        ]
        legend_handles = [
            Patch(facecolor="#a5d6a7", edgecolor="#355e3b", label="Pickup rack"),
            Patch(facecolor="#bbdefb", edgecolor="#1565c0", label="Dropoff square"),
            Patch(facecolor="#ffca28", edgecolor="#5d4037", label="Carried box"),
            Line2D([], [], marker="X", color="#ef6c00", linestyle="None", label="Active blockage"),
        ]
        for vehicle_id in range(self.config.vehicles):
            color = self.colors[vehicle_id % len(self.colors)]
            legend_handles.extend(
                (
                    Patch(
                        facecolor=color,
                        edgecolor="#263238",
                        label=f"AGV {vehicle_id}",
                    ),
                    Line2D(
                        [],
                        [],
                        color=color,
                        linestyle="--",
                        linewidth=2,
                        label=f"AGV {vehicle_id} route",
                    ),
                )
            )
        self.axes.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1))
        self.status_axes = self.figure.add_axes((0.20, 0.095, 0.64, 0.035))
        self.status_axes.set_axis_off()
        self.status = self.status_axes.text(
            0.01,
            0.5,
            "",
            ha="left",
            va="center",
            transform=self.status_axes.transAxes,
            clip_on=True,
        )

    def _draw_dynamic_state(self) -> tuple[Artist, ...]:
        artists: list[Artist] = [self.blockage_artist, self.status]
        for vehicle, (body, wheels, nose, cargo), label in zip(
            self.model.vehicles,
            self.vehicle_artists,
            self.vehicle_labels,
            strict=True,
        ):
            x, y = vehicle.state.cell
            next_step = next(
                (step for step in vehicle.plan.steps if step.tick > self.model.tick), None
            )
            next_cell = next_step.cell if next_step is not None else (x, y)
            angle = degrees(atan2(next_cell[1] - y, next_cell[0] - x))
            transform = Affine2D().rotate_deg(angle).translate(x, y) + self.axes.transData
            for artist in (*wheels, body, nose, cargo):
                artist.set_transform(transform)
                artists.append(artist)
            cargo.set_visible(vehicle.state.carrying is not None)

            label.set_position((x, y))
            carried = vehicle.state.carrying
            label.set_text(f"{vehicle.state.id}" + (f":T{carried}" if carried is not None else ""))
            artists.append(label)
        for vehicle, route in zip(self.model.vehicles, self.route_artists, strict=True):
            if self.routes_visible:
                future_cells = [
                    step.cell for step in vehicle.plan.steps if step.tick > self.model.tick
                ]
                route_cells = [vehicle.state.cell, *future_cells]
                route.set_data(
                    [cell[0] for cell in route_cells],
                    [cell[1] for cell in route_cells],
                )
            route.set_visible(self.routes_visible)
            artists.append(route)

        blocked = list(self.model.active_blockages)
        self.blockage_artist.set_offsets(blocked if blocked else [[-1, -1]])
        completed = sum(task.dropoff_tick is not None for task in self.model.tasks.values())
        state = "Running" if self.model.running and self.playing else "Paused"
        if not self.model.running:
            state = "Finished"
        self.status.set_text(
            f"{state} | tick {self.model.tick}/{self.config.max_ticks} | "
            f"tasks {completed}/{self.config.total_tasks} | "
            f"active blockages {len(blocked)} | replans {self.model.planner.replans}"
        )
        return tuple(artists)

    def _add_controls(self) -> None:
        play_axes = self.figure.add_axes((0.20, 0.025, 0.13, 0.05))
        step_axes = self.figure.add_axes((0.36, 0.025, 0.13, 0.05))
        reset_axes = self.figure.add_axes((0.52, 0.025, 0.13, 0.05))
        routes_axes = self.figure.add_axes((0.68, 0.025, 0.16, 0.05))
        self.play_button = Button(play_axes, "Pause" if self.playing else "Play")
        self.play_button.on_clicked(self._toggle_play)
        self.step_button = Button(step_axes, "Step")
        self.step_button.on_clicked(self._step)
        self.reset_button = Button(reset_axes, "Reset")
        self.reset_button.on_clicked(self._reset)
        self.routes_button = Button(routes_axes, "Show routes")
        self.routes_button.on_clicked(self._toggle_routes)

    def _toggle_routes(self, _event) -> None:
        self.routes_visible = not self.routes_visible
        self.routes_button.label.set_text("Hide routes" if self.routes_visible else "Show routes")
        self._draw_dynamic_state()
        self.figure.canvas.draw_idle()

    def _init_animation(self) -> tuple[Artist, ...]:
        return self._draw_dynamic_state()

    def _update(self, _frame: int) -> tuple[Artist, ...]:
        if self.playing and self.model.running:
            self.model.step()
            if not self.model.running:
                self.playing = False
                self.play_button.label.set_text("Play")
        return self._draw_dynamic_state()

    def _toggle_play(self, _event) -> None:
        if not self.model.running:
            self._reset(None)
            self.playing = True
            self.play_button.label.set_text("Pause")
        else:
            self.playing = not self.playing
            self.play_button.label.set_text("Pause" if self.playing else "Play")
        self.figure.canvas.draw_idle()

    def _step(self, _event) -> None:
        self.playing = False
        self.play_button.label.set_text("Play")
        if self.model.running:
            self.model.step()
        self._draw_dynamic_state()
        self.figure.canvas.draw_idle()

    def _reset(self, _event) -> None:
        self.model = self._new_model()
        self.playing = not START_PAUSED
        self.play_button.label.set_text("Pause" if self.playing else "Play")
        self._draw_dynamic_state()
        self.figure.canvas.draw_idle()

    def show(self) -> None:
        plt.show()


def main() -> None:
    WarehouseDemo().show()


if __name__ == "__main__":
    main()
