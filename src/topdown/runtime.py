"""Non-graphical helpers for running the simulation loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, Set

from .simulation import Simulation
from .state import SimulationSnapshot
from .track import Track


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters controlling Box2D stepping."""

    time_step: float = 1.0 / 60.0
    velocity_iterations: int = 6
    position_iterations: int = 2


class SimulationSession:
    """Wraps a Simulation with step scheduling useful for Gym integration."""

    def __init__(
        self,
        track: Track | None = None,
        *,
        config: SimulationConfig | None = None,
        on_best_lap: Callable[[float, Sequence[float]], None] | None = None,
    ) -> None:
        self._config = config or SimulationConfig()
        self._base_track = track
        self._on_best_lap = on_best_lap
        self._elapsed_time = 0.0
        self._step_index = 0
        self._simulation = Simulation(track=track, on_best_lap=on_best_lap)

    @property
    def config(self) -> SimulationConfig:
        return self._config

    @property
    def simulation(self) -> Simulation:
        return self._simulation

    def reset(self, track: Track | None = None) -> SimulationSnapshot:
        """Create a fresh simulation, optionally swapping the active track."""
        active_track = track or self._base_track
        self._simulation = Simulation(track=active_track, on_best_lap=self._on_best_lap)
        self._base_track = active_track
        self._elapsed_time = 0.0
        self._step_index = 0
        return self.snapshot()

    def step(
        self,
        controls: Iterable[str] | None = None,
        *,
        time_step: float | None = None,
        velocity_iterations: int | None = None,
        position_iterations: int | None = None,
    ) -> SimulationSnapshot:
        """Advance the simulation using the provided control inputs."""
        keys: Set[str] = set(controls or ())

        step_dt = self._config.time_step if time_step is None else time_step
        vel_iters = self._config.velocity_iterations if velocity_iterations is None else velocity_iterations
        pos_iters = self._config.position_iterations if position_iterations is None else position_iterations

        self._simulation.step(step_dt, vel_iters, pos_iters, keys)
        self._elapsed_time += step_dt
        self._step_index += 1
        return self.snapshot()

    def snapshot(self) -> SimulationSnapshot:
        """Return the current state without advancing the world."""
        return self._simulation.snapshot(
            elapsed_time=self._elapsed_time,
            step_index=self._step_index,
        )


__all__ = ["SimulationConfig", "SimulationSession"]
