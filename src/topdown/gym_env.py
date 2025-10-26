"""Gymnasium environment wrapper for the top-down car simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pygame
import gymnasium as gym

from .control import DiscreteControl, control_to_keys, enumerate_controls
from .render import RenderConfig, Renderer
from .simulation import Simulation
from .sensors import RaycastConfig
from .track import Track
from .track_loader import discover_tracks, load_track


DEFAULT_TIME_STEP = 1.0 / 60.0
DEFAULT_VELOCITY_ITERATIONS = 6
DEFAULT_POSITION_ITERATIONS = 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_track_from_sources(
    track: Track | None,
    *,
    track_file: str | Path | None,
    track_name: str | None,
    tracks_dir: str | Path | None,
) -> tuple[Track, Path | None]:
    if track is not None:
        return track, Path(track_file) if track_file is not None else None

    if track_file is not None:
        path = Path(track_file)
        return load_track(path), path

    search_dirs: list[Path] = []
    if tracks_dir is not None:
        search_dirs.append(Path(tracks_dir))
    search_dirs.append(_project_root() / "tracks")

    if track_name:
        candidate_path = Path(track_name)
        if candidate_path.exists():
            return load_track(candidate_path), candidate_path

        for directory in search_dirs:
            if not directory.exists():
                continue
            tracks = discover_tracks(directory)
            if track_name in tracks:
                loaded = tracks[track_name]
                return loaded.track, loaded.path
            candidate = directory / f"{track_name}.json"
            if candidate.exists():
                return load_track(candidate), candidate

    for directory in search_dirs:
        if not directory.exists():
            continue
        tracks = discover_tracks(directory)
        if tracks:
            first_loaded = next(iter(tracks.values()))
            return first_loaded.track, first_loaded.path

    default_path = _project_root() / "tracks" / "test_track.json"
    if default_path.exists():
        return load_track(default_path), default_path
    raise FileNotFoundError(
        "Unable to locate a track definition. Provide 'track', 'track_file', or 'track_name'."
    )


ObservationBuilder = Callable[[Simulation], np.ndarray]
RewardFunction = Callable[[Simulation, dict], float]


@dataclass(frozen=True)
class EnvironmentConfig:
    """Configuration bundle for the Gymnasium environment."""

    time_step: float = DEFAULT_TIME_STEP
    velocity_iterations: int = DEFAULT_VELOCITY_ITERATIONS
    position_iterations: int = DEFAULT_POSITION_ITERATIONS
    max_episode_steps: int = 5000
    frame_skip: int = 1
    speed_limit: float = 100.0
    steering_limit: float = np.deg2rad(40.0)
    show_sensors: bool = False
    sensor_config: RaycastConfig | None = None


class TopdownCarEnv(gym.Env):
    """Gymnasium-compatible wrapper around the Simulation class."""

    metadata = {
        "render_modes": ("none", "human", "rgb_array"),
        "render_fps": int(1.0 / DEFAULT_TIME_STEP),
    }

    def __init__(
        self,
        *,
        track: Track | None = None,
        track_file: str | Path | None = None,
        track_name: str | None = None,
        tracks_dir: str | Path | None = None,
        render_mode: str | None = None,
        action_controls: Sequence[DiscreteControl] | None = None,
        sensor_config: RaycastConfig | None = None,
        reward_fn: RewardFunction | None = None,
        env_config: EnvironmentConfig | None = None,
        observation_builder: ObservationBuilder | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode or "none"
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{self.render_mode}'")

        self._env_config = env_config or EnvironmentConfig()

        loaded_track, resolved_file = _load_track_from_sources(
            track,
            track_file=track_file,
            track_name=track_name,
            tracks_dir=tracks_dir,
        )
        self._track_file = resolved_file

        sensor_cfg = sensor_config or self._env_config.sensor_config
        self._simulation = Simulation(
            track=loaded_track,
            track_file=resolved_file,
            sensor_config=sensor_cfg,
        )

        self._controls = (
            list(action_controls)
            if action_controls is not None
            else enumerate_controls()
        )
        if not self._controls:
            raise ValueError("action_controls must contain at least one control option")

        self.action_space = gym.spaces.Discrete(len(self._controls))

        sensor_count = len(self._simulation.sensor_angles())
        observation_dim = sensor_count + 2

        low = np.zeros(observation_dim, dtype=np.float32)
        high = np.ones(observation_dim, dtype=np.float32)
        low[-1] = -1.0
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self._reward_fn = reward_fn or self._default_reward
        self._observation_builder = (
            observation_builder
            if observation_builder is not None
            else self._build_observation
        )

        self._renderer: Renderer | None = None
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None

        self._step_count = 0
        self._episode_terminated = False
        self._last_info: dict = {}

        self.seed()

    def seed(self, seed: int | None = None) -> None:
        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)

        self._simulation.reset()
        self._step_count = 0
        self._episode_terminated = False
        self._last_info = self._gather_info()

        observation = self._observation_builder(self._simulation)

        return observation, dict(self._last_info)

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(
                f"Action {action!r} is outside {self.action_space}"
            )

        if self._episode_terminated:
            raise gym.error.ResetNeeded(
                "Cannot call step() on a terminated episode. Call reset() first."
            )

        control = self._controls[action]
        keys = control_to_keys(control)

        for _ in range(self._env_config.frame_skip):
            self._simulation.step(
                self._env_config.time_step,
                self._env_config.velocity_iterations,
                self._env_config.position_iterations,
                keys,
            )

        self._step_count += 1

        observation = self._observation_builder(self._simulation)
        info = self._gather_info()
        reward = self._reward_fn(self._simulation, info)

        timing_state = self._simulation.timing_state
        terminated = bool(timing_state and timing_state.invalid)
        truncated = self._step_count >= self._env_config.max_episode_steps
        self._episode_terminated = terminated or truncated

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "none":
            return None

        self._ensure_renderer()
        assert self._screen is not None
        assert self._renderer is not None

        self._renderer.draw(self._simulation)

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self._screen)
            return np.transpose(frame, (1, 0, 2))
        elif self.render_mode == "human":
            pygame.event.pump()
        return None

    def close(self) -> None:
        if self._renderer is not None and self.render_mode == "human":
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()
        self._renderer = None
        self._screen = None
        self._clock = None

    def _ensure_renderer(self) -> None:
        if self._renderer is not None:
            return

        if not pygame.get_init():
            pygame.init()

        render_config = RenderConfig(draw_sensor_rays=self._env_config.show_sensors)

        if self.render_mode == "human":
            self._screen = pygame.display.set_mode(render_config.screen_size)
            pygame.display.set_caption("Topdown Car - Gymnasium")
            self._clock = pygame.time.Clock()
        else:
            self._screen = pygame.Surface(render_config.screen_size)

        self._renderer = Renderer(self._screen, render_config)

    def _build_observation(self, simulation: Simulation) -> np.ndarray:
        sensor_sample = simulation.sample_sensors()
        distances = np.array(
            sensor_sample.normalized(simulation.sensor_max_distance()), dtype=np.float32
        )
        speed = min(simulation.car_speed() / self._env_config.speed_limit, 1.0)
        steering = simulation.front_wheel_angle() / self._env_config.steering_limit
        steering = np.clip(steering, -1.0, 1.0)
        return np.concatenate(
            [
                distances,
                np.array([speed, steering], dtype=np.float32),
            ]
        ).astype(np.float32)

    def _gather_info(self) -> dict:
        timing_state = self._simulation.timing_state
        info = {
            "speed": self._simulation.car_speed(),
            "front_wheel_angle": self._simulation.front_wheel_angle(),
            "timing_state": timing_state,
        }
        if timing_state is not None:
            info.update(
                {
                    "best_lap_time": timing_state.best_lap_time,
                    "last_lap_time": timing_state.last_lap_time,
                }
            )
        return info

    def _default_reward(self, simulation: Simulation, info: dict) -> float:
        speed = simulation.car_speed() / max(self._env_config.speed_limit, 1e-6)
        penalty = 0.0
        if info.get("timing_state") and info["timing_state"].invalid:
            penalty = -1.0
        return float(speed - penalty)

    def _get_info(self) -> dict:
        """Gymnasium hook: provide extra debugging information to wrappers."""
        info = dict(self._last_info)
        info.update(
            {
                "step_count": self._step_count,
                "track_file": str(self._track_file) if self._track_file else None,
                "sensor_distances": tuple(self._simulation.sample_sensors().distances),
            }
        )
        return info


__all__ = ["TopdownCarEnv", "EnvironmentConfig"]
