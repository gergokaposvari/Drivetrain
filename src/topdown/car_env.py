from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from mpmath import cos

from src.topdown import LoadedTrack, load_track
from src.topdown.input import InputHandler
from src.topdown.render import RenderConfig, Renderer
from src.topdown.simulation import Simulation

# (Unused imports removed: Segment)
from src.topdown.timing import TimingState

TIME_STEP = 1.0 / 60.0
VELOCITY_ITERATIONS = 6
POSITION_ITERATIONS = 2


class CarEnv(gym.Env):
    """Minimal Gym environment wrapper for the top-down car simulation.

    Rendering logic kept simple: we eagerly create the display surface once
    and reuse a single Renderer. No lazy indirection like _ensure_renderer.
    """

    metadata = {
        "render_modes": ("human", "rgb_array"),
        "render_fps": int(1.0 / TIME_STEP),
    }

    def __init__(self, render_mode: str | None = "human"):
        super().__init__()

        actual_track = load_track(
            Path(__file__).parent.parent.parent / "tracks" / "simple.json"
        )
        test_track = LoadedTrack(
            name="Test-track",
            path=Path(__file__).parent / "test_track.json",
            track=actual_track,
        )

        if render_mode is None:
            render_mode = "human"
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode '{render_mode}'")
        self.render_mode = render_mode

        pygame.init()

        config = RenderConfig(draw_sensor_rays=False)
        self._display_active = self.render_mode == "human"

        if self._display_active:
            screen = pygame.display.set_mode(config.screen_size)
            pygame.display.set_caption("Top Down Car")
            self.clock = pygame.time.Clock()
        else:
            screen = pygame.Surface(config.screen_size)
            self.clock = None

        self.input_handler = InputHandler()
        self.simulation = Simulation(
            track=test_track.track if test_track else None,
            track_file=test_track.path if test_track else None,
        )
        self.screen = screen
        self.renderer = Renderer(self.screen, config)

        low = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                -20,
                -0.7,
                -1,
                -1,
                0,
            ],
            dtype=np.float32,
        )
        high = np.array(
            [
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                150,
                90,
                0.7,
                1,
                1,
                4,
            ],
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(15,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1, 0.2]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32,
        )

        self._last_info: dict | None = None

    def _get_obs(self):
        tires_on_grass = sum(
            1 for tire in self.simulation.car.tires if tire.is_fully_on_grass
        )
        unit_car_to_sector = self.unit_vector(self.simulation.vector_car_to_sector)
        obs = np.array(
            list(self.simulation.sample_sensors().distances)
            + [
                self.simulation.car_speed(),
                self.simulation.front_wheel_angle(),
                unit_car_to_sector.x,
                unit_car_to_sector.y,
                tires_on_grass,
            ],
            dtype=np.float32,
        )
        return obs

    def _get_info(self):
        obs = np.array(
            list(self.simulation.sample_sensors().distances)
            + [self.simulation.car_speed(), self.simulation.front_wheel_angle()],
            dtype=np.float32,
        )
        timing_state = self.simulation.timing_state
        timing_info = {
            "current_time": timing_state.current_time,
            "last_lap_time": timing_state.last_lap_time,
            "best_lap_time": timing_state.best_lap_time,
            "sector_times": [s.current_time for s in timing_state.sector_statuses],
            "sector_best": [s.best_time for s in timing_state.sector_statuses],
            "sector_completed": [s.completed for s in timing_state.sector_statuses],
        }
        telemetry = {
            "speed": float(self.simulation.car_speed()),
            "wheel_angle": float(self.simulation.front_wheel_angle()),
            "throttle": float(self.simulation.car.throttle_input),
            "brake": float(max(0.0, -self.simulation.car.throttle_input)),
            "steer": float(self.simulation.car.steering_input),
        }
        return {"obs": obs, "timing": timing_info, "telemetry": telemetry}

    def _normalize_to_int(self, value: float) -> int:
        src_min, src_max = -20, 90
        dst_min, dst_max = -1, 2

        value = max(src_min, min(value, src_max))

        normalized = (value - src_min) / (src_max - src_min)
        mapped = dst_min + normalized * (dst_max - dst_min)

        return int(round(mapped))

    def unit_vector(self, vector):
        """Returns the unit vector of the vector."""
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'"""
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def _get_reward(
        self,
    ) -> tuple[float, bool]:
        reward: float = 0.0
        terminated: bool = False

        tires_on_grass = sum(
            1 for tire in self.simulation.car.tires if tire.is_fully_on_grass
        )

        if tires_on_grass == 4:
            reward = -10.0
            terminated = True
            return reward, terminated

        #       completed_sectors_this_turn = sum(
        #           s.completed for s in current_sectors.sector_statuses
        #       ) - sum(s.completed for s in last_sectors.sector_statuses)
        #
        #       faster_sectors_this_turn = sum(
        #           bool(s.faster) for s in current_sectors.sector_statuses
        #       ) - sum(bool(s.faster) for s in last_sectors.sector_statuses)
        #
        #       reward += completed_sectors_this_turn * 100.0
        #       reward += faster_sectors_this_turn * 150.0
        #
        v = self.simulation.car.forward_speed
        alpha = self.angle_between(
            self.simulation.car.forward_vector,
            self.simulation.vector_between_sectors(0),
        )
        distance = np.clip(self.simulation.distance_to_spline / 25, 0, 1)

        reward = float(v * (cos(alpha) - distance))

        return reward, terminated

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulation.reset()
        if hasattr(self.simulation.car, "clear_crash"):
            self.simulation.car.clear_crash()
        obs = self._get_obs()
        info = self._get_info()
        self._last_info = info
        return obs, info

    def step(self, action):
        reward = 0

        self.simulation.car.steering_input = float(action[0])
        self.simulation.car.throttle_input = float(action[1])

        self.simulation.step(
            TIME_STEP,
            velocity_iterations=VELOCITY_ITERATIONS,
            position_iterations=POSITION_ITERATIONS,
            keys=None,
        )

        obs = self._get_obs()
        info = self._get_info()
        terminated = False  # No terminal condition defined yet
        truncated = False  # No time limit enforcement here

        reward, terminated = self._get_reward()
        self._last_info = info
        return obs, reward, terminated, truncated, info

    def render(self, mode: str | None = None):
        target_mode = mode or self.render_mode
        if target_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render mode: {target_mode}")

        if target_mode == "human":
            if not self._display_active and self.render_mode == "rgb_array":
                # VecEnv may request 'human' even if we operate off-screen; provide frame.
                return self.render("rgb_array")
            pygame.event.pump()
            self.renderer.draw(self.simulation)
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None

        if target_mode == "rgb_array":
            self.renderer.draw(self.simulation)
            frame = pygame.surfarray.array3d(self.screen)
            return np.transpose(frame, (1, 0, 2))

        raise ValueError(f"Unsupported render mode: {target_mode}")

    def close(self):
        """Clean up pygame resources."""
        if self._display_active and pygame.display.get_init():
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()
