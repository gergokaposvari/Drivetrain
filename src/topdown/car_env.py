import gymnasium as gym
import pygame
import numpy as np

from pathlib import Path
from typing import Optional

from src.topdown import LoadedTrack, TrackLoadError, discover_tracks, load_track
from src.topdown.input import InputHandler
from src.topdown.render import RenderConfig, Renderer
from src.topdown.simulation import Simulation
from src.topdown.sensors import Segment


TIME_STEP = 1.0 / 60.0
VELOCITY_ITERATIONS = 6
POSITION_ITERATIONS = 2


class CarEnv(gym.Env):
    def __init__(self):
        super().__init__()

        actual_track = load_track(Path(__file__).parent.parent.parent / "tracks" / "test_track.json")
        test_track = LoadedTrack(
            name="Test-track",
            path=Path(__file__).parent / "test_track.json",
            track=actual_track,
        )

        pygame.init()

        config = RenderConfig(draw_sensor_rays=False)
        screen = pygame.display.set_mode(config.screen_size)
        pygame.display.set_caption("Top Down Car (pybox2d example)")

        self.input_handler = InputHandler()
        self.clock = pygame.time.Clock()
        self.simulation = Simulation(
            track=test_track.track if test_track else None,
            track_file=test_track.path if test_track else None,
        )
        self.renderer = Renderer(screen, config)

        low = np.array([0, 0, 0, 0, 0, 0, 0, -20, -0.7], dtype=np.float32)
        high = np.array([150, 150, 150, 150, 150, 150, 150, 90, 0.7], dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=low, high=high, shape=(9,), dtype=np.float32)
        #  action_space[0] = acceleration,
        #  action_space[1] = braking,
        #  action_space[2] = steering left,
        #  action_space[3] = steering right
        self.action_space = gym.spaces.MultiBinary(4)

    def _get_obs(self):
        obs = np.array(
            list(self.simulation.sample_sensors().distances)
            + [self.simulation.car_speed(), self.simulation.front_wheel_angle()],
            dtype=np.float32
        )
        print("obs:", obs)
        return obs

    def _get_info(self):
        obs = np.array(
            list(self.simulation.sample_sensors().distances)
            + [self.simulation.car_speed(), self.simulation.front_wheel_angle()],
            dtype=np.float32
        )
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulation.reset()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        obs = self._get_obs()
        info = self._get_info()

        directions = set()
        if action[0]:
            directions.add("up")
        if action[1]:
            directions.add("down")
        if action[2]:
            directions.add("left")
        if action[3]:
            directions.add("right")

        self.simulation.step(TIME_STEP, velocity_iterations=VELOCITY_ITERATIONS, position_iterations=POSITION_ITERATIONS, keys=directions)

        return obs, 1, False, False, info
