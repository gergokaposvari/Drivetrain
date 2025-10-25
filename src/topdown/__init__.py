"""Top-down car simulation components based on the pybox2d example."""

from .car import Car
from .contact import TopDownContactListener
from .ground import GroundArea
from .input import InputHandler, InputMapping
from .render import RenderConfig, Renderer
from .runtime import SimulationConfig, SimulationSession
from .simulation import Simulation
from .state import CarState, SimulationSnapshot, TireState
from .gym_env import RewardWeights, TopDownCarEnv
from .tire import Tire
from .track import Track, TrackSurface, default_track
from .track_loader import LoadedTrack, TrackLoadError, discover_tracks, load_track, save_track_timing
from .timing import TimingManager
from .track_builder import (
    DEFAULT_GRASS_FRICTION,
    DEFAULT_ROAD_FRICTION,
    SplineTrackConfig,
    build_track_from_spline,
)
from .vector_env import make_vector_env

__all__ = [
    "Car",
    "GroundArea",
    "InputHandler",
    "InputMapping",
    "RenderConfig",
    "Renderer",
    "Simulation",
    "SimulationConfig",
    "SimulationSession",
    "TopDownCarEnv",
    "RewardWeights",
    "make_vector_env",
    "Tire",
    "TopDownContactListener",
    "Track",
    "TrackSurface",
    "default_track",
    "LoadedTrack",
    "TrackLoadError",
    "discover_tracks",
    "load_track",
    "save_track_timing",
    "SplineTrackConfig",
    "build_track_from_spline",
    "DEFAULT_ROAD_FRICTION",
    "DEFAULT_GRASS_FRICTION",
    "TimingManager",
    "CarState",
    "TireState",
    "SimulationSnapshot",
]
