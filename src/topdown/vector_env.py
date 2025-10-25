"""Factory helpers for creating vectorised top-down car environments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence

import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

from .gym_env import TopDownCarEnv


def _make_env_factory(**env_kwargs: Any) -> Callable[[], TopDownCarEnv]:
    def _factory() -> TopDownCarEnv:
        return TopDownCarEnv(**env_kwargs)

    return _factory


def make_vector_env(
    num_envs: int,
    *,
    env_kwargs: Dict[str, Any] | None = None,
    asynchronous: bool = False,
) -> gym.vector.VectorEnv:
    """Return a Gymnasium VectorEnv wrapping multiple top-down car envs."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive")

    kwargs = dict(env_kwargs or {})
    factories: Sequence[Callable[[], TopDownCarEnv]] = tuple(
        _make_env_factory(**kwargs) for _ in range(num_envs)
    )

    if asynchronous:
        return AsyncVectorEnv(factories)
    return SyncVectorEnv(factories)


__all__ = ["make_vector_env"]
