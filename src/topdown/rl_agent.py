import gymnasium as gym
from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np


class DriverAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self._rng = np.random.default_rng()
        self.training_error = []
        self.q_values: defaultdict[tuple, np.ndarray]
        self.set_env(env)

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(self._action_size, dtype=np.float32))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

    def set_env(self, env: gym.Env) -> None:
        """Update the environment reference (useful when switching wrappers)."""
        self.env = env
        self._single_action_space = getattr(env, "single_action_space", env.action_space)
        if not hasattr(self._single_action_space, "n"):
            raise ValueError("DriverAgent requires a discrete action space.")
        self._action_size = self._single_action_space.n

    def _to_state(self, obs: np.ndarray) -> tuple:
        arr = np.asarray(obs, dtype=np.float32)
        return tuple(np.round(arr, decimals=3))

    def _sample_random_action(self) -> int:
        sample = self._single_action_space.sample()
        if np.isscalar(sample):
            return int(sample)
        return int(np.asarray(sample).item())

    def _select_action(self, obs: np.ndarray) -> int:
        state = self._to_state(obs)
        if self._rng.random() < self.epsilon:
            return self._sample_random_action()
        return int(np.argmax(self.q_values[state]))

    def get_action(self, obs: np.ndarray) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        obs_arr = np.asarray(obs)
        if obs_arr.ndim == 1:
            return self._select_action(obs_arr)
        if obs_arr.ndim >= 2:
            actions = [self._select_action(sample) for sample in obs_arr]
            return np.asarray(actions, dtype=np.int64)
        raise ValueError("Unexpected observation shape in get_action")

    def update(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        next_obs: np.ndarray,
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        obs_arr = np.asarray(obs)
        next_obs_arr = np.asarray(next_obs)
        reward_arr = np.asarray(reward, dtype=np.float32)
        done_arr = np.asarray(done, dtype=bool)

        if np.isscalar(action):
            td = self._update_single(obs_arr, int(action), float(reward_arr), bool(done_arr), next_obs_arr)
            self.training_error.append(td)
            return

        action_arr = np.asarray(action).astype(int)
        for idx in range(obs_arr.shape[0]):
            td = self._update_single(
                obs_arr[idx],
                action_arr[idx],
                float(reward_arr[idx]),
                bool(done_arr[idx]),
                next_obs_arr[idx],
            )
            self.training_error.append(td)

    def _update_single(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_obs: np.ndarray,
    ) -> float:
        state = self._to_state(obs)
        next_state = self._to_state(next_obs)
        future_q_value = 0.0 if done else np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state][action]
        self.q_values[state][action] += self.lr * temporal_difference
        return float(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: Path) -> None:
        """Persist the learned Q-table to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            tuple(state): np.array(values, dtype=np.float32)
            for state, values in self.q_values.items()
        }
        data = {
            "q_values": serializable,
            "epsilon": self.epsilon,
            "learning_rate": self.lr,
            "discount_factor": self.discount_factor,
        }
        with path.open("wb") as handle:
            pickle.dump(data, handle)

    def load(self, path: Path) -> None:
        """Load a previously saved Q-table."""
        path = Path(path)
        with path.open("rb") as handle:
            data = pickle.load(handle)

        restored = defaultdict(lambda: np.zeros(self.env.action_space.n))
        for state, values in data.get("q_values", {}).items():
            state_tuple = tuple(state)
            restored[state_tuple] = np.array(values, dtype=np.float32)
        self.q_values = restored
        self.epsilon = data.get("epsilon", self.epsilon)
