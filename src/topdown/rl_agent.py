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
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def _to_state(self, obs: np.ndarray) -> tuple:
        return tuple(np.round(np.asarray(obs, dtype=np.float32), decimals=3))

    def get_action(self, obs: np.ndarray) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        state = self._to_state(obs)

        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        else:
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        state = self._to_state(obs)
        next_state = self._to_state(next_obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

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
