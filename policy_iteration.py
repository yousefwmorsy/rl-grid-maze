import numpy as np


class PolicyIteration:
    """Exact policy iteration for an environment with a deterministic tabular model."""

    ACTION_SYMBOLS = {
        0: "→",
        1: "↑",
        2: "←",
        3: "↓",
    }

    @staticmethod
    def print_results(policy, value_function, grid_size):
        """Print a grid of policy actions and corresponding state values."""
        policy = np.asarray(policy)
        value_function = np.asarray(value_function, dtype=float)
        policy_grid = policy.reshape(grid_size, grid_size)
        value_grid = value_function.reshape(grid_size, grid_size)
        cell_width = max(
            11,
            max(
                len(
                    f"{PolicyIteration.ACTION_SYMBOLS[action]} "
                    f"{value_grid[row, column]:.3f}"
                )
                for row in range(grid_size)
                for column, action in enumerate(policy_grid[row])
            )
            + 2,
        )
        separator = "+" + "+".join("-" * cell_width for _ in range(grid_size)) + "+"

        print("Policy and value function")
        print("Legend: → RIGHT, ↑ UP, ← LEFT, ↓ DOWN")
        print(separator)
        for row in range(grid_size):
            cells = [
                f" {PolicyIteration.ACTION_SYMBOLS[action]} "
                f"{value_grid[row, column]:.3f}".center(cell_width)
                for column, action in enumerate(policy_grid[row])
            ]
            print("|" + "|".join(cells) + "|")
            print(separator)

    def __init__(
        self,
        env,
        initial_policy=None,
        initial_value_function=None,
        discount_factor=0.99,
        theta=1e-9,
    ):

        self.env = env
        self.discount_factor = discount_factor
        self.theta = theta
        self.num_iterations = 0
        self.num_states = env.num_states
        self.num_actions = env.action_space.n

        self.policy = (
            np.asarray(initial_policy, dtype=int).copy()
            if initial_policy is not None
            else np.random.default_rng().integers(
                self.num_actions, size=self.num_states
            )
        )
        self.value_function = (
            np.asarray(initial_value_function, dtype=float).copy()
            if initial_value_function is not None
            else np.zeros(self.num_states)
        )

    def _action_value(self, state, action):
        """Calculates Q(s, a) = R(s, a) + γ * V(s')"""
        next_state, reward, terminated = self.env.transition(state, action)
        if terminated:
            return reward
        return reward + self.discount_factor * self.value_function[next_state]

    def _state_value(self, state): 
        """Calculates V(s) = Q(s, π(s)) for the current policy"""
        action = self.policy[state]
        return self._action_value(state, action)

    def policy_evaluation(self):
        while True:
            value_function_iteration = self.value_function.copy()
            delta = 0.0
            for state in range(self.num_states):
                value_function_iteration[state] = self._state_value(state)
                delta = max(delta, abs(value_function_iteration[state] - self.value_function[state]))
            self.value_function = value_function_iteration.copy()
            if delta < self.theta:
                return

    def policy_improvement(self):
        policy_stable = True
        for state in range(self.num_states):
            old_action = self.policy[state]
            
            action_values = [self._action_value(state, action) for action in range(self.num_actions)]
            self.policy[state] = np.argmax(action_values)
            
            if old_action != self.policy[state]:
                policy_stable = False
        return policy_stable

    def run_policy_iteration(self):
        while True:
            self.num_iterations += 1
            self.policy_evaluation()
            if self.policy_improvement():
                return self.policy, self.value_function
