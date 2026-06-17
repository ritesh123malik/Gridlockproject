"""
modules/signal_controller.py
----------------------------
Autonomous signal control with Q-learning emulator.
Simulates training feedback loops where the agent chooses to extend green phases,
reduce green phases, or maintain current intervals based on queue length status.
"""

import random

# Mapping of Q-learning actions to readable string commands
SIGNAL_ACTIONS = {
    0: "Extend Green Phase (+15s)",
    1: "Reduce Green Phase (-10s)",
    2: "Maintain Phase (0s)"
}

class AdaptiveSignalController:
    def __init__(self):
        # Q-Table stores: state (e.g. "high_density_inbound") -> {action_id: Q-value}
        self.q_table = {}

    def get_action(self, state: str) -> int:
        """
        Retrieves action selection using an epsilon-greedy choice policy.
        States might be "HIGH_CONGESTION", "MODERATE_CONGESTION", or "LOW_CONGESTION".
        Actions:
          0 - Extend green light
          1 - Throttle incoming flow (gating)
          2 - Do nothing / maintain cycle
        """
        if state not in self.q_table:
            # Initialize with small float priors
            self.q_table[state] = {0: 0.0, 1: 0.0, 2: 0.0}
            
        # Exploration rate (epsilon) of 10%
        if random.random() < 0.1:
            return random.choice([0, 1, 2])
            
        # Exploitation (greedy choice)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state: str, action: int, reward: float, next_state: str, alpha: float = 0.2, gamma: float = 0.9):
        """
        Updates the Q-value based on the Bellman equation (mocked Q-value updating).
        """
        if state not in self.q_table:
            self.q_table[state] = {0: 0.0, 1: 0.0, 2: 0.0}
        if next_state not in self.q_table:
            self.q_table[next_state] = {0: 0.0, 1: 0.0, 2: 0.0}
            
        max_next = max(self.q_table[next_state].values())
        old_val = self.q_table[state][action]
        
        # Temporal difference update
        self.q_table[state][action] = old_val + alpha * (reward + gamma * max_next - old_val)
