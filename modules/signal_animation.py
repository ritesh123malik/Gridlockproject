"""
modules/signal_animation.py
----------------------------
Simulates 2-junction traffic signal control across N time steps.
Compares FIXED-cycle signals vs ADAPTIVE (RL-style) signals.
Returns per-step queue lengths for both junctions under both modes.
"""

import numpy as np


class SignalAnimator:
    def __init__(self, n_steps: int = 20, arrival_rate: float = 8.0,
                 fixed_green: int = 5, max_queue: int = 60):
        """
        n_steps       — number of time steps to simulate.
        arrival_rate  — vehicles arriving per step (Poisson mean).
        fixed_green   — vehicles discharged per step on FIXED green.
        max_queue     — queue cap (road capacity).
        """
        self.n_steps = n_steps
        self.arrival_rate = arrival_rate
        self.fixed_green = fixed_green
        self.max_queue = max_queue

    def run(self, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)

        # Two junctions: J1 (main arterial) and J2 (feeder)
        # Arrivals drawn from Poisson
        arrivals_j1 = rng.poisson(self.arrival_rate, self.n_steps)
        arrivals_j2 = rng.poisson(self.arrival_rate * 0.7, self.n_steps)

        # ── FIXED MODE ──
        fixed_q1, fixed_q2 = [], []
        q1, q2 = 0, 0
        for t in range(self.n_steps):
            q1 = min(self.max_queue, q1 + arrivals_j1[t] - self.fixed_green)
            q2 = min(self.max_queue, q2 + arrivals_j2[t] - self.fixed_green)
            fixed_q1.append(max(0, q1))
            fixed_q2.append(max(0, q2))

        # ── ADAPTIVE MODE ──
        # Green time proportional to queue length, capped at max_green
        adaptive_q1, adaptive_q2 = [], []
        q1, q2 = 0, 0
        max_green = int(self.fixed_green * 2.5)
        for t in range(self.n_steps):
            # Adaptive discharge: scale green time with queue length
            green1 = max(2, min(max_green, int(q1 * 0.6 + 2)))
            green2 = max(2, min(max_green, int(q2 * 0.6 + 2)))
            q1 = min(self.max_queue, q1 + arrivals_j1[t] - green1)
            q2 = min(self.max_queue, q2 + arrivals_j2[t] - green2)
            adaptive_q1.append(max(0, q1))
            adaptive_q2.append(max(0, q2))

        steps = list(range(1, self.n_steps + 1))

        return {
            "steps": steps,
            "fixed_j1": fixed_q1,
            "fixed_j2": fixed_q2,
            "adaptive_j1": adaptive_q1,
            "adaptive_j2": adaptive_q2,
            "fixed_total_delay": sum(fixed_q1) + sum(fixed_q2),
            "adaptive_total_delay": sum(adaptive_q1) + sum(adaptive_q2),
            "improvement_pct": round(
                (1 - (sum(adaptive_q1) + sum(adaptive_q2)) /
                 max(1, sum(fixed_q1) + sum(fixed_q2))) * 100, 1
            )
        }
