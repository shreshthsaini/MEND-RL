"""Proximal temperature controller and step-size schedules."""

from __future__ import annotations

from dataclasses import dataclass


def step_growth_scale(update: int, growth: float, growth_max: float) -> float:
    """Trust-region curriculum (hillclimb flag step_growth): s(u) = min(growth_max, 1 + growth * u) after u
    optimizer updates. The caller multiplies the hint steps eta_j by s and tau by s^2, so the move cost
    ||s d||^2 / (2 s^2 tau) of a scaled candidate equals the unscaled one: only the reward decides whether the
    bigger move wins. growth = 0 gives s = 1 (the method)."""
    if growth <= 0:
        return 1.0
    return float(min(float(growth_max), 1.0 + float(growth) * max(int(update), 0)))


def lr_at(lr0: float, update: int, schedule: str = "const", n: int = 50, min_frac: float = 0.1) -> float:
    """Learning rate for the round after ``update`` optimizer updates (hillclimb flag lr_schedule). 'const' = lr0
    (the method); 'cosine' = lr0 (m + (1 - m) (1 + cos(pi min(u, n) / n)) / 2), m = min_frac: lr0 at u = 0, m lr0
    from u = n on."""
    if schedule == "const":
        return float(lr0)
    if schedule != "cosine":
        raise ValueError(f"unknown lr schedule {schedule!r} (const|cosine)")
    import math
    frac = min(max(int(update), 0), int(n)) / float(n)
    return float(lr0) * (float(min_frac) + (1.0 - float(min_frac)) * 0.5 * (1.0 + math.cos(math.pi * frac)))


@dataclass
class TauController:
    """Acceptance controller for tau: multiplicative step toward [lo, hi], clamped, fixed within a round."""

    tau: float
    tau_min: float
    tau_max: float
    lo: float = 0.3
    hi: float = 0.6
    gamma: float = 1.5

    def __post_init__(self):
        if not 0 < self.tau_min <= self.tau_max:
            raise ValueError("need 0 < tau_min <= tau_max")
        self.tau = min(max(float(self.tau), self.tau_min), self.tau_max)

    def update(self, acceptance: float) -> float:
        """Call once per round after the verdict; the new tau applies to the next round."""
        if acceptance < self.lo:
            self.tau *= self.gamma
        elif acceptance > self.hi:
            self.tau /= self.gamma
        self.tau = min(max(self.tau, self.tau_min), self.tau_max)
        return self.tau

    def state_dict(self):
        return {"tau": self.tau}

    def load_state_dict(self, state):
        self.tau = min(max(float(state["tau"]), self.tau_min), self.tau_max)
