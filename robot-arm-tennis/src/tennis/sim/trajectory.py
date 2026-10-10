"""Time-parameterised piecewise-quintic joint trajectories (evaluated at the servo rate)."""

from __future__ import annotations

import numpy as np


def quintic_coeffs(
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    q1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
    T: float,
) -> np.ndarray:
    """Coefficients c[:, 0..5] of q(tau) = sum c_k tau^k matching position/velocity/acceleration."""
    d = q1 - q0
    t2, t3, t4, t5 = T * T, T**3, T**4, T**5
    c = np.empty((len(q0), 6))
    c[:, 0] = q0
    c[:, 1] = v0
    c[:, 2] = 0.5 * a0
    c[:, 3] = (20 * d - (8 * v1 + 12 * v0) * T - (3 * a0 - a1) * t2) / (2 * t3)
    c[:, 4] = (-30 * d + (14 * v1 + 16 * v0) * T + (3 * a0 - 2 * a1) * t2) / (2 * t4)
    c[:, 5] = (12 * d - 6 * (v1 + v0) * T + (a1 - a0) * t2) / (2 * t5)
    return c


def quintic_peaks(c: np.ndarray, T: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Peak |velocity| and |acceleration| per joint over [0, T] (sampled)."""
    tau = np.linspace(0.0, T, n)
    p = np.vstack(
        [np.zeros_like(tau), np.ones_like(tau), 2 * tau, 3 * tau**2, 4 * tau**3, 5 * tau**4]
    )
    a = np.vstack(
        [
            np.zeros_like(tau),
            np.zeros_like(tau),
            2 * np.ones_like(tau),
            6 * tau,
            12 * tau**2,
            20 * tau**3,
        ]
    )
    return np.abs(c @ p).max(axis=1), np.abs(c @ a).max(axis=1)


class JointTrajectory:
    """Sequence of quintic segments [t0, t0+T]; holds the final position afterwards."""

    def __init__(self, q: np.ndarray, t: float = 0.0) -> None:
        self.n = len(q)
        self.t0: list[float] = []
        self.dur: list[float] = []
        self.coef: list[np.ndarray] = []
        self.q_end = np.array(q, dtype=float)
        self.t_end = float(t)
        self.labels: list[str] = []

    @classmethod
    def hold(cls, q: np.ndarray, t: float = 0.0) -> JointTrajectory:
        return cls(q, t)

    def end_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Exact state at the end of the last segment (may be moving)."""
        if not self.coef:
            z = np.zeros(self.n)
            return self.q_end.copy(), z, z.copy()
        c, tau = self.coef[-1], self.dur[-1]
        q = c[:, 0] + tau * (
            c[:, 1] + tau * (c[:, 2] + tau * (c[:, 3] + tau * (c[:, 4] + tau * c[:, 5])))
        )
        v = c[:, 1] + tau * (
            2 * c[:, 2] + tau * (3 * c[:, 3] + tau * (4 * c[:, 4] + tau * 5 * c[:, 5]))
        )
        a = 2 * c[:, 2] + tau * (6 * c[:, 3] + tau * (12 * c[:, 4] + tau * 20 * c[:, 5]))
        return q, v, a

    def append(
        self,
        q1: np.ndarray,
        v1: np.ndarray,
        T: float,
        label: str = "",
        a1: np.ndarray | None = None,
        start: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        t_start: float | None = None,
    ) -> None:
        """Append a quintic from the end state (or ``start``) reaching (q1, v1, a1) after T."""
        if start is None:
            q0, v0, a0 = self.end_state()
        else:
            q0, v0, a0 = start
        t0 = self.t_end if t_start is None else t_start
        zero = np.zeros(self.n)
        c = quintic_coeffs(
            q0, v0, a0, np.asarray(q1, float), np.asarray(v1, float), zero if a1 is None else a1, T
        )
        self.t0.append(t0)
        self.dur.append(T)
        self.coef.append(c)
        self.labels.append(label)
        self.q_end = np.asarray(q1, float).copy()
        self.t_end = t0 + T

    def wait_until(self, t: float) -> None:
        """Extend the end (at rest) until ``t``; only valid if the trajectory ends at rest."""
        self.t_end = max(self.t_end, t)

    def segment_at(self, t: float) -> int:
        for i in range(len(self.t0) - 1, -1, -1):
            if t >= self.t0[i]:
                return i
        return -1

    def label_at(self, t: float) -> str:
        i = self.segment_at(t)
        if i < 0 or t > self.t0[i] + self.dur[i]:
            return "hold"
        return self.labels[i]

    def sample(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        i = self.segment_at(t)
        if i < 0:
            c = self.coef[0] if self.coef else None
            if c is None:
                z = np.zeros(self.n)
                return self.q_end.copy(), z, z.copy()
            return c[:, 0].copy(), np.zeros(self.n), np.zeros(self.n)
        tau = t - self.t0[i]
        T = self.dur[i]
        c = self.coef[i]
        if tau >= T:
            if i == len(self.t0) - 1:
                z = np.zeros(self.n)
                return self.q_end.copy(), z, z.copy()
            tau = T
        q = c[:, 0] + tau * (
            c[:, 1] + tau * (c[:, 2] + tau * (c[:, 3] + tau * (c[:, 4] + tau * c[:, 5])))
        )
        v = c[:, 1] + tau * (
            2 * c[:, 2] + tau * (3 * c[:, 3] + tau * (4 * c[:, 4] + tau * 5 * c[:, 5]))
        )
        a = 2 * c[:, 2] + tau * (6 * c[:, 3] + tau * (12 * c[:, 4] + tau * 20 * c[:, 5]))
        return q, v, a

    def truncate(self, t: float) -> None:
        """Drop everything after ``t`` (the state at ``t`` becomes the new end, may be moving)."""
        q, _, _ = self.sample(t)
        keep = [i for i in range(len(self.t0)) if self.t0[i] < t]
        self.t0 = [self.t0[i] for i in keep]
        self.dur = [min(self.dur[i], t - self.t0[i]) for i in keep]
        self.coef = [self.coef[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]
        self.q_end = q
        self.t_end = t
