"""Ball-flight prediction and the ballistic shot solver."""

import math

import numpy as np

from tennis.control.ballistics import BallModel, BallPredictor
from tennis.sim.physics import PhysicsParams


def _reference_predict(bp: BallPredictor, pos, vel, horizon, spin=None, max_bounces=2):
    """Straightforward implementation (uses BallPredictor._acc) - the fast loop must match it."""
    m, c, dt = bp.model, bp.court, bp.dt
    r = m.radius
    x, y, z = (float(v) for v in pos)
    vx, vy, vz = (float(v) for v in vel)
    sx, sy, sz = (0.0, 0.0, 0.0) if spin is None else (float(v) for v in spin)
    n = int(horizon / dt) + 1
    out, bounces = [], 0
    for i in range(n):
        out.append((x, y, z))
        if i == n - 1:
            break
        ax, ay, az = bp._acc(vx, vy, vz, sx, sy, sz)
        hx, hy, hz = vx + 0.5 * dt * ax, vy + 0.5 * dt * ay, vz + 0.5 * dt * az
        ax, ay, az = bp._acc(hx, hy, hz, sx, sy, sz)
        nx_, ny_, nz_ = x + dt * hx, y + dt * hy, z + dt * hz
        nvx, nvy, nvz = vx + dt * ax, vy + dt * ay, vz + dt * az
        if (x < 0.0) != (nx_ < 0.0) and abs(x - nx_) > 1e-12:
            f = x / (x - nx_)
            if abs(y + f * (ny_ - y)) <= c.net_half_span and z + f * (nz_ - z) < c.net_height + r:
                break
        if nz_ <= r and nvz < 0.0:
            f = (z - r) / max(z - nz_, 1e-12)
            bx, by = x + f * (nx_ - x), y + f * (ny_ - y)
            bvx, bvy, bvz = vx + f * (nvx - vx), vy + f * (nvy - vy), vz + f * (nvz - vz)
            bounces += 1
            vt = math.hypot(bvx, bvy)
            loss = min(m.bounce_mu * (1 + m.bounce_e) * abs(bvz), (1 - m.bounce_beta) * vt)
            scale = (vt - loss) / vt if vt > 1e-9 else 1.0
            rem = (1.0 - f) * dt
            nvx, nvy, nvz = bvx * scale, bvy * scale, -m.bounce_e * bvz
            nx_, ny_, nz_ = bx + rem * nvx, by + rem * nvy, r + rem * nvz
            if bounces >= max_bounces:
                out.append((nx_, ny_, nz_))
                break
        x, y, z, vx, vy, vz = nx_, ny_, nz_, nvx, nvy, nvz
    return np.array(out)


def test_fast_predictor_is_identical_to_reference(rng) -> None:
    bp = BallPredictor(BallModel(wind=(0.4, -0.6, 0.0)), dt=0.005)
    for k in range(60):
        p = np.array([rng.uniform(-2.8, -0.5), rng.uniform(-1, 1), rng.uniform(0.3, 1.3)])
        v = np.array([rng.uniform(2, 9), rng.uniform(-2, 2), rng.uniform(-2, 4)])
        spin = rng.normal(0, 30, 3) if k % 2 else None
        fast = bp.predict(p, v, 2.2, spin=spin, max_bounces=2).pos
        ref = _reference_predict(bp, p, v, 2.2, spin=spin, max_bounces=2)
        assert fast.shape == ref.shape
        assert np.array_equal(fast, ref)


def test_prediction_matches_mujoco_flight_and_bounce(sim) -> None:
    sim.set_params(PhysicsParams())
    sim.reset(ball_pos=(-2.0, 0.2, 1.0), ball_vel=(4.0, -0.3, 1.5))
    pred = BallPredictor(BallModel(), dt=0.005).predict(
        np.array([-2.0, 0.2, 1.0]), np.array([4.0, -0.3, 1.5]), horizon=1.2, max_bounces=2
    )
    while sim.time < 0.9:
        sim.step()
    pos, _ = sim.ball()
    expected, _ = pred.state_at(sim.time)
    assert pred.first("bounce") is not None  # the comparison covers one court bounce
    assert np.linalg.norm(pos - expected) < 0.06


def test_shot_solver_lands_on_target() -> None:
    bp = BallPredictor(BallModel(), dt=0.01)
    p0 = np.array([-2.6, 0.1, 0.7])
    for target, speed in (((2.0, 0.0), 5.5), ((1.3, 0.8), 4.0), ((2.6, -0.8), 7.0)):
        res = bp.solve_shot(p0, target, speed)
        assert res is not None
        v0 = res[0] if isinstance(res, tuple) else res["v0"]
        bounce, net_z = bp.landing(p0, v0)
        assert bounce is not None
        assert math.dist(bounce.pos[:2], target) < 0.08  # solver acceptance threshold
        assert net_z is not None and net_z > 0.35 + 0.033
