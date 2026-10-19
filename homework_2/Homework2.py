"""MAE 263F Homework 2: discrete elastic beam (falling beam + simply supported beam).

The beam is a chain of N nodes with DOF vector q = [x1, y1, x2, y2, ..., xN, yN].
Neighboring nodes are joined by stretching springs and every interior node
carries a bending spring. Both parts share the same implicit Euler / Newton
solver; Part 1 adds weight and viscous drag, Part 2 adds a point load and
boundary conditions (solve only for the free DOFs).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Elastic energies
# ---------------------------------------------------------------------------
def gradEs(xk, yk, xkp1, ykp1, l_k, EA):
    """Gradient of Es = 0.5*EA*(1 - l/l_k)^2*l_k w.r.t. (xk, yk, xkp1, ykp1)."""
    d = np.array([xkp1 - xk, ykp1 - yk])
    l = np.linalg.norm(d)
    t = d / l
    return EA * (l / l_k - 1.0) * np.concatenate([-t, t])


def hessEs(xk, yk, xkp1, ykp1, l_k, EA):
    """Hessian (4x4) of the stretching energy."""
    d = np.array([xkp1 - xk, ykp1 - yk])
    l = np.linalg.norm(d)
    t = d / l
    dl = np.concatenate([-t, t])
    A = (np.eye(2) - np.outer(t, t)) / l
    d2l = np.block([[A, -A], [-A, A]])
    return EA / l_k * np.outer(dl, dl) + EA * (l / l_k - 1.0) * d2l


def _angleDerivs(e):
    """Gradient and Hessian of the edge angle theta = atan2(ey, ex) w.r.t. e."""
    ex, ey = e
    r2 = ex**2 + ey**2
    g = np.array([-ey, ex]) / r2
    H = np.array([[2 * ex * ey, ey**2 - ex**2], [ey**2 - ex**2, -2 * ex * ey]]) / r2**2
    return g, H


_Be = np.hstack([-np.eye(2), np.eye(2), np.zeros((2, 2))])  # e = x_k - x_{k-1}
_Bf = np.hstack([np.zeros((2, 2)), -np.eye(2), np.eye(2)])  # f = x_{k+1} - x_k


def _curvatureDerivs(xkm1, ykm1, xk, yk, xkp1, ykp1):
    """kappa = 2 tan(phi/2) (phi = turning angle) with its gradient and Hessian."""
    e = np.array([xk - xkm1, yk - ykm1])
    f = np.array([xkp1 - xk, ykp1 - yk])
    phi = np.arctan2(e[0] * f[1] - e[1] * f[0], e @ f)
    ge, He = _angleDerivs(e)
    gf, Hf = _angleDerivs(f)
    gphi = _Bf.T @ gf - _Be.T @ ge
    Hphi = _Bf.T @ Hf @ _Bf - _Be.T @ He @ _Be

    kappa = 2.0 * np.tan(phi / 2)
    dk = 1.0 + kappa**2 / 4  # d kappa / d phi
    ddk = dk * kappa / 2  # d^2 kappa / d phi^2
    gk = dk * gphi
    Hk = ddk * np.outer(gphi, gphi) + dk * Hphi
    return kappa, gk, Hk


def gradEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Gradient of Eb = 0.5*EI*(kappa - kappa0)^2/l_k w.r.t. the 6 DOFs."""
    kappa, gk, _ = _curvatureDerivs(xkm1, ykm1, xk, yk, xkp1, ykp1)
    return EI / l_k * (kappa - curvature0) * gk


def hessEb(xkm1, ykm1, xk, yk, xkp1, ykp1, curvature0, l_k, EI):
    """Hessian (6x6) of the bending energy."""
    kappa, gk, Hk = _curvatureDerivs(xkm1, ykm1, xk, yk, xkp1, ykp1)
    return EI / l_k * (np.outer(gk, gk) + (kappa - curvature0) * Hk)


def elasticFJ(q, params):
    """Gradient (2N) and Hessian (2N x 2N) of the total elastic energy."""
    dL, EA, EI = params["dL"], params["EA"], params["EI"]
    N = len(q) // 2
    F = np.zeros(2 * N)
    J = np.zeros((2 * N, 2 * N))

    for k in range(N - 1):  # stretching springs
        idx = np.arange(2 * k, 2 * k + 4)
        F[idx] += gradEs(*q[idx], dL, EA)
        J[np.ix_(idx, idx)] += hessEs(*q[idx], dL, EA)

    for k in range(1, N - 1):  # bending springs
        idx = np.arange(2 * k - 2, 2 * k + 4)
        F[idx] += gradEb(*q[idx], 0.0, dL, EI)
        J[np.ix_(idx, idx)] += hessEb(*q[idx], 0.0, dL, EI)

    return F, J


# ---------------------------------------------------------------------------
# Time stepping
# ---------------------------------------------------------------------------
def implicitEulerStep(q_old, u_old, params, max_iter=100, return_iters=False):
    """One implicit Euler step solved with Newton's method on the free DOFs.

    f = m/dt*((q - q_old)/dt - u_old) + C*(q - q_old)/dt + dE/dq - Fext
    """
    m, C, Fext = params["m"], params["C"], params["Fext"]
    dt, tol, free = params["dt"], params["tol"], params["free"]

    q_new = q_old.copy()
    for n_iter in range(1, max_iter + 1):
        Fel, Jel = elasticFJ(q_new, params)
        f = m / dt * ((q_new - q_old) / dt - u_old) + C * (q_new - q_old) / dt + Fel - Fext
        J = np.diag(m / dt**2 + C / dt) + Jel
        q_new[free] -= np.linalg.solve(J[np.ix_(free, free)], f[free])
        err = np.linalg.norm(f[free])
        if err < tol:
            break
    else:
        raise RuntimeError(f"Newton did not converge (err = {err:.3e})")

    u_new = (q_new - q_old) / dt
    if return_iters:
        return q_new, u_new, n_iter
    return q_new, u_new


def simulate(q0, params, totalTime):
    """Run the time loop. Returns times (n+1,), Q (n+1, 2N), U (n+1, 2N), iters (n,)."""
    dt = params["dt"]
    n_steps = int(round(totalTime / dt))
    Q = np.zeros((n_steps + 1, len(q0)))
    U = np.zeros_like(Q)
    iters = np.zeros(n_steps, dtype=int)
    Q[0] = q0
    for n in range(n_steps):
        Q[n + 1], U[n + 1], iters[n] = implicitEulerStep(Q[n], U[n], params, return_iters=True)
    return np.arange(n_steps + 1) * dt, Q, U, iters


def straightBeam(N, length):
    q0 = np.zeros(2 * N)
    q0[0::2] = np.linspace(0, length, N)
    return q0


# ---------------------------------------------------------------------------
# Part 1: elastic beam falling in a viscous fluid
# ---------------------------------------------------------------------------
L1, R0, E1 = 0.10, 1e-3, 1e9
RHO_METAL, RHO_FLUID, G = 7000.0, 1000.0, 9.8
R_MID = 0.025


def fallingBeamParams(nv, dt=1e-2, visc=1000.0, R_all=None):
    deltaL = L1 / (nv - 1)
    R = np.full(nv, deltaL / 10)
    R[nv // 2] = R_MID  # middle node
    if R_all is not None:  # same radius at every node
        R[:] = R_all

    EI = E1 * np.pi * R0**4 / 4
    m_node = 4 / 3 * np.pi * R**3 * RHO_METAL
    W_node = 4 / 3 * np.pi * R**3 * (RHO_METAL - RHO_FLUID) * G
    Fext = np.zeros(2 * nv)
    Fext[1::2] = -W_node  # weight (with buoyancy) along -y
    return {
        "m": np.repeat(m_node, 2),
        "C": np.repeat(6 * np.pi * visc * R, 2),
        "Fext": Fext,
        "dL": deltaL,
        "EA": E1 * np.pi * R0**2,
        "EI": EI,
        "dt": dt,
        "tol": EI / L1**2 * 1e-3,
        "free": np.arange(2 * nv),
        "R": R,
    }


def runFallingBeam(nv=21, totalTime=50.0, **kwargs):
    params = fallingBeamParams(nv, **kwargs)
    t, Q, U, _ = simulate(straightBeam(nv, L1), params, totalTime)
    mid = nv // 2
    return t, Q[:, 2 * mid + 1], U[:, 2 * mid + 1], params


def terminalVelocity(R, visc=1000.0):
    """Eq. (12): steady-state velocity of the beam."""
    return -2 * (RHO_METAL - RHO_FLUID) * G / (9 * visc) * np.sum(R**3) / np.sum(R)


# ---------------------------------------------------------------------------
# Part 2: simply supported beam with a point load
# ---------------------------------------------------------------------------
L2, R_OUT, R_IN, E2, RHO2, D_LOAD = 1.0, 0.013, 0.011, 70e9, 2700.0, 0.75


def supportedBeamParams(P, nv=50, dt=1e-2):
    deltaL = L2 / (nv - 1)
    EI = E2 * np.pi / 4 * (R_OUT**4 - R_IN**4)
    load_node = int(np.argmin(np.abs(np.linspace(0, L2, nv) - D_LOAD)))
    Fext = np.zeros(2 * nv)
    Fext[2 * load_node + 1] = -P

    fixed = np.array([0, 1, 2 * nv - 1])  # x_1, y_1, y_N
    return {
        "m": np.full(2 * nv, np.pi * (R_OUT**2 - R_IN**2) * L2 * RHO2 / (nv - 1)),
        "C": np.zeros(2 * nv),
        "Fext": Fext,
        "dL": deltaL,
        "EA": E2 * np.pi * (R_OUT**2 - R_IN**2),
        "EI": EI,
        "dt": dt,
        "tol": EI / L2**2 * 1e-3,
        "free": np.setdiff1d(np.arange(2 * nv), fixed),
        "load_node": load_node,
    }


def runSupportedBeam(P, nv=50, totalTime=1.0):
    params = supportedBeamParams(P, nv)
    t, Q, _, iters = simulate(straightBeam(nv, L2), params, totalTime)
    ymax = Q[:, 1::2].min(axis=1)
    return t, Q, ymax, iters, params


def ebDeflection(P, EI, d=D_LOAD, l=L2):
    """Eq. (6): Euler-Bernoulli maximum deflection (positive)."""
    c = min(d, l - d)
    return P * c * (l**2 - c**2) ** 1.5 / (9 * np.sqrt(3) * EI * l)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def _fdCheck(energy_grad, energy_hess, x, h=1e-7):
    """Finite-difference check of a Hessian against its gradient."""
    H_fd = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        dx = np.zeros(len(x))
        dx[i] = h
        H_fd[:, i] = (energy_grad(x + dx) - energy_grad(x - dx)) / (2 * h)
    H = energy_hess(x)
    return np.allclose(H, H_fd, rtol=1e-5, atol=1e-6 * np.abs(H).max())


def run_tests():
    rng = np.random.default_rng(0)

    # 1-2: stretching and bending Hessians match finite differences of the gradients
    x4 = rng.random(4)
    assert _fdCheck(lambda x: gradEs(*x, 0.3, 10.0), lambda x: hessEs(*x, 0.3, 10.0), x4)
    x6 = np.array([0.0, 0.0, 1.0, 0.1, 2.0, -0.3]) + 0.1 * rng.random(6)
    assert _fdCheck(lambda x: gradEb(*x, 0.2, 1.0, 5.0), lambda x: hessEb(*x, 0.2, 1.0, 5.0), x6)

    # 3: bending gradient matches finite differences of the energy
    def Eb(x):
        kappa = _curvatureDerivs(*x)[0]
        return 0.5 * 5.0 * (kappa - 0.2) ** 2 / 1.0

    g_fd = np.array([(Eb(x6 + 1e-7 * np.eye(6)[i]) - Eb(x6 - 1e-7 * np.eye(6)[i])) / 2e-7 for i in range(6)])
    assert np.allclose(gradEb(*x6, 0.2, 1.0, 5.0), g_fd, rtol=1e-5)

    # 4: curvature of a 90-degree turn is 2 tan(45 deg) = 2
    assert np.isclose(_curvatureDerivs(0, 0, 1, 0, 1, 1)[0], 2.0)

    # 5: elastic forces are internal (sum to zero) for a random shape
    p = supportedBeamParams(1.0, nv=6)
    F, _ = elasticFJ(straightBeam(6, 1.0) + 0.05 * rng.random(12), p)
    assert np.isclose(F[0::2].sum(), 0, atol=1e-6 * np.abs(F).max())
    assert np.isclose(F[1::2].sum(), 0, atol=1e-6 * np.abs(F).max())

    # 6: Part 2 checks from the handout
    p = supportedBeamParams(2000.0)
    assert np.isclose(p["dL"], 0.020408, atol=1e-6)
    assert np.isclose(p["EI"], 765.29, atol=0.01)
    assert np.isclose(p["EA"], 1.0556e7, rtol=1e-4)
    assert np.isclose(p["m"][0], 8.309e-3, rtol=1e-3)
    assert p["load_node"] == 37 and p["Fext"][75] == -2000.0
    assert len(p["free"]) == 97

    print("All 6 test cases passed.\n")


# ---------------------------------------------------------------------------
# Gradescope questions
# ---------------------------------------------------------------------------
def part1():
    print("Part 1: Falling beam")
    vinf = 2 * R_MID**2 * (RHO_METAL - RHO_FLUID) * G / (9 * 1000.0)

    _, _, v, _ = runFallingBeam(totalTime=1.0, visc=0.0, R_all=R_MID)
    print(f"  Q1: v_mid(1 s)  = {v[-1]:.4f} m/s     (analytical {-G * (1 - RHO_FLUID / RHO_METAL):.4f})")

    _, _, v, _ = runFallingBeam(totalTime=1.0, R_all=R_MID)
    print(f"  Q2: v_mid(1 s)  = {v[-1]:.4e} m/s (analytical {-vinf:.4e})")

    print("  Q5: terminal velocity vs N (simulated / Eq. 12)")
    results = {}
    for nv in (3, 5, 11, 21):
        t, y, v, p = runFallingBeam(nv=nv, totalTime=50.0)
        results[nv] = (t, y, v)
        print(f"       N = {nv:2d}: v = {v[-1]:.4e} m/s / {terminalVelocity(p['R']):.4e} m/s")
    v_inf_N = -2 * (RHO_METAL - RHO_FLUID) * G / 9000.0 * R_MID**3 / (R_MID + L1 / 10)
    print(f"       N -> inf: {v_inf_N:.4e} m/s")

    t, y, v = results[21]
    print(f"  Q3: v_mid(50 s) = {v[-1]:.4e} m/s (N = 21)")
    print(f"  Q4: y_mid(50 s) = {y[-1]:.4f} m       (v_terminal * t = {v[-1] * 50:.4f})")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for nv, (t, y, v) in results.items():
        ax[0].plot(t, v, label=f"N = {nv}")
        ax[1].plot(t, y, label=f"N = {nv}")
    ax[0].set(xlabel="t [s]", ylabel="middle node velocity [m/s]")
    ax[1].set(xlabel="t [s]", ylabel="middle node y [m]")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "falling_beam.png", dpi=150)
    plt.close(fig)


def part2():
    print("Part 2: Simply supported beam")
    t, Q, ymax, iters, p = runSupportedBeam(2000.0)
    EI = p["EI"]
    x_load = p["load_node"] * p["dL"]
    print(f"  Newton iterations per step: max {iters.max()}, first steps {iters[:5].tolist()}")
    print(f"  Q6: |ymax|(1 s) = {abs(ymax[-1]):.4f} m  (EB d=0.75: {ebDeflection(2000.0, EI):.4f}, "
          f"d={x_load:.4f}: {ebDeflection(2000.0, EI, d=x_load):.4f})")
    k_min = int(np.argmin(Q[-1, 1::2]))
    c = min(D_LOAD, L2 - D_LOAD)
    print(f"  Q7: x of largest deflection = {Q[-1, 2 * k_min]:.4f} m  "
          f"(EB: {np.sqrt((L2**2 - c**2) / 3):.4f} m)")

    w1 = (np.pi / L2) ** 2 * np.sqrt(EI / (RHO2 * np.pi * (R_OUT**2 - R_IN**2)))
    rel_change = np.abs(np.diff(ymax)) / abs(ymax[-1])
    settled = np.argmax(rel_change < 1e-3) + 1
    print(f"  Q8: ymax settles (<0.1% change/step) after {settled} steps; "
          f"omega_1*dt = {w1 * p['dt']:.1f}, damping factor/step = {(1 + (w1 * p['dt'])**2) ** -0.5:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(t, ymax, "o-", ms=3)
    ax.set(xlabel="t [s]", ylabel="ymax [m]", title="P = 2000 N")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ymax_vs_time.png", dpi=150)
    plt.close(fig)

    t, Q, ymax, _, _ = runSupportedBeam(20000.0)
    stretch = np.linalg.norm(np.diff(Q[-1].reshape(-1, 2), axis=0), axis=1) - p["dL"]
    print(f"  Q9: |ymax|(1 s) = {abs(ymax[-1]):.4f} m  (EB: {ebDeflection(20000.0, EI):.4f})")
    print(f"  Q11: x_N = {Q[-1, -2]:.4f} m (was {L2}), max spring stretch = {np.abs(stretch).max():.3e} m "
          f"({np.abs(stretch).max() / p['dL'] * 100:.4f}% strain)")

    print("  Q10: e(P) = (y_EB - |y_sim|)/y_EB")
    loads = [20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 10000, 15000, 20000]
    sim, eb = [], []
    for P in loads:
        _, _, ymax, _, _ = runSupportedBeam(float(P))
        sim.append(abs(ymax[-1]))
        eb.append(ebDeflection(P, EI))
        print(f"       P = {P:6d} N: sim = {sim[-1]:.4f} m, EB = {eb[-1]:.4f} m, e = {(eb[-1] - sim[-1]) / eb[-1] * 100:6.2f}%")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loads, sim, "o-", label="simulation")
    ax.plot(loads, eb, "s--", label="Euler-Bernoulli")
    ax.set(xlabel="P [N]", ylabel="|ymax| [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ymax_vs_load.png", dpi=150)
    plt.close(fig)


def main():
    run_tests()
    part1()
    part2()


if __name__ == "__main__":
    main()
