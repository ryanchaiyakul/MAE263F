"""MAE 263F Homework 1: modular multi-mass-spring-damper simulator.

Node 0 is the fixed wall. Spring/damper i (i = 1..N) connects node i-1 and
node i. Parameter arrays m, k, b have length N+1 with entry 0 unused.
"""

import numpy as np

PATTERN = np.array([[1.0, -1.0], [-1.0, 1.0]])


# ---------------------------------------------------------------------------
# Element functions
# ---------------------------------------------------------------------------
def gradEs(xa, xb, k):
    """Gradient of spring energy 0.5*k*(xb-xa)^2 w.r.t. (xa, xb)."""
    return k * (xb - xa) * np.array([-1.0, 1.0])


def hessEs(k):
    """Hessian of spring energy w.r.t. (xa, xb)."""
    return k * PATTERN


def dampingForce(xa_new, xb_new, xa_old, xb_old, b, dt):
    """Damper force vector g_i using implicit Euler velocities."""
    ua = (xa_new - xa_old) / dt
    ub = (xb_new - xb_old) / dt
    return b * (ub - ua) * np.array([-1.0, 1.0])


def dampingJacobian(b, dt):
    """Derivative of dampingForce w.r.t. (xa_new, xb_new)."""
    return (b / dt) * PATTERN


# ---------------------------------------------------------------------------
# Assembly and time stepping
# ---------------------------------------------------------------------------
def getFJ(x_new, x_old, u_old, params):
    """Residual f (N+1) and Jacobian J (N+1 x N+1), Eqs. (7)-(8)."""
    m, k, b = params["m"], params["k"], params["b"]
    g, dt = params["g"], params["dt"]
    N = len(x_new) - 1

    # Inertia and gravity (node by node)
    f = m / dt * ((x_new - x_old) / dt - u_old) - m * g
    J = np.diag(m / dt**2)

    # Springs and dampers (element by element)
    for i in range(1, N + 1):
        idx = [i - 1, i]
        f[idx] += gradEs(x_new[i - 1], x_new[i], k[i])
        J[np.ix_(idx, idx)] += hessEs(k[i])
        f[idx] += dampingForce(x_new[i - 1], x_new[i], x_old[i - 1], x_old[i], b[i], dt)
        J[np.ix_(idx, idx)] += dampingJacobian(b[i], dt)

    return f, J


def implicitEulerStep(x_old, u_old, params, tol=1e-9, max_iter=100, return_iters=False):
    """One implicit Euler step solved with Newton's method (Algorithm 1)."""
    dt = params["dt"]
    N = len(x_old) - 1
    free = np.arange(1, N + 1)

    x_new = x_old.copy()
    f, _ = getFJ(x_new, x_old, u_old, params)
    scale = 1.0 + np.linalg.norm(f[free])  # relative tolerance (f ~ m/dt^2 is large for small dt)
    for n_iter in range(1, max_iter + 1):
        f, J = getFJ(x_new, x_old, u_old, params)
        dx = np.linalg.solve(J[np.ix_(free, free)], f[free])
        x_new[free] -= dx
        f, _ = getFJ(x_new, x_old, u_old, params)
        err = np.linalg.norm(f[free])
        if err < tol * scale:
            break
    else:
        raise RuntimeError(f"Newton did not converge (err = {err:.3e})")

    u_new = (x_new - x_old) / dt
    if return_iters:
        return x_new, u_new, n_iter, err
    return x_new, u_new


def explicitEulerStep(x_old, u_old, params):
    """One forward Euler step, Eqs. (10)-(12)."""
    m, k, b = params["m"], params["k"], params["b"]
    g, dt = params["g"], params["dt"]
    N = len(x_old) - 1

    # Internal force sum: sum of grad E_i + sum of g_i, at old values
    F_int = np.zeros(N + 1)
    for i in range(1, N + 1):
        idx = [i - 1, i]
        F_int[idx] += gradEs(x_old[i - 1], x_old[i], k[i])
        F_int[idx] += b[i] * (u_old[i] - u_old[i - 1]) * np.array([-1.0, 1.0])

    a = np.zeros(N + 1)
    a[1:] = g - F_int[1:] / m[1:]

    x_new = x_old + dt * u_old
    u_new = u_old + dt * a
    x_new[0] = 0.0
    u_new[0] = 0.0
    return x_new, u_new


def simulate(m, k, b, dt, T, g=9.8, step=implicitEulerStep):
    """Run Algorithm 2. m, k, b are lists for i = 1..N (no wall entry).

    Returns times (n+1,), X (n+1, N+1), U (n+1, N+1).
    """
    params = {
        "m": np.concatenate([[0.0], np.asarray(m, dtype=float)]),
        "k": np.concatenate([[0.0], np.asarray(k, dtype=float)]),
        "b": np.concatenate([[0.0], np.asarray(b, dtype=float)]),
        "g": g,
        "dt": dt,
    }
    N = len(m)
    n_steps = int(round(T / dt))

    X = np.zeros((n_steps + 1, N + 1))
    U = np.zeros((n_steps + 1, N + 1))
    for n in range(n_steps):
        X[n + 1], U[n + 1] = step(X[n], U[n], params)

    times = np.arange(n_steps + 1) * dt
    return times, X, U


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    # 1-4: element functions
    assert np.allclose(gradEs(0, 0.1, 10), [-1, 1])
    assert np.allclose(hessEs(10), [[10, -10], [-10, 10]])
    assert np.allclose(dampingForce(0, 0.01, 0, 0, 2, 0.01), [-2, 2])
    assert np.allclose(dampingJacobian(2, 0.01), [[200, -200], [-200, 200]])

    # 5: assembly, stiffness only
    p5 = {"m": np.zeros(4), "k": np.array([0.0, 1, 2, 3]), "b": np.zeros(4), "g": 0.0, "dt": 1.0}
    _, J = getFJ(np.zeros(4), np.zeros(4), np.zeros(4), p5)
    assert np.allclose(J[1:, 1:], [[3, -2, 0], [-2, 5, -3], [0, -3, 3]])

    # 6: assembly, all terms
    p6 = {
        "m": np.array([0.0, 1, 2, 3]),
        "k": np.array([0.0, 10, 20, 30]),
        "b": np.array([0.0, 1, 2, 3]),
        "g": 9.8,
        "dt": 0.1,
    }
    x_old6 = np.zeros(4)
    u_old6 = np.array([0.0, 0.1, 0.2, 0.3])
    x_new6 = np.array([0.0, 0.01, 0.03, 0.06])
    f, J = getFJ(x_new6, x_old6, u_old6, p6)
    assert np.allclose(f[1:], [-10.4, -18.6, -18.6])
    assert np.allclose(J[1:, 1:], [[160, -40, 0], [-40, 300, -60], [0, -60, 360]])

    # 7: one implicit step, free fall
    p7 = {"m": np.array([0.0, 1]), "k": np.zeros(2), "b": np.zeros(2), "g": 9.8, "dt": 0.01}
    x, u, it7, err7 = implicitEulerStep(np.zeros(2), np.zeros(2), p7, return_iters=True)
    assert np.allclose(x, [0, 0.00098]) and np.allclose(u, [0, 0.098])

    # 8: one implicit step, full chain
    x, u, it8, err8 = implicitEulerStep(x_old6, u_old6, p6, return_iters=True)
    assert np.allclose(x, [0, 0.096696, 0.116786, 0.126131], atol=1e-6)
    assert np.allclose(u, [0, 0.966964, 1.167857, 1.261310], atol=1e-6)

    # 9: Newton converges in one iteration
    assert it7 == 1 and it8 == 1, (it7, it8)
    assert err7 < 1e-10 and err8 < 1e-10

    # 10: one explicit step, single spring
    p10 = {"m": np.array([0.0, 1]), "k": np.array([0.0, 1]), "b": np.zeros(2), "g": 0.0, "dt": 0.1}
    x, u = explicitEulerStep(np.array([0.0, 1]), np.zeros(2), p10)
    assert np.allclose(x, [0, 1]) and np.allclose(u, [0, -0.1])

    # 11: one explicit step, full chain
    x, u = explicitEulerStep(x_new6, u_old6, p6)
    assert np.allclose(x, [0, 0.02, 0.05, 0.09]) and np.allclose(u, [0, 1.12, 1.21, 1.24])

    print("All 11 test cases passed.\n")


# ---------------------------------------------------------------------------
# Gradescope questions
# ---------------------------------------------------------------------------
def at_time(times, Y, t):
    return Y[int(np.argmin(np.abs(times - t)))]


def answer_questions():
    g = 9.8

    print("Part A: Free fall")
    t, X, _ = simulate([1], [0], [0], dt=1e-3, T=1)
    print(f"  Q1: x1(1 s)   = {at_time(t, X, 1)[1]:.4f} m   (analytical {0.5 * g:.4f})")
    t, X, _ = simulate([1, 2, 3], [0, 0, 0], [0, 0, 0], dt=1e-3, T=2)
    print(f"  Q2: x3(2 s)   = {at_time(t, X, 2)[3]:.4f} m   (analytical {0.5 * g * 4:.4f})")

    print("Part B: Terminal velocity")
    t, _, U = simulate([1], [0], [2], dt=0.01, T=20)
    print(f"  Q3: u1(20 s)  = {at_time(t, U, 20)[1]:.4f} m/s (analytical {g / 2:.4f})")
    t, _, U = simulate([1, 1], [0, 0], [2, 2], dt=0.01, T=20)
    print(f"  Q4: u2(20 s)  = {at_time(t, U, 20)[2]:.4f} m/s (analytical {2 * g / 2 + g / 2:.4f})")

    print("Part C: Static equilibrium")
    t, X, _ = simulate([1], [10], [1], dt=0.01, T=50)
    print(f"  Q5: x1(50 s)  = {at_time(t, X, 50)[1]:.4f} m   (analytical {g / 10:.4f})")
    t, X, _ = simulate([1] * 5, [10] * 5, [5] * 5, dt=0.01, T=100)
    print(f"  Q6: x5(100 s) = {at_time(t, X, 100)[5]:.4f} m   (analytical {g / 10 * 15:.4f})")
    t, X, _ = simulate([1, 2, 3], [10, 20, 30], [5, 10, 15], dt=0.01, T=100)
    x3_exact = g * (6 / 10 + 5 / 20 + 3 / 30)
    print(f"  Q7: x3(100 s) = {at_time(t, X, 100)[3]:.4f} m   (analytical {x3_exact:.4f})")

    print("Part D: Artificial damping in implicit Euler (N=1, m=1, k=1, b=0)")
    for dt in (0.1, 0.01):
        t, X, _ = simulate([1], [1], [0], dt=dt, T=100)
        late = X[t >= 90, 1]
        amp = 0.5 * (late.max() - late.min())
        print(f"  dt = {dt}: x1(100 s) = {X[-1, 1]:.4f} m, oscillation amplitude over 90-100 s = {amp:.4f} m")
    print("  Q8: see dt = 0.1 line; Q9: compare the two amplitudes")

    print("Part E: Explicit Euler (N=1, m=1, k=1, b=0, dt=0.1)")
    t, X, _ = simulate([1], [1], [0], dt=0.1, T=100, step=explicitEulerStep)
    print(f"  Q10: max|x1 - 9.8| over 0-10 s = {np.abs(X[t <= 10, 1] - g).max():.3f} m, "
          f"over 90-100 s = {np.abs(X[t >= 90, 1] - g).max():.3f} m")
    print(f"  Q11: max x1 over 0-10 s = {X[t <= 10, 1].max():.4f} m")


def main():
    run_tests()
    answer_questions()


if __name__ == "__main__":
    main()
