"""MAE 263F Homework 4: Discrete Elastic Plates and Shells.

Part 1: edges and hinges of a triangle mesh.
Part 2: hinge angle and bending energy Eb = 0.5*kb*(theta - thetaBar)^2 of one hinge.
Part 3: finite-difference check of the analytic bending gradient.
Part 4: time integration (conceptual); small undamped spring-mass runs back up the answers.

Everything is written from scratch (no course files are imported).
"""

import numpy as np

# Hinge used in Parts 2 and 3 (plate: thetaBar = 0, kb = 1 N m)
Q_HINGE = np.array([0.0, 0.0, 0.0,    # x0
                    1.0, 0.0, 0.0,    # x1
                    0.5, 1.0, 0.0,    # x2
                    0.5, -1.0, 1.0])  # x3
THETA_BAR, KB = 0.0, 1.0
EPS = 0.01


# ---------------------------------------------------------------------------
# Part 1: edges and hinges
# ---------------------------------------------------------------------------
def plateMesh(n=3):
    """n x n nodes, node 3j+i at (i, j); each square split along its lower-left -> upper-right diagonal."""
    nodes = np.array([[i, j, 0.0] for j in range(n) for i in range(n)])
    elements = []
    for j in range(n - 1):
        for i in range(n - 1):
            a, b, c, d = n * j + i, n * j + i + 1, n * (j + 1) + i, n * (j + 1) + i + 1
            elements += [[a, b, d], [a, d, c]]
    return nodes, np.array(elements)


def getEdgesAndHinges(elements):
    """Return edges (E, 2) and hinges (H, 4) = [x0, x1, x2, x3], hinge edge x0-x1.

    x2 is the third node of the triangle whose normal (x1-x0) x (x2-x0) follows the
    triangle's node ordering, so that n0 and n1 of every hinge are consistently oriented.
    """
    third = {}  # undirected edge -> list of (directed edge as it appears in the triangle, opposite node)
    for tri in elements:
        for k in range(3):
            a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            third.setdefault(tuple(sorted((a, b))), []).append((a, b, c))
    edges = np.array(sorted(third))
    hinges = []
    for faces in third.values():
        if len(faces) == 2:
            (a, b, c), (_, _, d) = faces
            hinges.append([a, b, c, d])
    return edges, np.array(hinges)


# ---------------------------------------------------------------------------
# Part 2: hinge angle and bending energy
# ---------------------------------------------------------------------------
def _split(q):
    q = np.asarray(q, dtype=float)
    return q[0:3], q[3:6], q[6:9], q[9:12]


def getTheta(q):
    """Signed angle from n0 to n1 about e0 = x1 - x0."""
    x0, x1, x2, x3 = _split(q)
    e0 = x1 - x0
    n0 = np.cross(e0, x2 - x0)
    n1 = np.cross(x3 - x0, e0)
    w = np.cross(n0, n1)
    theta = np.arctan2(np.linalg.norm(w), n0 @ n1)
    return theta if w @ e0 >= 0 else -theta


def Eb(q, thetaBar=THETA_BAR, kb=KB):
    """Eq. (1): bending energy of one hinge."""
    return 0.5 * kb * (getTheta(q) - thetaBar) ** 2


# ---------------------------------------------------------------------------
# Part 3: analytic gradient and finite-difference check
# ---------------------------------------------------------------------------
def gradTheta(q):
    """Analytic d(theta)/dq (size 12).

    Moving x2 (x3) along its triangle normal rotates that triangle about the hinge:
    d(theta)/dx2 = -n0_hat/h0 and d(theta)/dx3 = -n1_hat/h1, where h is the height of the
    triangle over the hinge edge. Translation and rotation invariance give x0 and x1.
    """
    x0, x1, x2, x3 = _split(q)
    e0 = x1 - x0
    l2 = e0 @ e0
    n0 = np.cross(e0, x2 - x0)
    n1 = np.cross(x3 - x0, e0)
    g2 = -n0 * np.sqrt(l2) / (n0 @ n0)  # -n0_hat / h0, h0 = |n0| / |e0|
    g3 = -n1 * np.sqrt(l2) / (n1 @ n1)
    a2 = (x2 - x0) @ e0 / l2  # position of each tip's projection along the hinge
    a3 = (x3 - x0) @ e0 / l2
    g1 = -(a2 * g2 + a3 * g3)
    g0 = -((1 - a2) * g2 + (1 - a3) * g3)
    return np.concatenate([g0, g1, g2, g3])


def gradEb(q, thetaBar=THETA_BAR, kb=KB):
    """dEb/dq = kb*(theta - thetaBar)*d(theta)/dq (plays the role of dF from gradEb_hessEb_Shell)."""
    return kb * (getTheta(q) - thetaBar) * gradTheta(q)


def perturbedEnergies(q, eps=EPS):
    """Eb(q + eps*e_i) for i = 0..11."""
    return np.array([Eb(q + eps * np.eye(12)[i]) for i in range(12)])


def gradFD(q, eps=EPS):
    """Eq. (2): forward finite difference."""
    return (perturbedEnergies(q, eps) - Eb(q)) / eps


# ---------------------------------------------------------------------------
# Part 4: undamped spring-mass (m = k = 1) for the time-integration questions
# ---------------------------------------------------------------------------
def implicitEuler(dt=0.1, steps=1000):
    x, v = 1.0, 0.0
    E = [0.5 * (x**2 + v**2)]
    for _ in range(steps):
        x = (x + dt * v) / (1 + dt**2)  # x_{n+1} = x_n + dt*v_{n+1}, v_{n+1} = v_n - dt*x_{n+1}
        v = v - dt * x
        E.append(0.5 * (x**2 + v**2))
    return np.array(E)


def newmark(beta, gamma, dt=0.1, steps=1000):
    x, v = 1.0, 0.0
    a = -x
    E = [0.5 * (x**2 + v**2)]
    for _ in range(steps):
        x_new = (x + dt * v + dt**2 * (0.5 - beta) * a) / (1 + beta * dt**2)
        a_new = -x_new
        v = v + dt * ((1 - gamma) * a + gamma * a_new)
        x, a = x_new, a_new
        E.append(0.5 * (x**2 + v**2))
    return np.array(E)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def run_tests():
    rng = np.random.default_rng(0)

    # 1: mesh counts satisfy Euler's formula for a disk (V - E + F = 1); boundary edges = 4(n-1)
    for n in (2, 3, 5):
        nodes, elems = plateMesh(n)
        edges, hinges = getEdgesAndHinges(elems)
        assert len(nodes) - len(edges) + len(elems) == 1
        assert len(edges) - len(hinges) == 4 * (n - 1)

    # 2: a flat mesh has theta = 0 at every hinge (consistent orientation)
    nodes, elems = plateMesh(3)
    for h in getEdgesAndHinges(elems)[1]:
        assert abs(getTheta(nodes[h].ravel())) < 1e-12

    # 3: folding x3 about the hinge by a known angle gives that angle, with the right sign
    for phi in (0.3, -1.0, 2.5):
        x3 = np.array([0.5, -np.cos(phi), np.sin(phi)])
        q = np.concatenate([[0, 0, 0, 1, 0, 0, 0.5, 1, 0], x3])
        n0, n1 = np.array([0, 0, 1.0]), np.cross(x3, [1.0, 0, 0])
        sign = np.sign(np.cross(n0, n1) @ [1.0, 0, 0])
        assert np.isclose(getTheta(q), sign * abs(phi))

    # 4: analytic gradient matches a central finite difference on random hinges
    h = 1e-6
    for _ in range(20):
        q = rng.normal(size=12)
        fd = np.array([(Eb(q + h * np.eye(12)[i], 0.3, 2.0) - Eb(q - h * np.eye(12)[i], 0.3, 2.0)) / (2 * h)
                       for i in range(12)])
        assert np.allclose(gradEb(q, 0.3, 2.0), fd, rtol=1e-5, atol=1e-7), np.abs(gradEb(q, 0.3, 2.0) - fd).max()

    # 5: forward-difference error shrinks linearly with eps
    e1 = np.abs(gradFD(Q_HINGE, 1e-2) - gradEb(Q_HINGE)).max()
    e2 = np.abs(gradFD(Q_HINGE, 1e-3) - gradEb(Q_HINGE)).max()
    assert 8 < e1 / e2 < 12

    # 6: gradient is invariant to rigid motion (sums to zero force and zero torque)
    q = rng.normal(size=12)
    g = gradEb(q).reshape(4, 3)
    assert np.allclose(g.sum(axis=0), 0, atol=1e-12)
    assert np.allclose(np.cross(q.reshape(4, 3), g).sum(axis=0), 0, atol=1e-12)

    print("All 6 test cases passed.\n")


# ---------------------------------------------------------------------------
# Gradescope questions
# ---------------------------------------------------------------------------
def fmt(v, p=3):
    v = np.where(np.abs(v) < 0.5 * 10**-p, 0.0, v)
    return "[ " + " | ".join(", ".join(f"{c:.{p}f}" for c in v[3 * k:3 * k + 3]) for k in range(4)) + " ]"


def part1():
    print("Part 1: Edges and hinges")
    _, elems = plateMesh(3)
    edges, hinges = getEdgesAndHinges(elems)
    print(f"  Q1: {len(edges)} edges and {len(hinges)} hinges  -> (c)")
    print(f"       hinges [x0, x1, x2, x3]: {hinges.tolist()}")


def part2():
    print("Part 2: Hinge angle and bending energy")
    theta, E0 = getTheta(Q_HINGE), Eb(Q_HINGE)
    print(f"  Q2: theta = {theta:.6f} rad = {theta / np.pi:+.4f} pi, Eb = {E0:.5f} J  -> (b)")

    Ep = perturbedEnergies(Q_HINGE)
    answers = {0: "(a)", 1: "(d)", 2: "(c)", 3: "(b)"}
    for k in range(4):
        vals = ", ".join(f"{e:.5f}" for e in Ep[3 * k:3 * k + 3])
        print(f"  Q{k + 3}: moving x{k} by eps in x, y, z: Eb = {vals} J  -> {answers[k]}")
    return Ep


def part3():
    print("Part 3: Finite-difference check of the gradient")
    gFD, dF = gradFD(Q_HINGE), gradEb(Q_HINGE)
    print(f"  Q7: gradFD = {fmt(gFD)} N  -> (d)")
    print(f"  Q8: dF     = {fmt(dF)} N  -> (a)")
    print(f"       max |gradFD - dF| = {np.abs(gFD - dF).max():.2e} N (O(eps) forward-difference error)")


def part4():
    print("Part 4: Time integration (undamped spring-mass, m = k = 1, dt = 0.1, 100 s)")
    E = implicitEuler()
    print(f"  Q9: implicit Euler: E(100 s)/E(0) = {E[-1] / E[0]:.2e} (numerical damping)  -> (c)")
    for beta, gamma in ((0.5, 1.0), (0.0, 0.0), (0.25, 0.5)):
        E = newmark(beta, gamma)
        print(f"  Q10: Newmark beta = {beta}, gamma = {gamma}: E(100 s)/E(0) = {E[-1] / E[0]:.3e}, "
              f"max |E/E0 - 1| = {np.abs(E / E[0] - 1).max():.2e}")
    print("       -> (d) beta = 0.25, gamma = 0.5 (trapezoidal rule) conserves energy")
    print("  Q11: the corrector is needed only when the contact set changes (a free node penetrates the")
    print("       ground, or a fixed node has an adhesive reaction force)  -> (a)")


def main():
    run_tests()
    part1()
    part2()
    part3()
    part4()


if __name__ == "__main__":
    main()
