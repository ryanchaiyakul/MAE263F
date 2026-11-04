"""MAE 263F Homework 3: 3D Discrete Elastic Rods (DER).

DOF vector q = [x1, theta1, x2, theta2, ..., theta_{N-1}, xN] (size 4N - 1).
Stretching, bending and twisting energies follow Chapters 7-8 of the notes;
the reference frame is updated by time-parallel transport and the equations of
motion are integrated with implicit Euler (Newton's method on the free DOFs).

Part 1: straight clamped-free rod under gravity vs. Euler-Bernoulli theory.
Part 2: naturally curved rod (and the same rod with its natural curvature removed).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

# Default parameters (Part 1)
L_FREE, R0, E_DEF, NU, RHO, G_DEF = 0.2, 1e-3, 10e6, 0.5, 1000.0, 9.81
NAT_R = 0.02  # natural radius of the curved rod (Part 2)
ARC_LEN = 0.2  # arc length of the curved rod (Part 2)


# ---------------------------------------------------------------------------
# Frame utilities
# ---------------------------------------------------------------------------
def crossMat(a):
    return np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def parallelTransport(u, t1, t2):
    """Transport u from tangent t1 to tangent t2 (rotation about t1 x t2)."""
    b = np.cross(t1, t2)
    nb = np.linalg.norm(b)
    if nb < 1e-14:
        return u.copy()
    b = b / nb
    n1 = np.cross(t1, b)
    n2 = np.cross(t2, b)
    return (u @ t1) * t2 + (u @ n1) * n2 + (u @ b) * b


def signedAngle(u, v, n):
    w = np.cross(u, v)
    ang = np.arctan2(np.linalg.norm(w), u @ v)
    return -ang if n @ w < 0 else ang


def rotateAxisAngle(v, z, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * v + s * np.cross(z, v) + (z @ v) * (1 - c) * z


def nodesOf(q):
    return np.vstack([q[0::4], q[1::4], q[2::4]]).T


def tangents(q):
    e = np.diff(nodesOf(q), axis=0)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def spaceParallelFrame(q):
    """Reference frame at t=0: a1 of the first edge, then space-parallel transport."""
    t = tangents(q)
    a1 = np.zeros_like(t)
    a1[0] = np.cross(t[0], [0.0, 0.0, -1.0])
    if np.linalg.norm(a1[0]) < 1e-6:
        a1[0] = np.cross(t[0], [0.0, 1.0, 0.0])
    a1[0] /= np.linalg.norm(a1[0])
    for k in range(1, len(t)):
        a1[k] = parallelTransport(a1[k - 1], t[k - 1], t[k])
        a1[k] /= np.linalg.norm(a1[k])
    a2 = np.cross(t, a1)
    return a1, a2


def timeParallelFrame(a1_old, t_old, q):
    """Transport each edge's a1 from its old tangent to its new tangent."""
    t = tangents(q)
    a1 = np.array([parallelTransport(a1_old[k], t_old[k], t[k]) for k in range(len(t))])
    a1 /= np.linalg.norm(a1, axis=1, keepdims=True)
    return a1, np.cross(t, a1)


def materialFrame(a1, a2, theta):
    c, s = np.cos(theta)[:, None], np.sin(theta)[:, None]
    return c * a1 + s * a2, -s * a1 + c * a2


def referenceTwist(a1, t, refTwist_old):
    """Reference twist at each interior node (index c = 1..N-2)."""
    rt = np.zeros(len(t) + 1)
    for c in range(1, len(t)):
        u = parallelTransport(a1[c - 1], t[c - 1], t[c])
        u = rotateAxisAngle(u, t[c], refTwist_old[c])
        rt[c] = refTwist_old[c] + signedAngle(u, a1[c], t[c])
    return rt


def curvatureBinormal(te, tf):
    return 2.0 * np.cross(te, tf) / (1.0 + te @ tf)


def getKappa(q, m1, m2):
    """Material curvatures (kappa1, kappa2) at each interior node."""
    x = nodesOf(q)
    kappa = np.zeros((len(x), 2))
    for c in range(1, len(x) - 1):
        te = (x[c] - x[c - 1]) / np.linalg.norm(x[c] - x[c - 1])
        tf = (x[c + 1] - x[c]) / np.linalg.norm(x[c + 1] - x[c])
        kb = curvatureBinormal(te, tf)
        kappa[c, 0] = 0.5 * kb @ (m2[c - 1] + m2[c])
        kappa[c, 1] = -0.5 * kb @ (m1[c - 1] + m1[c])
    return kappa


# ---------------------------------------------------------------------------
# Elastic energies: gradient and Hessian
# ---------------------------------------------------------------------------
def gradEs_hessEs(x0, x1, l_k, EA):
    """Es = 0.5*EA*(1 - |e|/l_k)^2*l_k; DOFs (x0, x1)."""
    d = x1 - x0
    l = np.linalg.norm(d)
    t = d / l
    dl = np.concatenate([-t, t])
    A = (np.eye(3) - np.outer(t, t)) / l
    d2l = np.block([[A, -A], [-A, A]])
    return EA * (l / l_k - 1.0) * dl, EA / l_k * np.outer(dl, dl) + EA * (l / l_k - 1.0) * d2l


def _blockAssemble(Dee, Dff, Def, Dte=None, Dtf=None, Dfe_te=None, Dff_tf=None):
    """Map edge derivatives (e = x1 - x0, f = x2 - x1) to the 11 element DOFs."""
    Dfe = Def.T
    H = np.zeros((11, 11))
    i0, i1, i2 = slice(0, 3), slice(4, 7), slice(8, 11)
    H[i0, i0] = Dee
    H[i0, i1] = -Dee + Def
    H[i1, i0] = -Dee + Dfe
    H[i1, i1] = Dee - Def - Dfe + Dff
    H[i0, i2] = -Def
    H[i2, i0] = -Dfe
    H[i1, i2] = Def - Dff
    H[i2, i1] = Dfe - Dff
    H[i2, i2] = Dff
    return H


def gradEb_hessEb(x0, x1, x2, m1e, m2e, m1f, m2f, kappaBar, l_k, EI):
    """Bending energy Eb = 0.5*EI/l_k*|kappa - kappaBar|^2 on DOFs (x0, th_e, x1, th_f, x2)."""
    ee, ef = x1 - x0, x2 - x1
    ne, nf = np.linalg.norm(ee), np.linalg.norm(ef)
    te, tf = ee / ne, ef / nf
    chi = 1.0 + te @ tf
    kb = 2.0 * np.cross(te, tf) / chi
    tt = (te + tf) / chi
    d1t = (m1e + m1f) / chi
    d2t = (m2e + m2f) / chi
    k1 = 0.5 * kb @ (m2e + m2f)
    k2 = -0.5 * kb @ (m1e + m1f)

    Dk1De = (-k1 * tt + np.cross(tf, d2t)) / ne
    Dk1Df = (-k1 * tt - np.cross(te, d2t)) / nf
    Dk2De = (-k2 * tt - np.cross(tf, d1t)) / ne
    Dk2Df = (-k2 * tt + np.cross(te, d1t)) / nf

    gk1, gk2 = np.zeros(11), np.zeros(11)
    gk1[0:3], gk1[4:7], gk1[8:11] = -Dk1De, Dk1De - Dk1Df, Dk1Df
    gk2[0:3], gk2[4:7], gk2[8:11] = -Dk2De, Dk2De - Dk2Df, Dk2Df
    gk1[3], gk1[7] = -0.5 * kb @ m1e, -0.5 * kb @ m1f
    gk2[3], gk2[7] = -0.5 * kb @ m2e, -0.5 * kb @ m2f

    I3 = np.eye(3)
    ttott = np.outer(tt, tt)
    ne2, nf2 = ne**2, nf**2

    # Hessian of kappa1
    A = np.outer(np.cross(tf, d2t), tt)
    B = np.outer(np.cross(te, d2t), tt)
    D2k1De2 = (2 * k1 * ttott - A - A.T) / ne2 - k1 / (chi * ne2) * (I3 - np.outer(te, te)) \
        + (np.outer(kb, m2e) + np.outer(m2e, kb)) / (4 * ne2)
    D2k1Df2 = (2 * k1 * ttott + B + B.T) / nf2 - k1 / (chi * nf2) * (I3 - np.outer(tf, tf)) \
        + (np.outer(kb, m2f) + np.outer(m2f, kb)) / (4 * nf2)
    D2k1DeDf = -k1 / (chi * ne * nf) * (I3 + np.outer(te, tf)) \
        + (2 * k1 * ttott - A + B.T - crossMat(d2t)) / (ne * nf)

    # Hessian of kappa2
    C = np.outer(np.cross(tf, d1t), tt)
    D = np.outer(np.cross(te, d1t), tt)
    D2k2De2 = (2 * k2 * ttott + C + C.T) / ne2 - k2 / (chi * ne2) * (I3 - np.outer(te, te)) \
        - (np.outer(kb, m1e) + np.outer(m1e, kb)) / (4 * ne2)
    D2k2Df2 = (2 * k2 * ttott - D - D.T) / nf2 - k2 / (chi * nf2) * (I3 - np.outer(tf, tf)) \
        - (np.outer(kb, m1f) + np.outer(m1f, kb)) / (4 * nf2)
    D2k2DeDf = -k2 / (chi * ne * nf) * (I3 + np.outer(te, tf)) \
        + (2 * k2 * ttott + C - D.T + crossMat(d1t)) / (ne * nf)

    H1 = _blockAssemble(D2k1De2, D2k1Df2, D2k1DeDf)
    H2 = _blockAssemble(D2k2De2, D2k2Df2, D2k2DeDf)

    # theta-theta and theta-position terms
    H1[3, 3], H1[7, 7] = -0.5 * kb @ m2e, -0.5 * kb @ m2f
    H2[3, 3], H2[7, 7] = 0.5 * kb @ m1e, 0.5 * kb @ m1f
    for H, ma, mb in ((H1, m1e, m1f), (H2, m2e, m2f)):
        DeTe = (0.5 * (kb @ ma) * tt - np.cross(tf, ma) / chi) / ne
        DeTf = (0.5 * (kb @ mb) * tt - np.cross(tf, mb) / chi) / ne
        DfTe = (0.5 * (kb @ ma) * tt + np.cross(te, ma) / chi) / nf
        DfTf = (0.5 * (kb @ mb) * tt + np.cross(te, mb) / chi) / nf
        for col, De, Df in ((3, DeTe, DfTe), (7, DeTf, DfTf)):
            H[0:3, col] = H[col, 0:3] = -De
            H[4:7, col] = H[col, 4:7] = De - Df
            H[8:11, col] = H[col, 8:11] = Df

    dk1, dk2 = k1 - kappaBar[0], k2 - kappaBar[1]
    c = EI / l_k
    grad = c * (dk1 * gk1 + dk2 * gk2)
    hess = c * (np.outer(gk1, gk1) + np.outer(gk2, gk2) + dk1 * H1 + dk2 * H2)
    return grad, hess


def gradEt_hessEt(x0, x1, x2, theta_e, theta_f, refTwist, twistBar, l_k, GJ):
    """Twisting energy Et = 0.5*GJ/l_k*(m - twistBar)^2, m = theta_f - theta_e + refTwist."""
    ee, ef = x1 - x0, x2 - x1
    ne, nf = np.linalg.norm(ee), np.linalg.norm(ef)
    te, tf = ee / ne, ef / nf
    chi = 1.0 + te @ tf
    kb = 2.0 * np.cross(te, tf) / chi
    tt = (te + tf) / chi

    gm = np.zeros(11)
    gm[0:3] = -0.5 * kb / ne
    gm[8:11] = 0.5 * kb / nf
    gm[4:7] = -(gm[0:3] + gm[8:11])
    gm[3], gm[7] = -1.0, 1.0

    D2mDe2 = -0.25 / ne**2 * (np.outer(kb, te + tt) + np.outer(te + tt, kb))
    D2mDf2 = -0.25 / nf**2 * (np.outer(kb, tf + tt) + np.outer(tf + tt, kb))
    D2mDeDf = 0.5 / (ne * nf) * (2 / chi * crossMat(te) - np.outer(kb, tt))
    Hm = _blockAssemble(D2mDe2, D2mDf2, D2mDeDf)

    dm = theta_f - theta_e + refTwist - twistBar
    c = GJ / l_k
    return c * dm * gm, c * (np.outer(gm, gm) + dm * Hm)


# ---------------------------------------------------------------------------
# Rod model
# ---------------------------------------------------------------------------
class Rod:
    """Holds geometry, stiffness, mass, and the undeformed (natural) state."""

    def __init__(self, nodes, E, gval, dl_mass, zeroNaturalCurvature=False):
        N = len(nodes)
        self.N, self.ndof = N, 4 * N - 1
        A = np.pi * R0**2
        G = E / (2 * (1 + NU))
        self.EA, self.EI, self.GJ = E * A, E * np.pi * R0**4 / 4, G * np.pi * R0**4 / 2

        q0 = np.zeros(self.ndof)
        for k in range(N):
            q0[4 * k:4 * k + 3] = nodes[k]
        self.q0 = q0

        edges = np.diff(nodes, axis=0)
        self.refLen = np.linalg.norm(edges, axis=1)
        self.voronoiLen = np.zeros(N)
        self.voronoiLen[1:-1] = 0.5 * (self.refLen[:-1] + self.refLen[1:])

        # Lumped mass: rho*A*dl per edge split between its nodes; rotational inertia for theta
        dm = RHO * A * dl_mass
        m = np.zeros(self.ndof)
        for k in range(N):
            w = 0.5 if k in (0, N - 1) else 1.0
            m[4 * k:4 * k + 3] = w * dm
        m[3::4] = 0.5 * dm * R0**2
        self.m = m

        self.W = np.zeros(self.ndof)
        for k in range(N):
            self.W[4 * k + 2] = -m[4 * k] * gval  # gravity along -z

        # Natural state from the initial configuration
        self.a1, self.a2 = spaceParallelFrame(q0)
        self.tan = tangents(q0)
        self.refTwist = np.zeros(N)
        self.refTwist = referenceTwist(self.a1, self.tan, self.refTwist)
        m1, m2 = materialFrame(self.a1, self.a2, q0[3::4])
        self.kappaBar = getKappa(q0, m1, m2)
        if zeroNaturalCurvature:
            self.kappaBar = np.zeros_like(self.kappaBar)  # remove the natural curvature
        self.twistBar = np.zeros(N)

        self.fixed = np.arange(0, 7)  # first two nodes and first twist angle
        self.free = np.arange(7, self.ndof)
        self.tol = self.EI / (L_FREE**2) * 1e-6  # well below the per-node weight even at g = 0.05

    def elasticFJ(self, q, a1, a2, refTwist):
        """Gradient and Hessian of the total elastic energy."""
        F = np.zeros(self.ndof)
        J = np.zeros((self.ndof, self.ndof))
        x = nodesOf(q)
        theta = q[3::4]
        m1, m2 = materialFrame(a1, a2, theta)

        for k in range(self.N - 1):
            idx = np.r_[4 * k:4 * k + 3, 4 * k + 4:4 * k + 7]
            g, h = gradEs_hessEs(x[k], x[k + 1], self.refLen[k], self.EA)
            F[idx] += g
            J[np.ix_(idx, idx)] += h

        for c in range(1, self.N - 1):
            idx = np.arange(4 * c - 4, 4 * c + 7)
            g, h = gradEb_hessEb(x[c - 1], x[c], x[c + 1], m1[c - 1], m2[c - 1], m1[c], m2[c],
                                 self.kappaBar[c], self.voronoiLen[c], self.EI)
            F[idx] += g
            J[np.ix_(idx, idx)] += h
            g, h = gradEt_hessEt(x[c - 1], x[c], x[c + 1], theta[c - 1], theta[c], refTwist[c],
                                 self.twistBar[c], self.voronoiLen[c], self.GJ)
            F[idx] += g
            J[np.ix_(idx, idx)] += h
        return F, J

    def step(self, q_old, u_old, dt, max_iter=50):
        """Implicit Euler step; updates the reference frame by time-parallel transport."""
        q = q_old.copy()
        free = self.free
        for _ in range(max_iter):
            a1, a2 = timeParallelFrame(self.a1, self.tan, q)
            refTwist = referenceTwist(a1, tangents(q), self.refTwist)
            Fel, Jel = self.elasticFJ(q, a1, a2, refTwist)
            f = self.m / dt * ((q - q_old) / dt - u_old) + Fel - self.W
            J = np.diag(self.m / dt**2) + Jel
            err = np.linalg.norm(f[free])
            if err < self.tol:
                break
            q[free] -= np.linalg.solve(J[np.ix_(free, free)], f[free])
        else:
            raise RuntimeError(f"Newton did not converge (err = {err:.3e})")
        self.a1, self.a2 = a1, a2
        self.tan = tangents(q)
        self.refTwist = refTwist
        return q, (q - q_old) / dt


def straightNodes(N, L):
    """Eq. (1): x_k = (k - 2) dl, so x_2 is at the origin and x_N at x = L."""
    dl = L / (N - 2)
    nodes = np.zeros((N, 3))
    nodes[:, 0] = (np.arange(N) - 1) * dl
    return nodes, dl


def curvedNodes(N, Rn=NAT_R, arcLen=ARC_LEN):
    dth = arcLen / Rn / (N - 1)
    k = np.arange(N)
    nodes = np.column_stack([Rn * np.cos(k * dth), Rn * np.sin(k * dth), np.zeros(N)])
    return nodes, arcLen / (N - 1)


def run(N=52, L=L_FREE, E=E_DEF, g=G_DEF, natR=0.0, dt=0.1, totalTime=10.0, zeroKappaBar=False):
    """Simulate the rod; returns times, q history (n+1, 4N-1)."""
    if natR == 0:
        nodes, dl = straightNodes(N, L)
    else:
        nodes, dl = curvedNodes(N, natR)
    rod = Rod(nodes, E, g, dl, zeroNaturalCurvature=zeroKappaBar)
    n_steps = int(round(totalTime / dt))
    Q = np.zeros((n_steps + 1, rod.ndof))
    Q[0] = rod.q0
    u = np.zeros(rod.ndof)
    for n in range(n_steps):
        Q[n + 1], u = rod.step(Q[n], u, dt)
    return np.arange(n_steps + 1) * dt, Q


def deltaEB(E=E_DEF, g=G_DEF, L=L_FREE):
    """Eq. (2): Euler-Bernoulli tip deflection qL^4/(8EI)."""
    return RHO * np.pi * R0**2 * g * L**4 / (8 * E * np.pi * R0**4 / 4)


def settledChange(t, y, dt=0.1):
    """Relative change of y over the last second of simulation."""
    n = int(round(1.0 / dt))
    return abs(y[-1] - y[-1 - n]) / max(abs(y[-1]), 1e-30)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
def _elementEnergy(y, base, kappaBar, l_k, EI, GJ):
    """Bending + twisting energy of one element with frames transported from `base`."""
    x0, th_e, x1, th_f, x2 = y[0:3], y[3], y[4:7], y[7], y[8:11]
    a1e0, a1f0, te0, tf0 = base
    te = (x1 - x0) / np.linalg.norm(x1 - x0)
    tf = (x2 - x1) / np.linalg.norm(x2 - x1)
    a1e, a1f = parallelTransport(a1e0, te0, te), parallelTransport(a1f0, tf0, tf)
    a2e, a2f = np.cross(te, a1e), np.cross(tf, a1f)
    m1e, m2e = np.cos(th_e) * a1e + np.sin(th_e) * a2e, -np.sin(th_e) * a1e + np.cos(th_e) * a2e
    m1f, m2f = np.cos(th_f) * a1f + np.sin(th_f) * a2f, -np.sin(th_f) * a1f + np.cos(th_f) * a2f
    refTwist = signedAngle(parallelTransport(a1e, te, tf), a1f, tf)
    return (x0, x1, x2, m1e, m2e, m1f, m2f, th_e, th_f, refTwist)


def _elementGrad(y, base, kappaBar, l_k, EI, GJ, energy=False):
    x0, x1, x2, m1e, m2e, m1f, m2f, th_e, th_f, rt = _elementEnergy(y, base, kappaBar, l_k, EI, GJ)
    gb, hb = gradEb_hessEb(x0, x1, x2, m1e, m2e, m1f, m2f, kappaBar, l_k, EI)
    gt, ht = gradEt_hessEt(x0, x1, x2, th_e, th_f, rt, 0.0, l_k, GJ)
    if energy:
        te = (x1 - x0) / np.linalg.norm(x1 - x0)
        tf = (x2 - x1) / np.linalg.norm(x2 - x1)
        kb = curvatureBinormal(te, tf)
        k = np.array([0.5 * kb @ (m2e + m2f), -0.5 * kb @ (m1e + m1f)])
        return 0.5 * EI / l_k * np.sum((k - kappaBar) ** 2) + 0.5 * GJ / l_k * (th_f - th_e + rt) ** 2
    return gb + gt, hb + ht


def run_tests():
    rng = np.random.default_rng(1)
    y = np.array([0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.05, -0.2, 1.8, 0.6, 0.3])
    te0 = (y[4:7] - y[0:3]) / np.linalg.norm(y[4:7] - y[0:3])
    tf0 = (y[8:11] - y[4:7]) / np.linalg.norm(y[8:11] - y[4:7])
    a1e0 = np.cross(te0, [0, 0, 1.0]); a1e0 /= np.linalg.norm(a1e0)
    a1f0 = rotateAxisAngle(parallelTransport(a1e0, te0, tf0), tf0, 0.3)
    base = (a1e0, a1f0, te0, tf0)
    args = (base, np.array([0.2, -0.1]), 0.9, 2.0, 1.5)
    h = 1e-6

    # 1: bending + twisting gradient matches finite differences of the energy
    g, H = _elementGrad(y, *args)
    g_fd = np.array([(_elementGrad(y + h * np.eye(11)[i], *args, energy=True)
                      - _elementGrad(y - h * np.eye(11)[i], *args, energy=True)) / (2 * h) for i in range(11)])
    assert np.allclose(g, g_fd, rtol=1e-5, atol=1e-7), np.abs(g - g_fd).max()

    # 2: bending + twisting Hessian matches the symmetric part of the finite-difference
    #    Jacobian of the gradient (the transported frame adds a small antisymmetric part)
    H_fd = np.column_stack([(_elementGrad(y + h * np.eye(11)[i], *args)[0]
                             - _elementGrad(y - h * np.eye(11)[i], *args)[0]) / (2 * h) for i in range(11)])
    H_fd = 0.5 * (H_fd + H_fd.T)
    assert np.allclose(H, H_fd, rtol=1e-4, atol=1e-5 * np.abs(H).max()), np.abs(H - H_fd).max()

    # 3: stretching Hessian matches finite differences of its gradient
    x = rng.random(6)
    H = gradEs_hessEs(x[:3], x[3:], 0.4, 3.0)[1]
    H_fd = np.column_stack([(gradEs_hessEs(*np.split(x + h * np.eye(6)[i], 2), 0.4, 3.0)[0]
                             - gradEs_hessEs(*np.split(x - h * np.eye(6)[i], 2), 0.4, 3.0)[0]) / (2 * h)
                            for i in range(6)])
    assert np.allclose(H, H_fd, rtol=1e-5, atol=1e-6)

    # 4: curved rod: integrated natural curvature |kappaBar| = 2 tan(dtheta/2) (~ dl/Rn)
    rod = Rod(curvedNodes(50)[0], E_DEF, G_DEF, ARC_LEN / 49)
    k_exp = 2 * np.tan(ARC_LEN / NAT_R / 49 / 2)
    assert np.allclose(np.linalg.norm(rod.kappaBar[1:-1], axis=1), k_exp, rtol=1e-8)
    assert abs(np.sum(rod.refLen) - 0.1997) < 1e-4

    # 5: straight rod geometry (Eq. 1) and mass per edge rho*A*dl
    rod = Rod(straightNodes(52, L_FREE)[0], E_DEF, G_DEF, L_FREE / 50)
    assert np.isclose(rod.q0[4], 0.0) and np.isclose(rod.q0[-3], L_FREE)
    assert np.isclose(rod.m[0::4].sum(), RHO * np.pi * R0**2 * 51 * L_FREE / 50)
    assert np.allclose(rod.kappaBar, 0)

    # 6: Eq. (2) closed form from the appendix
    assert np.isclose(deltaEB(g=1.0), RHO * 1.0 * L_FREE**4 / (2 * E_DEF * R0**2))

    print("All 6 test cases passed.\n")


# ---------------------------------------------------------------------------
# Gradescope questions
# ---------------------------------------------------------------------------
def tipDeflection(**kwargs):
    t, Q = run(**kwargs)
    d = -Q[:, -1]
    return d[-1], settledChange(t, d), t, Q


def part1():
    print("Part 1: Straight clamped-free rod")
    dEB = deltaEB(g=0.05)
    print(f"  Q1: delta_EB(g = 0.05) = {dEB * 1e3:.4f} mm")

    d, ch, _, Q = tipDeflection(g=0.05)
    print(f"  Q2: delta_DER(g = 0.05) = {d * 1e3:.4f} mm  (e = {abs(d - dEB) / dEB * 100:.2f}%, "
          f"last-second change {ch * 100:.4f}%)")
    print("  Q3: x2 is the clamp; the first edge (x1-x2) is fixed, so the free length is the last N-2 edges")

    print("  Q4: mesh convergence (g = 0.05); free edges double 20 -> 40 -> 80")
    prev = None
    for N in (22, 42, 82):
        diff = tipDeflection(N=N, g=0.05)[0] - dEB
        ratio = "" if prev is None else f", diff halves by {prev / diff:.2f}x"
        print(f"       N = {N:2d} ({N - 2:2d} free edges): delta = {(dEB + diff) * 1e3:.4f} mm, "
              f"delta - delta_EB = {diff * 1e3:+.4f} mm{ratio}")
        prev = diff

    print("  Q5: E sweep (g = 9.81)")
    Es = [10e6, 30e6, 100e6, 300e6, 1000e6]
    eE = []
    for E in Es:
        d, ch, _, _ = tipDeflection(E=E)
        dEB = deltaEB(E=E)
        eE.append(abs(d - dEB) / dEB)
        print(f"       E = {E / 1e6:6.0f} MPa: DER = {d:.5f} m, EB = {dEB:.5f} m, e = {eE[-1] * 100:6.2f}%"
              f"  (last-second change {ch * 100:.4f}%)")
    print(f"       smallest E with e < 20%: {next(E for E, e in zip(Es, eE) if e < 0.2) / 1e6:.0f} MPa")

    print("  Q6: g sweep (E = 10 MPa)")
    gs = [0.1, 0.3, 1.0, 3.0, 9.81]
    eg, dg, dEBg = [], [], []
    for g in gs:
        d, ch, t7, Q7 = tipDeflection(g=g)
        dEB = deltaEB(g=g)
        dg.append(d)
        dEBg.append(dEB)
        eg.append(abs(d - dEB) / dEB)
        print(f"       g = {g:5.2f}: DER = {d:.5f} m, EB = {dEB:.5f} m, e = {eg[-1] * 100:6.2f}%"
              f"  (last-second change {ch * 100:.4f}%)")
    print(f"       smallest g with e > 20%: {next(g for g, e in zip(gs, eg) if e > 0.2)} m/s^2")

    # Last run of the g sweep is the default case (g = 9.81, E = 10 MPa)
    theta = Q7[-1, 3::4]
    print(f"  Q7: delta_DER(default) = {dg[-1]:.4f} m  (EB = {dEBg[-1]:.4f} m)")
    print(f"       max |theta| (straight rod) = {np.abs(theta).max():.3e} rad")
    print("  Q8: Euler-Bernoulli is linear (small slopes, near-straight shape); DER is geometrically exact")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(gs, dg, "o-", label="DER")
    ax[0].loglog(gs, dEBg, "s--", label="Euler-Bernoulli")
    ax[0].set(xlabel="g [m/s$^2$]", ylabel="tip deflection [m]", title="E = 10 MPa")
    ax[0].legend()
    x = nodesOf(Q7[-1])
    ax[1].plot(x[:, 0], x[:, 2], "o-", ms=2)
    ax[1].set(xlabel="x [m]", ylabel="z [m]", title="Steady shape, default parameters", aspect="equal")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part1_straight_rod.png", dpi=150)
    plt.close(fig)
    return theta


def part2(theta_straight):
    print("Part 2: Naturally curved rod")
    t, Q = run(N=50, natR=NAT_R)
    zN = Q[:, -1]
    theta = Q[-1, 3::4]
    print(f"  Q9: z_N(steady) = {zN[-1]:.4f} m  (last-second change {settledChange(t, zN) * 100:.4f}%, "
          f"theta last-second max change {np.abs(Q[-1, 3::4] - Q[-11, 3::4]).max():.2e} rad)")
    print(f"  Q10: max |theta| straight = {np.abs(theta_straight).max():.3e} rad, "
          f"curved = {np.abs(theta).max():.3e} rad")

    t0, Q0 = run(N=50, natR=NAT_R, g=0.0, zeroKappaBar=True)
    dist = np.linalg.norm(Q0[:, -3:] - Q0[:, 0:3], axis=1)
    print(f"  Q11: |xN - x1| initial = {dist[0]:.4f} m, steady = {dist[-1]:.4f} m "
          f"(last-second change {settledChange(t0, dist) * 100:.4f}%)")

    fig = plt.figure(figsize=(12, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    for Qi, lbl in ((Q[0], "t = 0"), (Q[-1], "t = 10 s")):
        x = nodesOf(Qi)
        ax.plot(x[:, 0], x[:, 1], x[:, 2], "o-", ms=2, label=lbl)
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="Q9: curved rod under gravity")
    ax.legend()
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(theta_straight, label="straight rod (Q7)")
    ax.plot(theta, label="curved rod (Q9)")
    ax.set(xlabel="edge index", ylabel=r"$\theta^k$ [rad]", title="Q10: twist angles")
    ax.legend()
    ax = fig.add_subplot(1, 3, 3)
    for Qi, lbl in ((Q0[0], "t = 0"), (Q0[-1], "t = 10 s")):
        x = nodesOf(Qi)
        ax.plot(x[:, 0], x[:, 1], "o-", ms=2, label=lbl)
    ax.set(xlabel="x [m]", ylabel="y [m]", title=r"Q11: $\bar\kappa = 0$, g = 0", aspect="equal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "part2_curved_rod.png", dpi=150)
    plt.close(fig)


def main():
    run_tests()
    theta_straight = part1()
    part2(theta_straight)


if __name__ == "__main__":
    main()
