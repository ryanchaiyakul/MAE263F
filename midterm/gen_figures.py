"""Generate publication-quality figures and exact verification metrics for midterm.tex."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import newton.examples
import newton.viewer
import numpy as np

EXAMPLES_DIR = Path("/home/rchaiyak/dev/dismech-newton/examples")
OUT_DIR = Path("/home/rchaiyak/dev/MAE263F/midterm")

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.0,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def load_example(name: str):
    path = EXAMPLES_DIR / f"{name}.py"
    for folder in (path.parent, EXAMPLES_DIR):
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(f"_fig_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_example(name: str, frames: int, **overrides):
    mod = load_example(name)
    base_args = newton.examples.create_parser().parse_args(["--viewer", "null", "--num-frames", str(frames)])
    args = SimpleNamespace(**{**vars(base_args), "test": True, **overrides})
    viewer = newton.viewer.ViewerNull(num_frames=frames)
    ex = mod.Example(viewer, args)
    post = getattr(ex, "test_post_step", None)
    for _ in range(frames):
        ex.step()
        if post is not None:
            post()
    ex.test_final()
    return mod, ex


def make_cantilever_figure():
    print("Running cantilever...")
    _, rod_ex = run_example("cantilever", 240)
    rod_sum = rod_ex.summary()
    print("Rod summary:", rod_sum)

    print("Running shell_cantilever...")
    _, shell_ex = run_example("shell_cantilever", 240)
    shell_sum = shell_ex.summary()
    print("Shell summary:", shell_sum)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=False)

    # Panel (a): Rod cantilever
    ax = axes[0]
    exact_rod = float(rod_ex.beam(np.array([rod_ex.length]))[0])
    freq_rod = rod_ex.beam_frequency()
    t_th = np.linspace(0.0, rod_ex.sim_time, 400)
    ax.plot(t_th, 1.0 - np.cos(2.0 * np.pi * freq_rod * t_th), "k--", lw=1.1, zorder=4, label="Euler--Bernoulli")
    short_labels = {
        "DER ADMM, implicit Euler": "DiSMech (Euler, 8 sub.)",
        "DER Gauss-Newton, implicit midpoint": "DiSMech (Midpoint, 8 sub.)",
        "VBD, 5 iterations": "VBD (5 iter., 10 sub.)",
        "VBD, 50 iterations": "VBD (50 iter., 100 sub.)",
    }
    for name, sim in rod_ex.sims.items():
        every = max(1, len(sim.t) // 1200)
        ax.plot(sim.t[::every], np.array(sim.sag[::every]) / exact_rod,
                color=rod_ex.colors[name], lw=1.2, label=short_labels.get(name, name))
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"Tip sag $w / w_{\mathrm{EB}}$")
    ax.set_title("(a) Rod Cantilever under Gravity")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 3.3)
    ax.legend(loc="upper right", frameon=True, framealpha=0.85)

    # Panel (b): Shell cantilever
    ax = axes[1]
    exact_sh = float(shell_ex.beam(np.array([shell_ex.length]))[0])
    freq_sh = shell_ex.beam_frequency()
    t_th_sh = np.linspace(0.0, shell_ex.sim_time, 400)
    ax.plot(t_th_sh, 1.0 - np.cos(2.0 * np.pi * freq_sh * t_th_sh), "k--", lw=1.1, zorder=4, label="Plate theory")
    short_labels_sh = {
        "Shell ADMM, implicit midpoint": "DiSMech (Midpoint, 8 sub.)",
        "Shell ADMM, implicit Euler": "DiSMech (Euler, 8 sub.)",
        "VBD, 5 iterations": "VBD (5 iter., 10 sub.)",
        "VBD, 100 iterations": "VBD (100 iter., 300 sub.)",
    }
    for name, sim in shell_ex.sims.items():
        every = max(1, len(sim.t) // 1200)
        ax.plot(sim.t[::every], np.array(sim.sag[::every]) / exact_sh,
                color=shell_ex.colors[name], lw=1.2, label=short_labels_sh.get(name, name))
    ax.set_xlabel("Time $t$ (s)")
    ax.set_ylabel(r"Tip sag $w / w_{\mathrm{plate}}$")
    ax.set_title("(b) Shell Cantilever under Gravity")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 3.0)
    ax.legend(loc="upper right", frameon=True, framealpha=0.85)

    fig.tight_layout(w_pad=1.8)
    out = OUT_DIR / "fig_cantilever.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def make_contact_theory_figure():
    print("Running overhand_knot...")
    knot_mod, knot_ex = run_example("overhand_knot", 1140)
    kc = knot_ex.comparison()
    k_ratio = float(np.median(kc["Fbar"] / kc["audoly"]))
    print(f"Knot: n={len(kc['Fbar'])}, median F/Audoly={k_ratio:.4f}, min_gap={knot_ex.min_gap:.4f} D")

    print("Running plectoneme...")
    _, plec_ex = run_example("plectoneme", 5040)
    pc = plec_ex.comparison()
    rows = np.array(plec_ex.rows)
    max_lk_err = float(np.max(np.abs(rows[:, -1] - rows[:, 1])))
    print(f"Plectoneme: n={len(pc['F'])}, max |Lk-turns|={max_lk_err:.4f}")
    for key in ("R", "F", "M"):
        v = pc[key]
        print(f"  {key}: median={np.median(v):.4f}, range={v.min():.4f}-{v.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))

    # Panel (a): Overhand knot
    ax = axes[0]
    mu = knot_ex.friction
    eps_max = max(0.8, 1.05 * float(kc["eps"].max()))
    eps_grid = np.linspace(0.0, eps_max, 200)
    ax.plot(eps_grid, knot_mod.audoly_force(eps_grid, mu), "k-", lw=1.2, label=rf"Audoly et al. ($\mu={mu}$)")
    ax.plot(eps_grid, knot_mod.audoly_force(eps_grid, 0.0), "0.55", ls="--", lw=1.0, label=r"Frictionless ($\mu=0$)")
    every = max(1, len(kc["eps"]) // 80)
    ax.plot(kc["eps"][::every], kc["Fbar"][::every], "o", ms=3.0, color="#2a9d8f", alpha=0.85, label="DiSMech Newton")
    ax.set_xlabel(r"Curvature parameter $\varepsilon = \sqrt{h / R}$")
    ax.set_ylabel(r"Normalized traction $F h^2 / B$")
    ax.set_title("(a) Overhand Knot Traction")
    ax.set_xlim(0.0, eps_max)
    ax.set_ylim(0.0, float(knot_mod.audoly_force(eps_max, mu)))
    ax.legend(loc="upper left", frameon=True, framealpha=0.85)

    # Panel (b): Plectoneme
    ax = axes[1]
    turns = pc["turns"]
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="Clauvelin et al. (2009)")
    ax.plot(turns, pc["F"], color="#2a9d8f", lw=1.2, label=r"Tension $F / F_{\mathrm{th}}$")
    ax.plot(turns, pc["M"], color="#6c5ce7", lw=1.2, label=r"Torque $M / M_{\mathrm{th}}$")
    ax.plot(turns, pc["R"], color="#e76f51", lw=1.2, label=r"Ply radius $R / a$")
    ax.set_xlabel("Applied turns $n$")
    ax.set_ylabel("Normalized quantity")
    ax.set_title("(b) Plectoneme Hard-Core Ply")
    ax.set_xlim(float(turns.min()), float(turns.max()))
    ax.set_ylim(0.85, 1.15)
    ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.85)

    fig.tight_layout(w_pad=1.8)
    out = OUT_DIR / "fig_theory_contact.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_cantilever_figure()
    make_contact_theory_figure()
