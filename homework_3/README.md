# Homework 3: Discrete Elastic Rods

## How to run

From the repository root:

```bash
uv sync
uv run homework_3/Homework3.py
```

The script runs its test cases, then prints the answer to each Gradescope
question (Q1–Q11) and saves the plots to `homework_3/`. A full run takes about 2 minutes.

## Files

- `Homework3.py`: the main file. It holds a 3D Discrete Elastic Rods simulator
  (stretching, bending and twisting energies with analytic gradients and Hessians,
  a time-parallel reference frame, and implicit Euler with Newton's method). It also has:
  - Part 1: a straight clamped–free rod under gravity, compared with Euler–Bernoulli
    beam theory (Q1–Q8: small load, mesh convergence, and E and g sweeps).
  - Part 2: a naturally curved rod under gravity (Q9), twist angles (Q10), and the
    rod with its natural curvature removed and g = 0 (Q11).
  - Test cases: finite-difference checks of the gradients and Hessians, plus
    geometry, mass and beam-theory checks.
- `part1_straight_rod.png`: tip deflection vs. g (DER and Euler–Bernoulli), and the
  steady shape of the default rod.
- `part2_curved_rod.png`: the curved rod before and after gravity, the twist angles of
  the straight and curved rods, and the rod opening up when its natural curvature is zero.
- `HW3_2026.pdf`: the assignment.
