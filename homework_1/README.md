# Homework 1: Modular Multi-Mass-Spring-Damper Simulator

## How to run

From the repository root:

```bash
uv sync
uv run homework_1/Homework1.py
```

This first runs the 11 test cases from the handout, then simulates every Gradescope question and prints each result next to its analytical value.

## Files

- `Homework1.py`: the single main file. It contains:
  - Element functions: `gradEs`, `hessEs`, `dampingForce`, `dampingJacobian`
  - `getFJ`: assembles the residual and Jacobian by looping over the springs and dampers
  - `implicitEulerStep`: one implicit Euler step solved with Newton's method
  - `explicitEulerStep`: one forward Euler step
  - `simulate`: the time loop (Algorithm 2), which can use either step function
  - `run_tests` / `answer_questions`: the handout's test cases and the Part A–E questions
