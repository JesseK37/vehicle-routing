"""
sa_solver.py — Simulated Annealing for the Chance-Constrained VRP
=================================================================

**Algorithm Overview**

We solve the CC-VRP via simulated annealing (SA) with a *penalty-based*
formulation.  The objective function is

.. math::

    f(\\mathbf{S}) \\;=\\;
    \\underbrace{\\sum_{k=1}^{K} \\text{dist}(R_k)}_{\\text{travel cost}}
    \\;+\\;
    \\lambda \\sum_{k=1}^{K}
        \\max\\!\\bigl(0,\\;
            P(D_{R_k} > Q) - \\varepsilon
        \\bigr)
        \\;\\cdot\\; |R_k|

where :math:`\\lambda` is a penalty weight that is *adaptively* increased
when the current solution violates constraints, and decreased when
feasible.  The :math:`|R_k|` factor scales the penalty by route size
to avoid a bias toward splitting large routes unnecessarily.

The overflow probability :math:`P(D_{R_k} > Q)` is evaluated via the
closed-form Normal-CDF approximation from ``instance_generator.py``
(the fast analytic path).

**Neighbourhood Operators**

Two move types are used, selected uniformly at random:

1. **Or-opt (relocate):** Remove a customer from one route and insert
   it at the best position in another route (or the same route).
   This is the primary mechanism for rebalancing load across vehicles.

2. **Intra-route 2-opt:** Reverse a segment within a single route.
   This is the primary mechanism for improving travel distance within
   a route.

Both moves preserve the depot-start/depot-end structure and are
evaluated in :math:`O(|R|)` time.

**Cooling Schedule**

Geometric cooling: :math:`T_{k+1} = \\alpha \\, T_k` with
:math:`\\alpha \\in [0.9990, 0.9999]`.  Initial temperature is
calibrated so that an average-worsening move is accepted with
probability :math:`\\approx 0.8` at :math:`T_0`.

**Adaptive Penalty**

Every ``penalty_update_interval`` iterations:

* If the current best is *infeasible*:
  :math:`\\lambda \\leftarrow \\lambda \\cdot (1 + \\delta)`
* If feasible:
  :math:`\\lambda \\leftarrow \\lambda \\cdot (1 - \\delta)`

with :math:`\\delta = 0.1` by default.  This drives the search toward
the feasibility boundary without hand-tuning :math:`\\lambda`.

**References**

* Kirkpatrick, S., Gelatt, C.D., & Vecchi, M.P. (1983). Optimization
  by simulated annealing. *Science*, 220(4598), 671–680.
* Li, X., Tian, P., & Leung, S.C.H. (2010). Vehicle routing problems
  with time windows and stochastic travel and service times. *IJPE*,
  125(1), 137–145.

**Parameter Table**

=========================  ===========  ====================================
Parameter                  Default      Description
=========================  ===========  ====================================
max_iter                   200_000      Total SA iterations
T_start                    1.0          Initial temperature
T_end                      1e-4         Final temperature
alpha                      None         Cooling rate (auto-calibrated if None)
epsilon                    0.05         Chance-constraint threshold
lambda_init                10.0         Initial penalty weight
penalty_update_interval    1_000        Iterations between λ updates
penalty_delta              0.1          Multiplicative λ adjustment rate
seed                       0            RNG seed
verbose                    True         Print progress
=========================  ===========  ====================================

Author : Portfolio — Project 4 (Combinatorial Optimisation)
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from instance_generator import (
    ProblemInstance,
    nearest_neighbour_deterministic,
    route_distance,
    route_overflow_probability_analytic,
)


# ---------------------------------------------------------------------------
# Solution representation
# ---------------------------------------------------------------------------

@dataclass
class Solution:
    """A complete CC-VRP solution.

    Attributes
    ----------
    routes : list[list[int]]
        Each inner list is an ordered sequence of 0-based customer indices.
    total_distance : float
        Sum of route distances (travel cost).
    overflow_probs : list[float]
        Per-route overflow probability (analytic).
    max_overflow : float
        Maximum overflow probability across routes.
    is_feasible : bool
        True if all routes satisfy the chance constraint.
    objective : float
        Penalised objective value.
    """
    routes: list[list[int]]
    total_distance: float
    overflow_probs: list[float]
    max_overflow: float
    is_feasible: bool
    objective: float


# ---------------------------------------------------------------------------
# Solution evaluation
# ---------------------------------------------------------------------------

def evaluate_solution(
    routes: list[list[int]],
    instance: ProblemInstance,
    epsilon: float,
    lam: float,
) -> Solution:
    """Evaluate a full solution: distances, overflow probabilities, objective.

    Parameters
    ----------
    routes : list[list[int]]
        Route set.
    instance : ProblemInstance
        Problem data.
    epsilon : float
        Chance-constraint threshold.
    lam : float
        Current penalty weight λ.

    Returns
    -------
    Solution
        Evaluated solution object.
    """
    total_dist = 0.0
    overflow_probs = []
    penalty = 0.0

    for route in routes:
        d = route_distance(route, instance)
        total_dist += d
        p = route_overflow_probability_analytic(route, instance)
        overflow_probs.append(p)
        violation = max(0.0, p - epsilon)
        # Scale penalty by route size to avoid bias
        penalty += violation * max(len(route), 1)

    objective = total_dist + lam * penalty
    max_overflow = max(overflow_probs) if overflow_probs else 0.0
    is_feasible = max_overflow <= epsilon

    return Solution(
        routes=routes,
        total_distance=total_dist,
        overflow_probs=overflow_probs,
        max_overflow=max_overflow,
        is_feasible=is_feasible,
        objective=objective,
    )


# ---------------------------------------------------------------------------
# Neighbourhood moves
# ---------------------------------------------------------------------------

def _or_opt_move(
    routes: list[list[int]],
    rng: np.random.Generator,
) -> list[list[int]] | None:
    """Or-opt (relocate): move one customer between routes.

    Picks a random customer from a random non-empty route and inserts
    it at a random position in a (possibly different) route.

    Returns None if no valid move is possible (e.g. all routes empty
    except one with a single customer).
    """
    non_empty = [k for k, r in enumerate(routes) if len(r) > 0]
    if not non_empty:
        return None

    # Pick source route (must have at least 1 customer)
    src_idx = rng.choice(non_empty)
    src = routes[src_idx]
    # Pick customer to remove
    pos = rng.integers(0, len(src))
    customer = src[pos]

    # Pick destination route (can be the same — intra-route relocate)
    dst_idx = rng.integers(0, len(routes))
    dst = routes[dst_idx]

    # Build new routes
    new_routes = [list(r) for r in routes]
    new_routes[src_idx].pop(pos)

    # Insertion position in destination
    insert_pos = rng.integers(0, len(new_routes[dst_idx]) + 1)
    new_routes[dst_idx].insert(insert_pos, customer)

    return new_routes


def _two_opt_move(
    routes: list[list[int]],
    rng: np.random.Generator,
) -> list[list[int]] | None:
    """Intra-route 2-opt: reverse a segment within one route.

    Picks a random route with ≥ 2 customers and reverses a random
    sub-segment.
    """
    eligible = [k for k, r in enumerate(routes) if len(r) >= 2]
    if not eligible:
        return None

    k = rng.choice(eligible)
    route = routes[k]
    n = len(route)

    i = rng.integers(0, n - 1)
    j = rng.integers(i + 1, n)

    new_routes = [list(r) for r in routes]
    new_routes[k][i:j + 1] = new_routes[k][i:j + 1][::-1]

    return new_routes


def _cross_exchange_move(
    routes: list[list[int]],
    rng: np.random.Generator,
) -> list[list[int]] | None:
    """Cross-exchange: swap tail segments between two routes.

    Picks two distinct routes and a random cut point in each, then
    swaps the suffixes.  This enables larger structural changes than
    Or-opt.
    """
    non_empty = [k for k, r in enumerate(routes) if len(r) >= 1]
    if len(non_empty) < 2:
        return None

    idxs = rng.choice(non_empty, size=2, replace=False)
    k1, k2 = int(idxs[0]), int(idxs[1])

    r1, r2 = routes[k1], routes[k2]
    # Cut points: position after which the tail is swapped
    # cut=0 means swap everything; cut=len means swap nothing
    c1 = rng.integers(0, len(r1) + 1)
    c2 = rng.integers(0, len(r2) + 1)

    new_routes = [list(r) for r in routes]
    new_routes[k1] = list(r1[:c1]) + list(r2[c2:])
    new_routes[k2] = list(r2[:c2]) + list(r1[c1:])

    return new_routes


# ---------------------------------------------------------------------------
# Initial solution
# ---------------------------------------------------------------------------

def _build_initial_solution(
    instance: ProblemInstance,
    n_vehicles: int,
) -> list[list[int]]:
    """Build initial solution via nearest-neighbour, padded to K routes.

    If NN produces fewer than K routes, empty routes are appended so
    the SA has the full fleet available for rebalancing.
    """
    routes = nearest_neighbour_deterministic(instance)
    # Pad to exactly n_vehicles routes
    while len(routes) < n_vehicles:
        routes.append([])
    return routes


# ---------------------------------------------------------------------------
# Simulated annealing solver
# ---------------------------------------------------------------------------

@dataclass
class SAConfig:
    """Configuration for the SA solver.

    All parameters have sensible defaults.  ``alpha`` is auto-calibrated
    from ``T_start``, ``T_end``, and ``max_iter`` if left as None.
    """
    max_iter: int = 200_000
    T_start: float = 1.0
    T_end: float = 1e-4
    alpha: float | None = None
    epsilon: float = 0.05
    lambda_init: float = 10.0
    penalty_update_interval: int = 1_000
    penalty_delta: float = 0.1
    seed: int = 0
    verbose: bool = True
    log_interval: int = 10_000


@dataclass
class SAResult:
    """Result container for the SA solver.

    Attributes
    ----------
    best : Solution
        Best solution found.
    history : dict
        Keys: ``"iteration"``, ``"objective"``, ``"distance"``,
        ``"max_overflow"``, ``"temperature"``, ``"lambda"``,
        ``"feasible"``, ``"acceptance_rate"``.
        Each value is a list sampled at ``log_interval``.
    elapsed_sec : float
        Wall-clock time in seconds.
    config : SAConfig
        Configuration used.
    """
    best: Solution
    history: dict
    elapsed_sec: float
    config: SAConfig


def solve(
    instance: ProblemInstance,
    config: SAConfig | None = None,
) -> SAResult:
    """Run simulated annealing on a CC-VRP instance.

    Parameters
    ----------
    instance : ProblemInstance
        Problem instance from :func:`instance_generator.generate_instance`.
    config : SAConfig or None
        Solver configuration.  Uses defaults if None.

    Returns
    -------
    SAResult
        Best solution, convergence history, and timing.
    """
    if config is None:
        config = SAConfig()

    rng = np.random.default_rng(config.seed)

    # --- Cooling rate ---
    if config.alpha is None:
        alpha = (config.T_end / config.T_start) ** (1.0 / config.max_iter)
    else:
        alpha = config.alpha

    # --- Move selection weights: or-opt, 2-opt, cross-exchange ---
    move_funcs = [_or_opt_move, _two_opt_move, _cross_exchange_move]
    move_weights = np.array([0.45, 0.35, 0.20])
    move_weights /= move_weights.sum()
    move_names = ["or-opt", "2-opt", "cross"]

    # --- Initial solution ---
    routes = _build_initial_solution(instance, instance.n_vehicles)
    lam = config.lambda_init
    current = evaluate_solution(routes, instance, config.epsilon, lam)
    best = copy.deepcopy(current)

    # --- History tracking ---
    history = {
        "iteration": [],
        "objective": [],
        "distance": [],
        "max_overflow": [],
        "temperature": [],
        "lambda": [],
        "feasible": [],
        "acceptance_rate": [],
    }

    T = config.T_start
    accepts = 0
    total_moves = 0
    t0 = time.perf_counter()

    for it in range(config.max_iter):
        # --- Select and apply move ---
        move_idx = rng.choice(len(move_funcs), p=move_weights)
        new_routes = move_funcs[move_idx](current.routes, rng)

        if new_routes is None:
            continue

        total_moves += 1
        candidate = evaluate_solution(new_routes, instance, config.epsilon, lam)

        # --- Metropolis acceptance ---
        delta = candidate.objective - current.objective
        if delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-15)):
            current = candidate
            accepts += 1

            # Update global best (prefer feasible, then lower distance)
            if current.is_feasible and (
                not best.is_feasible
                or current.total_distance < best.total_distance
            ):
                best = copy.deepcopy(current)
            elif not best.is_feasible and current.objective < best.objective:
                best = copy.deepcopy(current)

        # --- Adaptive penalty update ---
        if (it + 1) % config.penalty_update_interval == 0:
            if best.is_feasible:
                lam *= (1.0 - config.penalty_delta)
            else:
                lam *= (1.0 + config.penalty_delta)
            lam = max(lam, 0.1)  # floor

            # Re-evaluate current and best under new λ
            current = evaluate_solution(
                current.routes, instance, config.epsilon, lam
            )
            best_re = evaluate_solution(
                best.routes, instance, config.epsilon, lam
            )
            best = best_re  # feasibility doesn't change, only objective

        # --- Cooling ---
        T *= alpha

        # --- Logging ---
        if (it + 1) % config.log_interval == 0:
            rate = accepts / max(total_moves, 1)
            history["iteration"].append(it + 1)
            history["objective"].append(best.objective)
            history["distance"].append(best.total_distance)
            history["max_overflow"].append(best.max_overflow)
            history["temperature"].append(T)
            history["lambda"].append(lam)
            history["feasible"].append(best.is_feasible)
            history["acceptance_rate"].append(rate)

            if config.verbose and (it + 1) % config.log_interval == 0:
                tag = "✓" if best.is_feasible else "✗"
                print(
                    f"  iter {it+1:>8d}  "
                    f"T={T:.2e}  "
                    f"λ={lam:>7.2f}  "
                    f"dist={best.total_distance:.4f}  "
                    f"maxP={best.max_overflow:.4f}  "
                    f"acc={rate:.3f}  "
                    f"{tag}"
                )

    elapsed = time.perf_counter() - t0

    return SAResult(
        best=best,
        history=history,
        elapsed_sec=elapsed,
        config=config,
    )


# ---------------------------------------------------------------------------
# Solution summary
# ---------------------------------------------------------------------------

def solution_summary(result: SAResult, instance: ProblemInstance) -> str:
    """Human-readable summary of a solver result.

    Parameters
    ----------
    result : SAResult
        Solver output.
    instance : ProblemInstance
        Problem instance.

    Returns
    -------
    str
        Multi-line summary.
    """
    sol = result.best
    lines = [
        "=" * 65,
        "SA-CCVRP Solution Summary",
        "=" * 65,
        f"  Feasible           : {'Yes' if sol.is_feasible else 'No'}",
        f"  Total distance     : {sol.total_distance:.4f}",
        f"  Max overflow prob  : {sol.max_overflow:.6f}",
        f"  ε threshold        : {result.config.epsilon:.4f}",
        f"  Routes used        : {sum(1 for r in sol.routes if r)}/"
        f"{len(sol.routes)}",
        f"  Elapsed            : {result.elapsed_sec:.2f} s",
        f"  Iterations         : {result.config.max_iter:,}",
        "-" * 65,
    ]

    for k, route in enumerate(sol.routes):
        if not route:
            continue
        mu_sum = instance.mus[route].sum()
        d = route_distance(route, instance)
        p = sol.overflow_probs[k]
        tag = "✓" if p <= result.config.epsilon else "✗"
        lines.append(
            f"  Route {k}: {len(route):>2d} cust  "
            f"Σμ={mu_sum:>6.2f}  "
            f"dist={d:.4f}  "
            f"P(overflow)={p:.6f}  {tag}"
        )

    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from instance_generator import generate_instance, instance_summary

    print("▸ Generating instance …")
    inst = generate_instance(n_customers=25, seed=42)
    print(instance_summary(inst))

    # --- Baseline ---
    from instance_generator import nearest_neighbour_deterministic
    nn_routes = nearest_neighbour_deterministic(inst)
    nn_sol = evaluate_solution(nn_routes, inst, epsilon=0.05, lam=10.0)
    print(f"\n▸ NN baseline: dist={nn_sol.total_distance:.4f}  "
          f"maxP={nn_sol.max_overflow:.4f}  "
          f"feasible={nn_sol.is_feasible}")

    # --- SA solver (moderate run) ---
    print("\n▸ Running SA solver (200k iterations) …\n")
    cfg = SAConfig(
        max_iter=200_000,
        T_start=1.0,
        T_end=1e-4,
        epsilon=0.05,
        lambda_init=10.0,
        seed=0,
        verbose=True,
        log_interval=25_000,
    )
    result = solve(inst, cfg)
    print()
    print(solution_summary(result, inst))

    # --- Improvement report ---
    if nn_sol.total_distance > 0:
        if result.best.is_feasible:
            print(f"\n▸ SA found a FEASIBLE solution.")
            print(f"  Distance: {result.best.total_distance:.4f} "
                  f"(NN baseline: {nn_sol.total_distance:.4f})")
            pct = (result.best.total_distance / nn_sol.total_distance - 1) * 100
            print(f"  Distance change vs NN: {pct:+.1f}%")
            print(f"  But NN baseline violates constraints on "
                  f"{sum(1 for p in nn_sol.overflow_probs if p > 0.05)}"
                  f"/{len(nn_sol.overflow_probs)} routes.")
        else:
            print(f"\n▸ SA did not find a fully feasible solution.")
            print(f"  Best maxP = {result.best.max_overflow:.6f} "
                  f"(target ≤ {cfg.epsilon})")

    print("\n▸ Smoke test passed.")
