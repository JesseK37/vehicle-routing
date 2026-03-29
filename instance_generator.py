"""
instance_generator.py — Chance-Constrained VRP Instance Generator
=================================================================

**Model Specification**

We consider a single-depot vehicle routing problem on the unit square
:math:`[0,1]^2`.  A depot at coordinates :math:`(x_0, y_0)` dispatches
:math:`K` homogeneous vehicles, each with deterministic capacity :math:`Q`,
to serve :math:`N` customers.

Each customer :math:`i` has a *stochastic* demand

.. math::

    d_i \\sim \\text{LogNormal}(\\mu_i^{\\ln}, \\sigma_i^{\\ln\\,2})

parameterised so that the *marginal* mean and coefficient of variation
(CV) match user-specified targets:

.. math::

    \\mathbb{E}[d_i] = \\mu_i, \\qquad
    \\text{CV}_i = \\sigma_i / \\mu_i \\in [0.3,\\, 0.5].

The log-normal is chosen because demands are non-negative and right-skewed
— a better physical model than the Gaussian for unit quantities.

**Chance Constraint (Normal Approximation)**

For a candidate route :math:`R = \\{i_1, \\dots, i_m\\}`, total route demand
is :math:`D_R = \\sum_{j \\in R} d_j`.  Under independence the CLT gives
the Gaussian approximation

.. math::

    D_R \\;\\dot{\\sim}\\; \\mathcal{N}\\!\\left(
        \\sum_{j \\in R} \\mu_j,\\;
        \\sum_{j \\in R} \\sigma_j^2
    \\right),

and the chance constraint :math:`\\mathbb{P}(D_R > Q) \\le \\varepsilon`
becomes

.. math::

    \\sum_{j \\in R} \\mu_j
    + \\Phi^{-1}(1-\\varepsilon)\\,
      \\sqrt{\\sum_{j \\in R} \\sigma_j^2}
    \\;\\le\\; Q,

where :math:`\\Phi^{-1}` is the standard-normal quantile function.
This is the **deterministic equivalent** of the probabilistic constraint
and can be evaluated in :math:`O(|R|)` time.

**Two-Path Design**

* *Fast path* — ``route_overflow_probability_analytic``: closed-form
  Normal-CDF evaluation.  Used inside the solver's inner loop.
* *Thorough path* — ``route_overflow_probability_mc``: brute-force
  Monte Carlo with configurable sample count.  Used for final solution
  audit and to validate the CLT approximation.

**References**

* Gendreau, M., Laporte, G., & Séguin, R. (1996). Stochastic vehicle
  routing. *European Journal of Operational Research*, 88(1), 3–12.
* Dinh, T., Fukasawa, R., & Luedtke, J. (2018). Exact algorithms for
  the chance-constrained vehicle routing problem. *Math. Programming*,
  172, 105–138.

**Parameter Table**

===============  ========  ==========================================
Parameter        Default   Description
===============  ========  ==========================================
n_customers      25        Number of customers
n_vehicles       5         Fleet size :math:`K`
capacity         40.0      Vehicle capacity :math:`Q`
cv_range         (0.3,0.5) Demand CV drawn uniformly from this range
mean_demand_rng  (3.0,8.0) Mean demand drawn uniformly
depot            (0.5,0.5) Depot coordinates
seed             42        NumPy RNG seed
===============  ========  ==========================================

Author : Portfolio — Project 4 (Combinatorial Optimisation)
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Customer:
    """Single customer with location and demand distribution parameters.

    Attributes
    ----------
    x, y : float
        Coordinates on :math:`[0,1]^2`.
    mu : float
        Mean demand  :math:`\\mu_i`.
    sigma : float
        Demand standard deviation :math:`\\sigma_i`.
    cv : float
        Coefficient of variation  :math:`\\sigma_i / \\mu_i`.
    ln_mu : float
        Log-normal location parameter.
    ln_sigma : float
        Log-normal scale parameter.
    """
    x: float
    y: float
    mu: float
    sigma: float
    cv: float
    ln_mu: float
    ln_sigma: float


@dataclass
class ProblemInstance:
    """Complete CC-VRP instance.

    Encapsulates all information needed by the solver: customer data,
    fleet parameters, and the pre-computed distance matrix.

    Attributes
    ----------
    customers : list[Customer]
        Length-N list of customers (0-indexed).
    depot : tuple[float, float]
        Depot coordinates.
    n_vehicles : int
        Fleet size K.
    capacity : float
        Vehicle capacity Q.
    dist_matrix : NDArray[np.float64]
        Shape ``(N+1, N+1)`` Euclidean distance matrix.
        Index 0 is the depot; indices 1..N are customers.
    mus : NDArray[np.float64]
        Shape ``(N,)`` vector of mean demands.
    sigmas : NDArray[np.float64]
        Shape ``(N,)`` vector of demand standard deviations.
    seed : int
        RNG seed used to generate the instance.
    """
    customers: list[Customer]
    depot: Tuple[float, float]
    n_vehicles: int
    capacity: float
    dist_matrix: NDArray[np.float64]
    mus: NDArray[np.float64]
    sigmas: NDArray[np.float64]
    seed: int


# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------

def _lognormal_params(mu: float, sigma: float) -> Tuple[float, float]:
    """Convert marginal (mu, sigma) to log-normal (ln_mu, ln_sigma).

    Given :math:`\\mathbb{E}[X] = \\mu` and :math:`\\text{Var}(X) = \\sigma^2`
    for :math:`X \\sim \\text{LogNormal}(\\mu_{\\ln}, \\sigma_{\\ln}^2)`,
    the relations are

    .. math::

        \\sigma_{\\ln}^2 = \\ln\\!\\left(1 + \\frac{\\sigma^2}{\\mu^2}\\right),
        \\qquad
        \\mu_{\\ln} = \\ln(\\mu) - \\tfrac{1}{2}\\sigma_{\\ln}^2.

    Parameters
    ----------
    mu : float
        Desired marginal mean.
    sigma : float
        Desired marginal standard deviation.

    Returns
    -------
    ln_mu, ln_sigma : float
        Log-normal location and scale parameters.
    """
    ln_sigma2 = np.log(1.0 + (sigma / mu) ** 2)
    ln_sigma = np.sqrt(ln_sigma2)
    ln_mu = np.log(mu) - 0.5 * ln_sigma2
    return float(ln_mu), float(ln_sigma)


def generate_instance(
    n_customers: int = 25,
    n_vehicles: int = 5,
    capacity: float = 40.0,
    cv_range: Tuple[float, float] = (0.3, 0.5),
    mean_demand_range: Tuple[float, float] = (3.0, 8.0),
    depot: Tuple[float, float] = (0.5, 0.5),
    seed: int = 42,
) -> ProblemInstance:
    """Generate a CC-VRP instance on the unit square.

    Customer locations are drawn uniformly on :math:`[0,1]^2`.
    Mean demands are drawn from ``mean_demand_range`` and CVs from
    ``cv_range``, both uniformly.  The log-normal parameters are then
    computed analytically.

    The distance matrix is Euclidean on the (N+1)-node graph
    (depot = node 0).

    Parameters
    ----------
    n_customers : int
        Number of customers N.
    n_vehicles : int
        Fleet size K.
    capacity : float
        Vehicle capacity Q.
    cv_range : tuple[float, float]
        (min_cv, max_cv) for per-customer CV.
    mean_demand_range : tuple[float, float]
        (min_mu, max_mu) for per-customer mean demand.
    depot : tuple[float, float]
        Depot coordinates.
    seed : int
        NumPy RNG seed for reproducibility.

    Returns
    -------
    ProblemInstance
        Fully-specified problem instance.
    """
    rng = np.random.default_rng(seed)

    # --- Customer locations ---
    xs = rng.uniform(0.0, 1.0, size=n_customers)
    ys = rng.uniform(0.0, 1.0, size=n_customers)

    # --- Demand distributions ---
    mus = rng.uniform(*mean_demand_range, size=n_customers)
    cvs = rng.uniform(*cv_range, size=n_customers)
    sigmas = mus * cvs

    customers: list[Customer] = []
    for i in range(n_customers):
        ln_mu, ln_sigma = _lognormal_params(mus[i], sigmas[i])
        customers.append(Customer(
            x=float(xs[i]),
            y=float(ys[i]),
            mu=float(mus[i]),
            sigma=float(sigmas[i]),
            cv=float(cvs[i]),
            ln_mu=ln_mu,
            ln_sigma=ln_sigma,
        ))

    # --- Distance matrix (depot = index 0) ---
    coords = np.zeros((n_customers + 1, 2))
    coords[0] = depot
    coords[1:, 0] = xs
    coords[1:, 1] = ys

    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=-1))

    return ProblemInstance(
        customers=customers,
        depot=depot,
        n_vehicles=n_vehicles,
        capacity=capacity,
        dist_matrix=dist_matrix,
        mus=mus,
        sigmas=sigmas,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Chance-constraint evaluation  (two-path design)
# ---------------------------------------------------------------------------

def route_overflow_probability_analytic(
    route: list[int],
    instance: ProblemInstance,
) -> float:
    """Closed-form overflow probability under the CLT approximation.

    For route :math:`R`, compute

    .. math::

        \\mathbb{P}(D_R > Q)
        = 1 - \\Phi\\!\\left(
            \\frac{Q - \\sum_{j \\in R} \\mu_j}
                 {\\sqrt{\\sum_{j \\in R} \\sigma_j^2}}
        \\right)

    using :func:`scipy.stats.norm.sf` for numerical stability in the
    upper tail.

    **Complexity:** :math:`O(|R|)`.

    Parameters
    ----------
    route : list[int]
        Customer indices (0-based into the customers list, *not* the
        distance-matrix indices).
    instance : ProblemInstance
        Problem instance.

    Returns
    -------
    float
        Probability that total route demand exceeds capacity.
    """
    if len(route) == 0:
        return 0.0

    idx = np.array(route)
    mu_total = instance.mus[idx].sum()
    var_total = (instance.sigmas[idx] ** 2).sum()

    if var_total < 1e-15:
        return 0.0 if mu_total <= instance.capacity else 1.0

    # P(D > Q) = 1 - Phi((Q - mu) / sigma)  = sf(z)
    z = (instance.capacity - mu_total) / np.sqrt(var_total)
    return float(stats.norm.sf(z))


def route_overflow_probability_mc(
    route: list[int],
    instance: ProblemInstance,
    n_samples: int = 50_000,
    seed: int | None = None,
) -> float:
    """Monte Carlo overflow probability using the true log-normal demands.

    Draws ``n_samples`` realisations of each customer's demand from
    its log-normal distribution, sums them per route, and reports the
    fraction exceeding capacity.

    This is the *validation* path: it uses the true demand model (not
    the Normal approximation) and can therefore detect CLT approximation
    error for small routes or high-CV customers.

    Parameters
    ----------
    route : list[int]
        Customer indices (0-based).
    instance : ProblemInstance
        Problem instance.
    n_samples : int
        Number of Monte Carlo samples.
    seed : int or None
        RNG seed; ``None`` for non-deterministic.

    Returns
    -------
    float
        Estimated overflow probability.
    """
    if len(route) == 0:
        return 0.0

    rng = np.random.default_rng(seed)

    # Draw from log-normal for each customer
    total_demand = np.zeros(n_samples)
    for i in route:
        c = instance.customers[i]
        samples = rng.lognormal(mean=c.ln_mu, sigma=c.ln_sigma, size=n_samples)
        total_demand += samples

    return float(np.mean(total_demand > instance.capacity))


def check_chance_constraint(
    route: list[int],
    instance: ProblemInstance,
    epsilon: float = 0.05,
    method: str = "analytic",
    **mc_kwargs,
) -> Tuple[bool, float]:
    """Check whether a route satisfies the chance constraint.

    .. math::

        \\mathbb{P}(D_R > Q) \\le \\varepsilon

    Parameters
    ----------
    route : list[int]
        Customer indices (0-based).
    instance : ProblemInstance
        Problem instance.
    epsilon : float
        Maximum allowable overflow probability.
    method : {"analytic", "mc"}
        Which evaluation path to use.
    **mc_kwargs
        Forwarded to :func:`route_overflow_probability_mc` when
        ``method="mc"``.

    Returns
    -------
    feasible : bool
        True if :math:`P(\\text{overflow}) \\le \\varepsilon`.
    p_overflow : float
        Computed overflow probability.
    """
    if method == "analytic":
        p = route_overflow_probability_analytic(route, instance)
    elif method == "mc":
        p = route_overflow_probability_mc(route, instance, **mc_kwargs)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    return p <= epsilon, p


# ---------------------------------------------------------------------------
# Route cost helper
# ---------------------------------------------------------------------------

def route_distance(route: list[int], instance: ProblemInstance) -> float:
    """Total Euclidean travel distance for a closed route from the depot.

    Route: depot → route[0] → route[1] → ... → route[-1] → depot.
    Customer indices are 0-based; internally shifted to distance-matrix
    indices (depot = 0, customer i → matrix index i+1).

    Parameters
    ----------
    route : list[int]
        Ordered customer indices.
    instance : ProblemInstance
        Problem instance (provides the distance matrix).

    Returns
    -------
    float
        Total travel distance.
    """
    if len(route) == 0:
        return 0.0

    dm = instance.dist_matrix
    # Shift to distance-matrix indices: customer i → i+1
    nodes = [0] + [i + 1 for i in route] + [0]
    return float(sum(dm[nodes[k], nodes[k + 1]] for k in range(len(nodes) - 1)))


# ---------------------------------------------------------------------------
# Deterministic nearest-neighbour baseline
# ---------------------------------------------------------------------------

def nearest_neighbour_deterministic(
    instance: ProblemInstance,
    use_mean_demand: bool = True,
) -> list[list[int]]:
    """Nearest-neighbour heuristic using mean demands (deterministic baseline).

    Greedily assigns each unvisited customer to the nearest feasible
    position, opening a new route when the current vehicle would exceed
    capacity (based on mean demands).  Ignores demand variance entirely.

    Parameters
    ----------
    instance : ProblemInstance
        Problem instance.
    use_mean_demand : bool
        If True, use :math:`\\mu_i` as the deterministic demand.

    Returns
    -------
    list[list[int]]
        List of routes (each route is a list of 0-based customer indices).
    """
    n = len(instance.customers)
    demands = instance.mus if use_mean_demand else instance.mus  # extensible
    dm = instance.dist_matrix

    visited = np.zeros(n, dtype=bool)
    routes: list[list[int]] = []
    vehicles_used = 0

    while not visited.all():
        if vehicles_used >= instance.n_vehicles:
            # Force remaining customers into the last route (infeasible flag)
            remaining = list(np.where(~visited)[0])
            routes.append(remaining)
            visited[remaining] = True
            break

        route: list[int] = []
        load = 0.0
        current_node = 0  # depot in distance-matrix indexing

        while True:
            # Find nearest unvisited customer that fits
            best_dist = np.inf
            best_cust = -1
            for j in range(n):
                if visited[j]:
                    continue
                if load + demands[j] > instance.capacity:
                    continue
                d = dm[current_node, j + 1]
                if d < best_dist:
                    best_dist = d
                    best_cust = j

            if best_cust == -1:
                break  # no feasible insertion → close route

            route.append(best_cust)
            visited[best_cust] = True
            load += demands[best_cust]
            current_node = best_cust + 1  # distance-matrix index

        if route:
            routes.append(route)
            vehicles_used += 1

    return routes


# ---------------------------------------------------------------------------
# Instance summary
# ---------------------------------------------------------------------------

def instance_summary(instance: ProblemInstance) -> str:
    """Human-readable summary of a problem instance.

    Returns
    -------
    str
        Multi-line summary string.
    """
    n = len(instance.customers)
    total_mean = instance.mus.sum()
    total_cap = instance.n_vehicles * instance.capacity
    utilisation = total_mean / total_cap

    lines = [
        "=" * 60,
        "CC-VRP Instance Summary",
        "=" * 60,
        f"  Customers          : {n}",
        f"  Vehicles (K)       : {instance.n_vehicles}",
        f"  Capacity (Q)       : {instance.capacity:.1f}",
        f"  Depot              : ({instance.depot[0]:.2f}, {instance.depot[1]:.2f})",
        f"  Seed               : {instance.seed}",
        "-" * 60,
        f"  Mean demand range  : [{instance.mus.min():.2f}, {instance.mus.max():.2f}]",
        f"  CV range           : [{min(c.cv for c in instance.customers):.3f}, "
        f"{max(c.cv for c in instance.customers):.3f}]",
        f"  Total mean demand  : {total_mean:.1f}",
        f"  Total fleet cap    : {total_cap:.1f}",
        f"  Utilisation (mean) : {utilisation:.1%}",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("▸ Generating default instance …")
    inst = generate_instance()
    print(instance_summary(inst))

    # --- Nearest-neighbour baseline ---
    nn_routes = nearest_neighbour_deterministic(inst)
    total_dist = sum(route_distance(r, inst) for r in nn_routes)
    print(f"\n▸ Nearest-neighbour baseline: {len(nn_routes)} routes, "
          f"total distance = {total_dist:.4f}")

    # --- Chance-constraint audit (both paths) ---
    print(f"\n▸ Route-level chance-constraint audit (ε = 0.05):")
    print(f"  {'Route':>5s}  {'|R|':>4s}  {'Σμ':>7s}  {'P_analytic':>11s}"
          f"  {'P_mc':>11s}  {'Feasible':>8s}")
    print("  " + "-" * 55)
    for k, route in enumerate(nn_routes):
        ok_a, p_a = check_chance_constraint(route, inst, epsilon=0.05,
                                            method="analytic")
        _, p_mc = check_chance_constraint(route, inst, epsilon=0.05,
                                          method="mc", n_samples=100_000,
                                          seed=123)
        mu_sum = inst.mus[route].sum()
        tag = "✓" if ok_a else "✗"
        print(f"  {k:>5d}  {len(route):>4d}  {mu_sum:>7.2f}  {p_a:>11.6f}"
              f"  {p_mc:>11.6f}  {tag:>8s}")

    # --- CLT approximation quality check ---
    print("\n▸ CLT approximation quality (analytic vs MC, per route):")
    for k, route in enumerate(nn_routes):
        p_a = route_overflow_probability_analytic(route, inst)
        p_mc = route_overflow_probability_mc(route, inst, n_samples=200_000,
                                             seed=456)
        abs_err = abs(p_a - p_mc)
        print(f"  Route {k}: analytic={p_a:.6f}  MC={p_mc:.6f}  "
              f"|Δ|={abs_err:.6f}")

    print("\n▸ Smoke test passed.")
