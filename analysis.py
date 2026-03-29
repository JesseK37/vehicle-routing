"""
analysis.py — Solution Analysis for the Chance-Constrained VRP
==============================================================

**Purpose**

This module produces the full suite of research-grade figures for
Project 4.  It runs the SA solver, evaluates the stochastic vs
deterministic baseline, and generates publication-quality plots
on a dark background (#0d1117) following the portfolio style guide.

**Figure Catalogue**

1. **Route map** — Customer locations, depot, and route assignments
   on the unit square.  Stochastic (SA) vs deterministic (NN) side
   by side.
2. **SA convergence** — Distance, max overflow probability,
   temperature, and acceptance rate over iterations (4-panel).
3. **Cost-vs-robustness Pareto front** — Total travel distance vs
   ε threshold, swept across a grid of ε values.  Reveals the
   price of robustness.
4. **Monte Carlo validation** — Per-route overflow probability:
   analytic (CLT) vs Monte Carlo (true log-normal).  Validates
   the fast-path approximation.
5. **Demand distribution** — Histogram of realised route demands
   from MC simulation overlaid with the Normal approximation and
   capacity line.
6. **CLT approximation error** — Scatter of analytic vs MC overflow
   probability across all routes and ε values.

**Style Guide**

* Background: ``#0d1117`` (GitHub dark)
* Axes/text: ``#c9d1d9``
* Grid: ``#21262d``
* Palette: ``["#58a6ff", "#f78166", "#7ee787", "#d2a8ff",
  "#ff7b72", "#79c0ff", "#ffa657"]``
* Font: DejaVu Sans, 10pt default
* All figures saved at 200 dpi as PNG.

Author : Portfolio — Project 4 (Combinatorial Optimisation)
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm

from instance_generator import (
    ProblemInstance,
    generate_instance,
    instance_summary,
    nearest_neighbour_deterministic,
    route_distance,
    route_overflow_probability_analytic,
    route_overflow_probability_mc,
    check_chance_constraint,
)
from sa_solver import (
    SAConfig,
    SAResult,
    Solution,
    evaluate_solution,
    solve,
    solution_summary,
)


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#21262d"
PALETTE = ["#58a6ff", "#f78166", "#7ee787", "#d2a8ff",
           "#ff7b72", "#79c0ff", "#ffa657"]

DPI = 200
OUT_DIR = "figures"


def _apply_style(ax: plt.Axes) -> None:
    """Apply dark-background styling to an axes."""
    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, which="both")
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


def _save(fig: plt.Figure, name: str) -> str:
    """Save figure and return the path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ===================================================================
# Figure 1 — Route Map (side-by-side: NN baseline vs SA solution)
# ===================================================================

def plot_route_map(
    instance: ProblemInstance,
    nn_routes: list[list[int]],
    sa_routes: list[list[int]],
    sa_overflow: list[float],
    nn_overflow: list[float],
    epsilon: float,
) -> str:
    """Side-by-side route maps: deterministic NN vs stochastic SA."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor=BG)

    for ax_idx, (ax, routes, overflows, title_str) in enumerate(zip(
        axes,
        [nn_routes, sa_routes],
        [nn_overflow, sa_overflow],
        ["Nearest-Neighbour (deterministic)", "Simulated Annealing (stochastic)"],
    )):
        _apply_style(ax)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title_str, fontsize=12, fontweight="bold")

        # Depot
        dx, dy = instance.depot
        ax.plot(dx, dy, "s", color="#ffa657", markersize=12, zorder=5,
                markeredgecolor="#c9d1d9", markeredgewidth=1.2)
        ax.annotate("Depot", (dx, dy), textcoords="offset points",
                    xytext=(8, 8), color="#ffa657", fontsize=8,
                    fontweight="bold")

        # Routes
        total_dist = 0.0
        for k, route in enumerate(routes):
            if not route:
                continue
            color = PALETTE[k % len(PALETTE)]
            p = overflows[k]
            feasible = p <= epsilon

            # Build full path: depot → customers → depot
            xs = [dx] + [instance.customers[i].x for i in route] + [dx]
            ys = [dy] + [instance.customers[i].y for i in route] + [dy]

            linestyle = "-" if feasible else "--"
            linewidth = 1.6 if feasible else 1.2
            alpha = 1.0 if feasible else 0.6

            ax.plot(xs, ys, color=color, linewidth=linewidth,
                    linestyle=linestyle, alpha=alpha, zorder=2)

            # Customer markers
            cx = [instance.customers[i].x for i in route]
            cy = [instance.customers[i].y for i in route]
            ax.scatter(cx, cy, color=color, s=35, zorder=4,
                       edgecolors="#0d1117", linewidths=0.5)

            d = route_distance(route, instance)
            total_dist += d

        # Summary annotation
        n_feasible = sum(1 for p in overflows if p <= epsilon)
        n_routes = sum(1 for r in routes if r)
        max_p = max(overflows) if overflows else 0
        info = (f"Routes: {n_routes}  |  Dist: {total_dist:.3f}\n"
                f"Feasible: {n_feasible}/{n_routes}  |  "
                f"Max P(overflow): {max_p:.4f}")
        ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=7.5,
                color=FG, verticalalignment="bottom",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#161b22",
                          edgecolor=GRID, alpha=0.9))

    fig.suptitle("CC-VRP Route Comparison", color=FG, fontsize=14,
                 fontweight="bold", y=0.98)
    return _save(fig, "01_route_map.png")


# ===================================================================
# Figure 2 — SA Convergence (4-panel)
# ===================================================================

def plot_convergence(result: SAResult) -> str:
    """Four-panel SA convergence: distance, overflow, temperature, acceptance."""

    h = result.history
    iters = np.array(h["iteration"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BG)
    panels = [
        ("distance", "Total Distance", PALETTE[0], False),
        ("max_overflow", "Max P(overflow)", PALETTE[1], False),
        ("temperature", "Temperature", PALETTE[3], True),
        ("acceptance_rate", "Acceptance Rate", PALETTE[2], False),
    ]

    for ax, (key, ylabel, color, use_log) in zip(axes.flat, panels):
        _apply_style(ax)
        vals = np.array(h[key])
        ax.plot(iters, vals, color=color, linewidth=1.4)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if use_log:
            ax.set_yscale("log")

        # Add epsilon line on overflow panel
        if key == "max_overflow":
            ax.axhline(y=result.config.epsilon, color="#ff7b72",
                       linestyle="--", linewidth=1.0, alpha=0.8,
                       label=f"ε = {result.config.epsilon}")
            ax.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
                      fontsize=8)

    fig.suptitle("Simulated Annealing Convergence", color=FG, fontsize=14,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "02_convergence.png")


# ===================================================================
# Figure 3 — Cost-vs-Robustness Pareto Front
# ===================================================================

def plot_pareto_front(
    instance: ProblemInstance,
    epsilon_grid: np.ndarray | None = None,
    sa_iter: int = 80_000,
) -> tuple[str, dict]:
    """Sweep ε and plot the cost-of-robustness trade-off.

    Returns the figure path and the sweep data dict.
    """
    if epsilon_grid is None:
        epsilon_grid = np.array([0.01, 0.03, 0.05, 0.08,
                                 0.10, 0.15, 0.20, 0.30])

    results_data = {
        "epsilon": [],
        "distance": [],
        "max_overflow": [],
        "feasible": [],
        "routes_used": [],
    }

    print(f"\n  Pareto sweep: {len(epsilon_grid)} ε values × "
          f"{sa_iter:,} iterations each")

    for eps in epsilon_grid:
        cfg = SAConfig(
            max_iter=sa_iter,
            epsilon=eps,
            seed=0,
            verbose=False,
        )
        res = solve(instance, cfg)
        sol = res.best

        results_data["epsilon"].append(eps)
        results_data["distance"].append(sol.total_distance)
        results_data["max_overflow"].append(sol.max_overflow)
        results_data["feasible"].append(sol.is_feasible)
        results_data["routes_used"].append(
            sum(1 for r in sol.routes if r)
        )
        tag = "✓" if sol.is_feasible else "✗"
        print(f"    ε={eps:.2f}  dist={sol.total_distance:.4f}  "
              f"maxP={sol.max_overflow:.4f}  {tag}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5.5), facecolor=BG)
    _apply_style(ax1)

    eps_arr = np.array(results_data["epsilon"])
    dist_arr = np.array(results_data["distance"])
    feas_arr = np.array(results_data["feasible"])

    # Distance vs epsilon
    ax1.plot(eps_arr, dist_arr, "o-", color=PALETTE[0], linewidth=2,
             markersize=7, markeredgecolor=BG, markeredgewidth=1.2,
             label="Total distance", zorder=3)

    # Mark infeasible points
    if (~feas_arr).any():
        ax1.scatter(eps_arr[~feas_arr], dist_arr[~feas_arr],
                    s=120, facecolors="none", edgecolors="#ff7b72",
                    linewidths=2, zorder=4, label="Infeasible")

    ax1.set_xlabel("ε (allowable overflow probability)", fontsize=11)
    ax1.set_ylabel("Total Travel Distance", fontsize=11, color=PALETTE[0])
    ax1.tick_params(axis="y", labelcolor=PALETTE[0])

    # NN baseline reference
    nn_routes = nearest_neighbour_deterministic(instance)
    nn_dist = sum(route_distance(r, instance) for r in nn_routes)
    ax1.axhline(y=nn_dist, color="#ffa657", linestyle=":", linewidth=1.2,
                alpha=0.8, label=f"NN baseline ({nn_dist:.3f})")

    # Secondary axis: routes used
    ax2 = ax1.twinx()
    ax2.bar(eps_arr, results_data["routes_used"], width=0.008,
            color=PALETTE[2], alpha=0.35, zorder=1, label="Routes used")
    ax2.set_ylabel("Routes Used", fontsize=11, color=PALETTE[2])
    ax2.tick_params(axis="y", labelcolor=PALETTE[2])
    ax2.set_ylim(0, instance.n_vehicles + 2)
    for spine in ax2.spines.values():
        spine.set_color(GRID)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
               fontsize=9, loc="upper right")

    ax1.set_title("Cost of Robustness: Distance vs Overflow Tolerance",
                  fontsize=13, fontweight="bold", color=FG)

    fig.tight_layout()
    return _save(fig, "03_pareto_front.png"), results_data


# ===================================================================
# Figure 4 — Monte Carlo Validation (analytic vs MC per route)
# ===================================================================

def plot_mc_validation(
    instance: ProblemInstance,
    sa_routes: list[list[int]],
    nn_routes: list[list[int]],
    n_mc: int = 200_000,
) -> str:
    """Compare analytic and MC overflow probabilities per route."""

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BG)
    _apply_style(ax)

    all_routes = []
    labels = []
    for k, route in enumerate(sa_routes):
        if route:
            all_routes.append(route)
            labels.append(f"SA-{k}")
    for k, route in enumerate(nn_routes):
        if route:
            all_routes.append(route)
            labels.append(f"NN-{k}")

    p_analytic = []
    p_mc = []
    for route in all_routes:
        pa = route_overflow_probability_analytic(route, instance)
        pm = route_overflow_probability_mc(route, instance,
                                           n_samples=n_mc, seed=999)
        p_analytic.append(pa)
        p_mc.append(pm)

    p_analytic = np.array(p_analytic)
    p_mc = np.array(p_mc)

    # 45-degree line
    max_val = max(p_analytic.max(), p_mc.max()) * 1.15
    ax.plot([0, max_val], [0, max_val], "--", color=FG, alpha=0.4,
            linewidth=1, label="Perfect agreement")

    # SA routes
    n_sa = sum(1 for r in sa_routes if r)
    ax.scatter(p_mc[:n_sa], p_analytic[:n_sa], color=PALETTE[0], s=70,
               zorder=3, edgecolors=BG, linewidths=0.8, label="SA routes")

    # NN routes
    ax.scatter(p_mc[n_sa:], p_analytic[n_sa:], color=PALETTE[1], s=70,
               zorder=3, edgecolors=BG, linewidths=0.8, marker="^",
               label="NN routes")

    # Label points
    for i, label in enumerate(labels):
        ax.annotate(label, (p_mc[i], p_analytic[i]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color=FG, alpha=0.8)

    ax.set_xlabel("Monte Carlo P(overflow)", fontsize=11)
    ax.set_ylabel("Analytic (CLT) P(overflow)", fontsize=11)
    ax.set_title("CLT Approximation Validation",
                 fontsize=13, fontweight="bold", color=FG)
    ax.legend(facecolor="#161b22", edgecolor=GRID, labelcolor=FG,
              fontsize=9)

    fig.tight_layout()
    return _save(fig, "04_mc_validation.png")


# ===================================================================
# Figure 5 — Demand Distribution (MC histogram + Normal overlay)
# ===================================================================

def plot_demand_distributions(
    instance: ProblemInstance,
    sa_routes: list[list[int]],
    n_samples: int = 100_000,
) -> str:
    """For each SA route, show MC demand histogram vs Normal approximation."""

    active_routes = [(k, r) for k, r in enumerate(sa_routes) if r]
    n_routes = len(active_routes)
    ncols = min(n_routes, 3)
    nrows = (n_routes + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             facecolor=BG, squeeze=False)

    rng = np.random.default_rng(42)

    for idx, (k, route) in enumerate(active_routes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        _apply_style(ax)

        # MC samples from true log-normal
        total = np.zeros(n_samples)
        for i in route:
            c = instance.customers[i]
            total += rng.lognormal(mean=c.ln_mu, sigma=c.ln_sigma,
                                   size=n_samples)

        # Normal approximation parameters
        mu_sum = instance.mus[route].sum()
        var_sum = (instance.sigmas[route] ** 2).sum()
        sigma_sum = np.sqrt(var_sum)

        # Histogram
        ax.hist(total, bins=80, density=True, color=PALETTE[idx % len(PALETTE)],
                alpha=0.55, edgecolor="none", label="MC (log-normal)")

        # Normal overlay
        x_range = np.linspace(total.min(), total.max(), 300)
        pdf = norm.pdf(x_range, loc=mu_sum, scale=sigma_sum)
        ax.plot(x_range, pdf, color="#c9d1d9", linewidth=1.8,
                linestyle="--", label="Normal approx.")

        # Capacity line
        ax.axvline(x=instance.capacity, color="#ff7b72", linewidth=1.5,
                   linestyle="-", alpha=0.9, label=f"Q = {instance.capacity}")

        # Overflow shading
        overflow_mask = x_range >= instance.capacity
        if overflow_mask.any():
            ax.fill_between(x_range, 0, pdf, where=overflow_mask,
                            color="#ff7b72", alpha=0.15)

        ax.set_xlabel("Route Demand")
        ax.set_ylabel("Density")
        ax.set_title(f"Route {k}  (|R|={len(route)}, Σμ={mu_sum:.1f})",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#161b22", edgecolor=GRID,
                  labelcolor=FG)

    # Hide unused subplots
    for idx in range(n_routes, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle("Route Demand Distributions: MC vs Normal Approximation",
                 color=FG, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _save(fig, "05_demand_distributions.png")


# ===================================================================
# Figure 6 — CLT Error Heatmap across ε sweep
# ===================================================================

def plot_clt_error_summary(
    instance: ProblemInstance,
    pareto_data: dict,
    n_mc: int = 100_000,
) -> str:
    """Scatter of CLT approximation error across the Pareto sweep."""

    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=BG)
    _apply_style(ax)

    eps_values = pareto_data["epsilon"]

    all_pa = []
    all_pm = []
    all_sizes = []

    for eps in eps_values:
        cfg = SAConfig(max_iter=80_000, epsilon=eps, seed=0, verbose=False)
        res = solve(instance, cfg)
        for route in res.best.routes:
            if not route:
                continue
            pa = route_overflow_probability_analytic(route, instance)
            pm = route_overflow_probability_mc(route, instance,
                                               n_samples=n_mc, seed=777)
            all_pa.append(pa)
            all_pm.append(pm)
            all_sizes.append(len(route))

    all_pa = np.array(all_pa)
    all_pm = np.array(all_pm)
    all_sizes = np.array(all_sizes)
    errors = all_pa - all_pm

    scatter = ax.scatter(all_pm, errors, c=all_sizes, cmap="plasma",
                         s=50, edgecolors=BG, linewidths=0.5, zorder=3,
                         vmin=all_sizes.min(), vmax=all_sizes.max())
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Route size |R|", color=FG, fontsize=10)
    cbar.ax.tick_params(colors=FG)

    ax.axhline(y=0, color=FG, linewidth=0.8, alpha=0.4)
    ax.set_xlabel("MC P(overflow) [true log-normal]", fontsize=11)
    ax.set_ylabel("CLT Error  (analytic − MC)", fontsize=11)
    ax.set_title("CLT Approximation Error vs True Overflow Probability",
                 fontsize=13, fontweight="bold", color=FG)

    fig.tight_layout()
    return _save(fig, "06_clt_error.png")


# ===================================================================
# Main — generate all figures
# ===================================================================

if __name__ == "__main__":
    t0 = time.perf_counter()

    print("=" * 60)
    print("CC-VRP Analysis — Generating All Figures")
    print("=" * 60)

    # --- Instance ---
    print("\n▸ Generating instance …")
    inst = generate_instance(n_customers=25, seed=42)
    print(instance_summary(inst))

    # --- NN baseline ---
    print("\n▸ Running nearest-neighbour baseline …")
    nn_routes = nearest_neighbour_deterministic(inst)
    nn_sol = evaluate_solution(nn_routes, inst, epsilon=0.05, lam=10.0)
    nn_dist = nn_sol.total_distance
    print(f"  NN: {len(nn_routes)} routes, dist={nn_dist:.4f}, "
          f"maxP={nn_sol.max_overflow:.4f}, feasible={nn_sol.is_feasible}")

    # --- SA solver (main run) ---
    print("\n▸ Running SA solver (100k iterations) …")
    cfg = SAConfig(
        max_iter=100_000,
        epsilon=0.05,
        seed=0,
        verbose=True,
        log_interval=25_000,
    )
    result = solve(inst, cfg)
    print()
    print(solution_summary(result, inst))

    # --- Figure 1: Route map ---
    print("\n▸ Figure 1: Route map …")
    plot_route_map(inst, nn_routes, result.best.routes,
                   result.best.overflow_probs, nn_sol.overflow_probs,
                   epsilon=0.05)

    # --- Figure 2: Convergence ---
    print("\n▸ Figure 2: SA convergence …")
    plot_convergence(result)

    # --- Figure 3: Pareto front ---
    print("\n▸ Figure 3: Cost-vs-robustness Pareto front …")
    pareto_path, pareto_data = plot_pareto_front(inst, sa_iter=80_000)

    # --- Figure 4: MC validation ---
    print("\n▸ Figure 4: MC validation …")
    plot_mc_validation(inst, result.best.routes, nn_routes)

    # --- Figure 5: Demand distributions ---
    print("\n▸ Figure 5: Demand distributions …")
    plot_demand_distributions(inst, result.best.routes)

    # --- Figure 6: CLT error summary ---
    print("\n▸ Figure 6: CLT error summary …")
    plot_clt_error_summary(inst, pareto_data)

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"All figures generated in {elapsed:.1f} s")
    print(f"Output directory: {OUT_DIR}/")
    print(f"{'=' * 60}")
    print("\n▸ Smoke test passed.")
