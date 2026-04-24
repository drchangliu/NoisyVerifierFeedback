#!/usr/bin/env python3
"""Plot empirical adaptive-threshold trajectories from experiment traces.

Reconstructs the τ* trajectory that the AdaptiveAgent maintained during each
run by replaying the Beta-posterior update logic on the saved per-item traces.
Produces a 2×2 figure: top row = simulation, bottom row = empirical.

Usage:
    python scripts/plot_adaptive_empirical.py
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Publication-quality defaults ──────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
})

# Colors
C_HAIKU = "#2166ac"
C_QWEN = "#b2182b"

# Pooled τ* from Table 1 in paper
TAU_TRUE_HAIKU = 0.0    # r=0 for Haiku → τ*=0
TAU_TRUE_QWEN = 0.503   # r/(q+r) for Qwen


def reconstruct_tau_trajectory(traces_path: Path) -> dict:
    """Replay Beta-posterior updates to reconstruct per-item τ* trajectory."""
    q_alpha, q_beta = 1.0, 1.0
    r_alpha, r_beta = 1.0, 1.0

    tau_at_item = []
    update_indices = []  # item index of each update
    tau_after_update = []  # τ* right after the update

    with open(traces_path) as f:
        for i, line in enumerate(f):
            t = json.loads(line)
            if "error" in t:
                continue

            q = q_alpha / (q_alpha + q_beta)
            r = r_alpha / (r_alpha + r_beta)
            tau = r / (q + r) if (q + r) > 0 else 0.5
            tau_at_item.append(tau)

            iters = t["iterations"]
            if len(iters) >= 2:
                k0, k1 = iters[0], iters[1]
                v0, v1 = k0.get("has_vulnerability"), k1.get("has_vulnerability")
                f1 = k1.get("tests_passed")

                if v0 is not None and v1 is not None:
                    if v0:
                        fixed = not v1 and (f1 is True)
                        if fixed:
                            q_alpha += 1
                        else:
                            q_beta += 1
                    else:
                        regressed = v1 or (f1 is False)
                        if regressed:
                            r_alpha += 1
                        else:
                            r_beta += 1

                    q_new = q_alpha / (q_alpha + q_beta)
                    r_new = r_alpha / (r_alpha + r_beta)
                    tau_new = r_new / (q_new + r_new) if (q_new + r_new) > 0 else 0.5
                    update_indices.append(i)
                    tau_after_update.append(tau_new)

    q_final = q_alpha / (q_alpha + q_beta)
    r_final = r_alpha / (r_alpha + r_beta)
    return {
        "tau_at_item": tau_at_item,
        "update_indices": update_indices,
        "tau_after_update": tau_after_update,
        "final": {
            "q": q_final, "r": r_final,
            "tau": r_final / (q_final + r_final) if (q_final + r_final) > 0 else 0.5,
            "tp_fixed": int(q_alpha - 1), "tp_not_fixed": int(q_beta - 1),
            "fp_regressed": int(r_alpha - 1), "fp_ok": int(r_beta - 1),
        },
    }


def collect_runs(results_dir: Path, model_prefix: str, n_items: int = 50):
    """Collect all adaptive runs for a given model and item count."""
    runs = []
    for d in sorted(results_dir.iterdir()):
        if not d.name.startswith(f"adaptive_{model_prefix}"):
            continue
        traces = d / "traces.jsonl"
        if not traces.exists():
            continue
        traj = reconstruct_tau_trajectory(traces)
        if len(traj["tau_at_item"]) == n_items:
            runs.append(traj)
    return runs


def simulate_adaptive(q_true, r_true, n_items=50, n_update_fraction=0.16,
                      prior_strength=1.0, seed=None):
    """Simulate adaptive convergence with realistic update sparsity.

    n_update_fraction: fraction of items that trigger a q/r update
    (empirically ~8/50 ≈ 0.16 for our benchmark).
    """
    rng = random.Random(seed)
    q_alpha = q_beta = prior_strength
    r_alpha = r_beta = prior_strength

    tau_at_item = []
    update_indices = []
    tau_after_update = []

    for i in range(n_items):
        q = q_alpha / (q_alpha + q_beta)
        r = r_alpha / (r_alpha + r_beta)
        tau = r / (q + r) if (q + r) > 0 else 0.5
        tau_at_item.append(tau)

        # Only a fraction of items trigger updates
        if rng.random() < n_update_fraction:
            # Roughly 60% of updates are TP observations, 40% FP
            if rng.random() < 0.6:
                fixed = rng.random() < q_true
                if fixed:
                    q_alpha += 1
                else:
                    q_beta += 1
            else:
                regressed = rng.random() < r_true
                if regressed:
                    r_alpha += 1
                else:
                    r_beta += 1

            q_new = q_alpha / (q_alpha + q_beta)
            r_new = r_alpha / (r_alpha + r_beta)
            tau_new = r_new / (q_new + r_new) if (q_new + r_new) > 0 else 0.5
            update_indices.append(i)
            tau_after_update.append(tau_new)

    return {
        "tau_at_item": tau_at_item,
        "update_indices": update_indices,
        "tau_after_update": tau_after_update,
    }


def plot_trajectories(ax, runs, color, tau_true, model_label,
                      show_ylabel=True, show_136=None):
    """Plot τ* trajectories on a single axis."""
    for i, run in enumerate(runs):
        xs = np.arange(len(run["tau_at_item"]))
        label = f"Seeds ($n$={len(runs)})" if i == 0 else None
        ax.step(xs, run["tau_at_item"], where="post",
                color=color, alpha=0.45, linewidth=1.0, label=label)
        # Mark update points
        if run["update_indices"]:
            label_pt = "Posterior updates" if i == 0 else None
            ax.scatter(run["update_indices"], run["tau_after_update"],
                       color=color, s=20, zorder=5, edgecolors="white",
                       linewidths=0.4, label=label_pt)

    # 136-item run
    if show_136:
        for run in show_136:
            xs = np.arange(len(run["tau_at_item"]))
            ax.step(xs, run["tau_at_item"], where="post",
                    color=color, alpha=0.25, linewidth=0.9, linestyle="--",
                    label="136-item run")
            if run["update_indices"]:
                ax.scatter(run["update_indices"], run["tau_after_update"],
                           color=color, s=15, zorder=5, alpha=0.4,
                           edgecolors="white", linewidths=0.4, marker="D")

    ax.axhline(tau_true, color="#555555", linestyle=":", linewidth=1.0,
               label=f"Pooled $\\tau^* = {tau_true:.2f}$")
    ax.axhline(0.5, color="#aaaaaa", linestyle="-", linewidth=0.5, alpha=0.4)

    ax.set_ylim(-0.05, 0.85)
    ax.set_xlim(-1, max(len(runs[0]["tau_at_item"]), 55) if runs else 55)
    if show_ylabel:
        ax.set_ylabel(r"Estimated $\tau^*$")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)


def main():
    results_dir = Path("data/results")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Collect empirical runs
    haiku_50 = collect_runs(results_dir, "haiku", 50)
    qwen_50 = collect_runs(results_dir, "qwen3", 50)
    haiku_136 = collect_runs(results_dir, "haiku", 136)
    qwen_136 = collect_runs(results_dir, "qwen3", 136)

    print(f"Haiku 50-item runs: {len(haiku_50)}")
    print(f"Qwen  50-item runs: {len(qwen_50)}")

    # Generate simulation runs (matching empirical update sparsity)
    n_sim = 10
    sim_haiku = [simulate_adaptive(q_true=0.262, r_true=0.0, n_items=150,
                                    n_update_fraction=0.16, seed=42+i)
                 for i in range(n_sim)]
    sim_qwen = [simulate_adaptive(q_true=0.423, r_true=0.429, n_items=150,
                                   n_update_fraction=0.16, seed=100+i)
                for i in range(n_sim)]

    # ── 2×2 Figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.08)

    # ── Top row: Simulation ───────────────────────────────────────────────
    ax_sh = fig.add_subplot(gs[0, 0])
    ax_sh.set_title("(a) Simulation: Haiku-like ($q{=}0.26, r{=}0$)")
    for i, run in enumerate(sim_haiku):
        xs = np.arange(len(run["tau_at_item"]))
        label = f"Simulated runs ($n$={n_sim})" if i == 0 else None
        ax_sh.step(xs, run["tau_at_item"], where="post",
                   color=C_HAIKU, alpha=0.35, linewidth=0.9, label=label)
    ax_sh.axhline(TAU_TRUE_HAIKU, color="#555555", linestyle=":", linewidth=1.0,
                  label=r"True $\tau^* = 0.00$")
    ax_sh.axhline(0.5, color="#aaaaaa", linestyle="-", linewidth=0.5, alpha=0.4)
    ax_sh.set_ylim(-0.05, 0.85)
    ax_sh.set_xlim(-1, 155)
    ax_sh.set_ylabel(r"Estimated $\tau^*$")
    ax_sh.set_xlabel("Item index")
    ax_sh.legend(loc="upper right", framealpha=0.9, fontsize=8)

    ax_sq = fig.add_subplot(gs[0, 1], sharey=ax_sh)
    ax_sq.set_title("(b) Simulation: Qwen-like ($q{=}0.42, r{=}0.43$)")
    for i, run in enumerate(sim_qwen):
        xs = np.arange(len(run["tau_at_item"]))
        label = f"Simulated runs ($n$={n_sim})" if i == 0 else None
        ax_sq.step(xs, run["tau_at_item"], where="post",
                   color=C_QWEN, alpha=0.35, linewidth=0.9, label=label)
    ax_sq.axhline(TAU_TRUE_QWEN, color="#555555", linestyle=":", linewidth=1.0,
                  label=r"True $\tau^* = 0.50$")
    ax_sq.axhline(0.5, color="#aaaaaa", linestyle="-", linewidth=0.5, alpha=0.4)
    ax_sq.set_xlim(-1, 155)
    ax_sq.set_xlabel("Item index")
    ax_sq.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.setp(ax_sq.get_yticklabels(), visible=False)

    # ── Bottom row: Empirical ─────────────────────────────────────────────
    ax_eh = fig.add_subplot(gs[1, 0], sharey=ax_sh)
    ax_eh.set_title("(c) Empirical: Haiku 4.5 (50 items)")
    plot_trajectories(ax_eh, haiku_50, C_HAIKU, TAU_TRUE_HAIKU,
                      "Haiku", show_ylabel=True, show_136=haiku_136)
    ax_eh.set_xlabel("Item index")

    ax_eq = fig.add_subplot(gs[1, 1], sharey=ax_sh)
    ax_eq.set_title("(d) Empirical: Qwen3-8B (50 items)")
    plot_trajectories(ax_eq, qwen_50, C_QWEN, TAU_TRUE_QWEN,
                      "Qwen3-8B", show_ylabel=False, show_136=qwen_136)
    ax_eq.set_xlabel("Item index")
    plt.setp(ax_eq.get_yticklabels(), visible=False)

    out_png = fig_dir / "adaptive_empirical.png"
    out_pdf = fig_dir / "adaptive_empirical.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}, {out_pdf}")

    # ── Summary stats ─────────────────────────────────────────────────────
    for label, runs in [("Haiku 50", haiku_50), ("Qwen 50", qwen_50)]:
        if not runs:
            continue
        taus = [r["final"]["tau"] for r in runs]
        n_upd = [len(r["update_indices"]) for r in runs]
        print(f"{label}: final τ* = {np.mean(taus):.3f} ± {np.std(taus):.3f}, "
              f"updates/run = {np.mean(n_upd):.1f}")
        for r in runs:
            f = r["final"]
            print(f"  tp_fixed={f['tp_fixed']}, tp_not={f['tp_not_fixed']}, "
                  f"fp_reg={f['fp_regressed']}, fp_ok={f['fp_ok']} → τ*={f['tau']:.3f}")


if __name__ == "__main__":
    main()
