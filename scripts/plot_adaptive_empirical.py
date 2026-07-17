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
C_HAIKU  = "#2166ac"  # low-r model 1
C_GEMMA  = "#1a9850"  # low-r model 2
C_SONNET = "#762a83"  # high-r model 1
C_QWEN   = "#b2182b"  # high-r model 2

# Pooled τ* from Table 5 in paper (see scripts/compute_qr.py)
TAU_TRUE_HAIKU  = 0.262
TAU_TRUE_GEMMA  = 0.322
TAU_TRUE_SONNET = 0.704
TAU_TRUE_QWEN   = 0.656


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
    """Collect all valid adaptive runs for a given model and item count.

    Runs where the analyzer returned zero findings on every item are excluded:
    they are mechanically broken (the 2026-05-28/30 analyzer-PATH failure
    documented in the paper's R2/batch-drift discussion), the loop never
    engages, and their flat tau=0.5 trajectories would bias the panel.
    """
    runs = []
    for d in sorted(results_dir.iterdir()):
        if not d.name.startswith(f"adaptive_{model_prefix}"):
            continue
        traces = d / "traces.jsonl"
        if not traces.exists():
            continue
        recs = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
        recs = [t for t in recs if t.get("iterations")]
        if recs and all(t["iterations"][0].get("n_findings", 0) == 0 for t in recs):
            continue  # mechanically broken zero-finding run
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

    # 136-item run (one legend entry for all such runs)
    if show_136:
        for j, run in enumerate(show_136):
            xs = np.arange(len(run["tau_at_item"]))
            ax.step(xs, run["tau_at_item"], where="post",
                    color=color, alpha=0.25, linewidth=0.9, linestyle="--",
                    label=f"136-item runs ($n$={len(show_136)})" if j == 0 else None)
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
    # Trajectories live in the 0.3-0.6 band, so the bottom of the axis is
    # reliably empty; upper corners are not (dashed 136-item runs, tau* line).
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8)


def main():
    results_dir = Path("data/results")
    fig_dir = Path("figures")
    fig_dir.mkdir(exist_ok=True)

    # Collect empirical runs across all four models.
    haiku_50  = collect_runs(results_dir, "haiku",      50) + collect_runs(results_dir, "haiku",      51)
    qwen_50   = collect_runs(results_dir, "qwen3",      50) + collect_runs(results_dir, "qwen3",      51)
    sonnet_50 = collect_runs(results_dir, "sonnet",     51)
    gemma_50  = collect_runs(results_dir, "gemma4:31b", 51)
    haiku_136 = collect_runs(results_dir, "haiku",     136) + collect_runs(results_dir, "haiku",     137)
    qwen_136  = collect_runs(results_dir, "qwen3",     136)

    print(f"Haiku 50/51-item runs:    {len(haiku_50)}")
    print(f"Sonnet 51-item runs:      {len(sonnet_50)}")
    print(f"Gemma4-31B 51-item runs:  {len(gemma_50)}")
    print(f"Qwen3-8B 50/51-item runs: {len(qwen_50)}")
    print(f"Haiku 136-item runs:      {len(haiku_136)}")

    # Simulation runs (matching empirical update sparsity).
    n_sim = 10
    sim_haiku  = [simulate_adaptive(0.225, 0.080, n_items=150, n_update_fraction=0.16, seed=42+i)  for i in range(n_sim)]
    sim_sonnet = [simulate_adaptive(0.211, 0.500, n_items=150, n_update_fraction=0.16, seed=80+i)  for i in range(n_sim)]
    sim_gemma  = [simulate_adaptive(0.386, 0.183, n_items=150, n_update_fraction=0.16, seed=120+i) for i in range(n_sim)]
    sim_qwen   = [simulate_adaptive(0.280, 0.533, n_items=150, n_update_fraction=0.16, seed=160+i) for i in range(n_sim)]

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(2, 4, hspace=0.40, wspace=0.10)

    # ── Top row: simulation, one panel per model ─────────────────────────
    def _sim_panel(idx, runs, color, tau, label, show_ylabel):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_title(f"({chr(ord('a')+idx)}) Sim: {label}")
        for i, run in enumerate(runs):
            xs = np.arange(len(run["tau_at_item"]))
            lbl = f"Simulated ($n={n_sim}$)" if i == 0 else None
            ax.step(xs, run["tau_at_item"], where="post",
                    color=color, alpha=0.35, linewidth=0.9, label=lbl)
        ax.axhline(tau, color="#555555", linestyle=":", linewidth=1.0,
                   label=rf"True $\tau^*={tau:.2f}$")
        ax.axhline(0.5, color="#aaaaaa", linestyle="-", linewidth=0.5, alpha=0.4)
        ax.set_ylim(-0.05, 0.85)
        ax.set_xlim(-1, 155)
        if show_ylabel: ax.set_ylabel(r"Estimated $\tau^*$")
        else: plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel("Item index")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=7.5)
        return ax

    ax_a = _sim_panel(0, sim_haiku,  C_HAIKU,  TAU_TRUE_HAIKU,  r"Haiku ($q{=}0.23,r{=}0.08$)",  True)
    ax_b = _sim_panel(1, sim_gemma,  C_GEMMA,  TAU_TRUE_GEMMA,  r"Gemma ($q{=}0.39,r{=}0.18$)",  False)
    ax_c = _sim_panel(2, sim_sonnet, C_SONNET, TAU_TRUE_SONNET, r"Sonnet ($q{=}0.21,r{=}0.50$)", False)
    ax_d = _sim_panel(3, sim_qwen,   C_QWEN,   TAU_TRUE_QWEN,   r"Qwen ($q{=}0.28,r{=}0.53$)",   False)

    # ── Bottom row: empirical, one panel per model ──────────────────────
    def _emp_panel(idx, runs, runs136, color, tau, label, show_ylabel):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_title(f"({chr(ord('e')+idx)}) Empirical: {label}")
        if runs:
            plot_trajectories(ax, runs, color, tau, label, show_ylabel=show_ylabel,
                              show_136=runs136 if runs136 else None)
        else:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
            ax.set_ylim(-0.05, 0.85)
        ax.set_xlabel("Item index")
        if not show_ylabel: plt.setp(ax.get_yticklabels(), visible=False)

    _emp_panel(0, haiku_50,  haiku_136, C_HAIKU,  TAU_TRUE_HAIKU,  "Haiku",      True)
    _emp_panel(1, gemma_50,  None,      C_GEMMA,  TAU_TRUE_GEMMA,  "Gemma4-31B", False)
    _emp_panel(2, sonnet_50, None,      C_SONNET, TAU_TRUE_SONNET, "Sonnet",     False)
    _emp_panel(3, qwen_50,   qwen_136,  C_QWEN,   TAU_TRUE_QWEN,   "Qwen3-8B",   False)

    out_png = fig_dir / "adaptive_empirical.png"
    out_pdf = fig_dir / "adaptive_empirical.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"Saved: {out_png}, {out_pdf}")

    for label, runs in [("Haiku", haiku_50), ("Gemma4-31B", gemma_50),
                        ("Sonnet", sonnet_50), ("Qwen3-8B", qwen_50)]:
        if not runs: continue
        taus = [r["final"]["tau"] for r in runs]
        n_upd = [len(r["update_indices"]) for r in runs]
        print(f"{label:<11} final τ̂* = {np.mean(taus):.3f} ± {np.std(taus):.3f}, "
              f"mean updates/run = {np.mean(n_upd):.1f}")


if __name__ == "__main__":
    main()
