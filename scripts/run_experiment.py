#!/usr/bin/env python3
"""Main experiment runner.

Usage:
    python scripts/run_experiment.py --config configs/naive_loop.yaml
    python scripts/run_experiment.py --config configs/selective_loop.yaml --items 5
    python scripts/run_experiment.py --config configs/naive_loop.yaml --dry-run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from rich.console import Console
from tqdm import tqdm

load_dotenv()

console = Console()


def load_config(config_path: str) -> dict:
    """Load experiment config, merging with base.yaml."""
    base = yaml.safe_load(Path("configs/base.yaml").read_text())
    override = yaml.safe_load(Path(config_path).read_text())
    for key, val in override.items():
        if isinstance(val, dict) and key in base:
            base[key].update(val)
        else:
            base[key] = val
    return base


def build_agent(config: dict):
    """Build the appropriate agent from config."""
    from nvf.agents.llm_judge import LLMJudgeAgent
    from nvf.agents.naive import NaiveAgent
    from nvf.agents.selective import SelectiveAgent
    from nvf.analyzers.bandit import BanditAnalyzer
    from nvf.analyzers.semgrep import SemgrepAnalyzer
    from nvf.llm.client import LLMClient

    # Build LLM client
    ollama_url = config.get("agent", {}).get("ollama_base_url")
    llm = LLMClient(
        model=config["agent"]["model"],
        temperature=config["agent"]["temperature"],
        ollama_base_url=ollama_url,
    )

    # Build analyzer
    from nvf.analyzers.combined import CombinedAnalyzer

    analyzer_name = config["analyzer"]["name"]
    timeout = config["analyzer"]["timeout"]
    if analyzer_name == "semgrep":
        analyzer = SemgrepAnalyzer(
            config=config["analyzer"]["config"],
            timeout=timeout,
        )
    elif analyzer_name == "bandit":
        analyzer = BanditAnalyzer(timeout=timeout)
    elif analyzer_name == "combined":
        analyzer = CombinedAnalyzer([
            SemgrepAnalyzer(config=config["analyzer"].get("config", "auto"), timeout=timeout),
            BanditAnalyzer(timeout=timeout),
        ])
    else:
        raise ValueError(f"Unknown analyzer: {analyzer_name}")

    condition = config["feedback"]["condition"]
    fmt = config["feedback"]["format"]
    max_iter = config["agent"]["max_iterations"]

    if condition == "naive":
        return NaiveAgent(llm, analyzer, max_iter, fmt)
    elif condition == "selective":
        precision_map = _load_precision_map(config)
        return SelectiveAgent(
            llm, analyzer, precision_map,
            threshold_tau=config["selective"]["threshold_tau"],
            max_iterations=max_iter,
            feedback_format=fmt,
        )
    elif condition == "llm_judge":
        judge_ollama_url = config.get("llm_judge", {}).get("ollama_base_url")
        judge_llm = LLMClient(
            model=config["llm_judge"]["model"],
            temperature=0.0,
            ollama_base_url=judge_ollama_url,
        )
        use_cot = bool(config.get("llm_judge", {}).get("use_cot", False))
        return LLMJudgeAgent(llm, analyzer, judge_llm, max_iter, fmt, use_cot=use_cot)
    elif condition == "adaptive":
        from nvf.agents.adaptive import AdaptiveAgent
        precision_map = _load_precision_map(config)
        prior = config.get("adaptive", {}).get("prior_strength", 1.0)
        return AdaptiveAgent(
            llm, analyzer, precision_map,
            prior_strength=prior,
            max_iterations=max_iter,
            feedback_format=fmt,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


def _load_precision_map(config: dict) -> dict[str, float]:
    """Load precomputed per-rule precision map for selective condition."""
    override = config.get("selective", {}).get("precision_map_path")
    if override:
        path = Path(override)
        if path.exists():
            data = json.loads(path.read_text())
            console.print(f"[green]Loaded precision map (override): {path} ({len(data)} rules)[/green]")
            return {k: v["precision"] for k, v in data.items()}
        console.print(f"[red]precision_map_path set but file missing: {path}[/red]")
    analyzer_name = config["analyzer"]["name"]
    candidates = [
        Path(f"data/calibration/precision_map_{analyzer_name}.json"),
        Path("data/calibration/precision_map.json"),
    ]
    for precision_path in candidates:
        if precision_path.exists():
            data = json.loads(precision_path.read_text())
            console.print(f"[green]Loaded precision map: {precision_path} ({len(data)} rules)[/green]")
            return {k: v["precision"] for k, v in data.items()}
    console.print("[yellow]Warning: No precision map found. Using empty map (all findings suppressed).[/yellow]")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run noisy verifier feedback experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--items", type=int, default=None, help="Limit number of benchmark items")
    parser.add_argument(
        "--shuffle-seed", type=int, default=None,
        help="If set, shuffle the eval_items in-place with this seed before "
             "the loop runs. Used for the adaptive item-ordering ablation: the "
             "adaptive policy's posterior depends on what it sees early, so "
             "varying the order tests robustness to ordering.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Generate run ID
    condition = config["feedback"]["condition"]
    model_short = config["agent"]["model"].split("-")[1] if "-" in config["agent"]["model"] else config["agent"]["model"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = config["output"].get("run_id") or f"{condition}_{model_short}_{timestamp}"
    config["output"]["run_id"] = run_id

    # Create output directory
    output_dir = Path(config["output"]["dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    (output_dir / "config.yaml").write_text(yaml.dump(config, default_flow_style=False))

    if args.dry_run:
        console.print(yaml.dump(config, default_flow_style=False))
        return

    console.print(f"[bold]Run ID:[/bold] {run_id}")
    console.print(f"[bold]Output:[/bold] {output_dir}")
    console.print(f"[bold]Condition:[/bold] {condition}")
    console.print(f"[bold]Model:[/bold] {config['agent']['model']}")

    # Load benchmark
    from nvf.benchmark.loader import load_benchmark
    from nvf.benchmark.splits import split_calibration_eval

    items = load_benchmark(config["benchmark"]["name"])
    cal_frac = config["benchmark"]["calibration_fraction"]
    if cal_frac > 0:
        _, eval_items = split_calibration_eval(
            items,
            calibration_fraction=cal_frac,
            seed=config["seed"],
        )
    else:
        eval_items = items  # Use all items (e.g., SecurityEval with no cal split)

    if args.items:
        eval_items = eval_items[: args.items]

    if args.shuffle_seed is not None:
        import random
        rng = random.Random(args.shuffle_seed)
        eval_items = list(eval_items)
        rng.shuffle(eval_items)
        console.print(f"[bold]Item order shuffled with seed:[/bold] {args.shuffle_seed}")

    console.print(f"[bold]Eval items:[/bold] {len(eval_items)}")

    # Check for existing traces (resume support)
    traces_path = output_dir / "traces.jsonl"
    completed_ids = set()
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                trace_data = json.loads(line)
                completed_ids.add(trace_data["item_id"])
        console.print(f"[green]Resuming: {len(completed_ids)} items already completed[/green]")

    # Build agent
    agent = build_agent(config)

    # Import evaluation
    from nvf.execution.runner import evaluate_trace
    from nvf.llm.cost_tracker import BudgetExceededError, CostTracker

    cost_tracker = CostTracker(max_cost_usd=config["budget"]["max_cost_usd"])

    # Run experiment loop
    pending = [item for item in eval_items if item.item_id not in completed_ids]
    console.print(f"[bold]Running:[/bold] {len(pending)} items")

    for item in tqdm(pending, desc=f"{condition} loop"):
        try:
            cost_tracker.check_budget()
        except BudgetExceededError as e:
            console.print(f"[red]{e}[/red]")
            break

        try:
            trace = agent.run(item)
            evaluate_trace(trace, item)
            cost_tracker.record(
                model=config["agent"]["model"],
                input_tokens=0,
                output_tokens=0,
                cost_usd=trace.total_cost_usd,
            )

            # Append trace to JSONL (checkpoint)
            with open(traces_path, "a") as f:
                f.write(json.dumps(trace.to_dict()) + "\n")

            # Log progress
            last_iter = trace.iterations[-1] if trace.iterations else None
            status = ""
            if last_iter:
                tp = "PASS" if last_iter.tests_passed else "FAIL"
                sec = "SECURE" if not last_iter.has_vulnerability else "VULN"
                status = f"[{tp}/{sec}]"
            tqdm.write(f"  {item.item_id}: {len(trace.iterations)} iters, ${trace.total_cost_usd:.4f} {status}")

        except Exception as e:
            console.print(f"[red]Error on {item.item_id}: {e}[/red]")
            # Write error trace
            with open(traces_path, "a") as f:
                error_trace = {
                    "item_id": item.item_id,
                    "condition": condition,
                    "error": str(e),
                    "iterations": [],
                }
                f.write(json.dumps(error_trace) + "\n")

    # Summary
    console.print(f"\n[bold green]Done![/bold green] Total cost: ${cost_tracker.total_cost_usd:.4f}")
    console.print(f"Traces saved to: {traces_path}")

    # Compute summary metrics
    _print_summary(traces_path)


def _print_summary(traces_path: Path) -> None:
    """Print summary metrics from completed traces."""
    if not traces_path.exists():
        return

    traces = []
    errors = 0
    with open(traces_path) as f:
        for line in f:
            data = json.loads(line)
            if "error" in data:
                errors += 1
            else:
                traces.append(data)

    if not traces:
        return

    console.print(f"\n[bold]Summary ({len(traces)} items, {errors} errors):[/bold]")

    # Compute JointPass at each iteration
    max_k = max(len(t["iterations"]) for t in traces)
    for k in range(max_k):
        joint = 0
        func = 0
        secure = 0
        total = 0
        for t in traces:
            if k < len(t["iterations"]):
                total += 1
                it = t["iterations"][k]
                tp = it.get("tests_passed")
                hv = it.get("has_vulnerability")
                if tp:
                    func += 1
                if hv is False:
                    secure += 1
                if tp and hv is False:
                    joint += 1
        if total > 0:
            console.print(
                f"  k={k}: JointPass={joint}/{total} ({joint/total:.1%}), "
                f"Functional={func}/{total} ({func/total:.1%}), "
                f"Secure={secure}/{total} ({secure/total:.1%})"
            )


if __name__ == "__main__":
    main()
