#!/usr/bin/env python3
"""Small live test: run the naive loop on 2-3 CWEval items with a real LLM.

Requires ANTHROPIC_API_KEY in .env or environment.

Usage:
    python scripts/live_test.py
    python scripts/live_test.py --model claude-haiku-4-5-20251001 --items 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Live test of the experiment pipeline")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Model to use")
    parser.add_argument("--items", type=int, default=3, help="Number of items to test")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max feedback iterations")
    args = parser.parse_args()

    from nvf.agents.naive import NaiveAgent
    from nvf.analyzers.semgrep import SemgrepAnalyzer
    from nvf.benchmark.loader import load_cweval
    from nvf.execution.runner import evaluate_trace
    from nvf.llm.client import LLMClient

    console.print(f"[bold]Live test[/bold]: {args.items} items, model={args.model}, max_iter={args.max_iterations}")

    # Load benchmark
    items = load_cweval()
    # Pick items known to work well with our test runner
    good_items = [i for i in items if i.item_id in [
        "CWE-078_0", "CWE-079_0", "CWE-022_0", "CWE-502_0", "CWE-020_0",
        "CWE-095_0", "CWE-117_0", "CWE-377_0", "CWE-918_0",
    ]]
    test_items = good_items[:args.items]

    console.print(f"Testing on: {[i.item_id for i in test_items]}")

    # Build agent
    llm = LLMClient(model=args.model, temperature=0.2)
    analyzer = SemgrepAnalyzer(config="auto", timeout=30)
    agent = NaiveAgent(llm, analyzer, max_iterations=args.max_iterations)

    # Run
    total_cost = 0.0
    results = []

    for item in test_items:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]{item.item_id}[/bold] ({item.cwe_id})")
        console.print(f"Prompt: {item.prompt[:100]}...")

        trace = agent.run(item)
        evaluate_trace(trace, item)
        total_cost += trace.total_cost_usd

        for rec in trace.iterations:
            status_parts = []
            if rec.tests_passed is not None:
                status_parts.append(f"func={'PASS' if rec.tests_passed else 'FAIL'}")
            if rec.has_vulnerability is not None:
                status_parts.append(f"sec={'SECURE' if not rec.has_vulnerability else 'VULN'}")
            status = ", ".join(status_parts)

            console.print(
                f"  Iter {rec.iteration}: {len(rec.findings)} findings, "
                f"{len(rec.feedback_shown)} shown, {status}"
            )

        last = trace.iterations[-1]
        joint = (last.tests_passed and not last.has_vulnerability) if last.tests_passed is not None else None
        results.append({
            "item_id": item.item_id,
            "cwe_id": item.cwe_id,
            "iterations": len(trace.iterations),
            "final_functional": last.tests_passed,
            "final_secure": not last.has_vulnerability if last.has_vulnerability is not None else None,
            "joint_pass": joint,
            "cost_usd": trace.total_cost_usd,
        })

        console.print(f"  [bold]Result: {'JOINT PASS' if joint else 'FAIL'}[/bold] (${trace.total_cost_usd:.4f})")

    # Summary
    console.print(f"\n{'='*60}")
    console.print("[bold]Summary:[/bold]")
    joint_pass = sum(1 for r in results if r["joint_pass"])
    func_pass = sum(1 for r in results if r["final_functional"])
    sec_pass = sum(1 for r in results if r["final_secure"])
    n = len(results)
    console.print(f"  JointPass: {joint_pass}/{n} ({joint_pass/n:.0%})")
    console.print(f"  Functional: {func_pass}/{n} ({func_pass/n:.0%})")
    console.print(f"  Secure: {sec_pass}/{n} ({sec_pass/n:.0%})")
    console.print(f"  Total cost: ${total_cost:.4f}")

    # Save results
    output_path = Path("data/results/live_test_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    console.print(f"  Results saved to: {output_path}")


if __name__ == "__main__":
    main()
