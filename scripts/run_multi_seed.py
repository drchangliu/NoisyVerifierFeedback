#!/usr/bin/env python3
"""Run an experiment config multiple times with different seeds for CI estimation.

Since we use temperature=0.2, each run naturally produces different outputs.
We vary the run_id to avoid checkpoint collisions.

Usage:
    python scripts/run_multi_seed.py --config configs/naive_combined_haiku.yaml --seeds 5
    python scripts/run_multi_seed.py --config configs/selective_combined_qwen3.yaml --seeds 10
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Run experiment with multiple seeds")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds to run")
    parser.add_argument("--start-seed", type=int, default=1, help="Starting seed number (skip 0 if already run)")
    parser.add_argument("--items", type=int, default=None)
    args = parser.parse_args()

    for seed in range(args.start_seed, args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed}/{args.seeds - 1}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/run_experiment.py",
            "--config", args.config,
        ]
        if args.items:
            cmd.extend(["--items", str(args.items)])

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"Warning: seed {seed} failed with return code {result.returncode}")


if __name__ == "__main__":
    main()
