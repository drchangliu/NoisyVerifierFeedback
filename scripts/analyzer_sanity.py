#!/usr/bin/env python3
"""Analyzer warm-up: verify Semgrep+Bandit produce findings on known-vulnerable
snippets before a multi-seed batch burns API credits on a broken pipeline.

Exit codes:
  0  All canonical snippets produced the expected findings.
  1  At least one snippet produced zero findings — pipeline is broken.
  2  Required analyzer binary not found, even after prepending .venv/bin/.

Usage:
  python scripts/analyzer_sanity.py                   # print + exit code
  python scripts/analyzer_sanity.py --quiet           # exit code only
  scripts/analyzer_sanity.py && scripts/week1_batch.sh sonnet46   # guard
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = PROJ_ROOT / ".venv" / "bin"

# Each canonical snippet should fire at least one rule from at least one analyzer.
# Coverage spans the four CWE families the benchmark exercises most heavily.
CANONICAL = {
    "CWE-502 pickle.loads": "import pickle\ndef load(d): return pickle.loads(d)\n",
    "CWE-95  eval()":       "def run(x): return eval(x)\n",
    "CWE-327 weak hash":    "import hashlib\ndef h(x): return hashlib.md5(x).hexdigest()\n",
    "CWE-78  shell=True":   "import subprocess\ndef ls(d): return subprocess.run('ls ' + d, shell=True)\n",
}


def ensure_path() -> None:
    """Prepend .venv/bin to PATH so subprocess.run('semgrep', ...) resolves."""
    if str(VENV_BIN) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{VENV_BIN}:{os.environ.get('PATH', '')}"


def check_binaries() -> list[str]:
    missing = [b for b in ("semgrep", "bandit") if shutil.which(b) is None]
    return missing


def warm_up(quiet: bool = False) -> int:
    sys.path.insert(0, str(PROJ_ROOT / "src"))
    from nvf.analyzers.bandit import BanditAnalyzer
    from nvf.analyzers.semgrep import SemgrepAnalyzer

    semgrep = SemgrepAnalyzer()
    bandit = BanditAnalyzer()

    failures: list[str] = []
    if not quiet:
        print(f"{'snippet':<22}{'semgrep':>10}{'bandit':>10}   rules")
        print("-" * 70)

    for label, code in CANONICAL.items():
        s_findings = semgrep.analyze(code)
        b_findings = bandit.analyze(code)
        rules = sorted({f.rule_id for f in s_findings + b_findings})
        if not quiet:
            print(f"{label:<22}{len(s_findings):>10}{len(b_findings):>10}   {', '.join(rules[:4])}")
        if not s_findings and not b_findings:
            failures.append(label)

    if failures:
        if not quiet:
            print()
            print(f"FAIL: {len(failures)}/{len(CANONICAL)} canonical snippets produced 0 findings")
            print("Likely causes: corrupted Semgrep rule cache (`rm -rf ~/.semgrep`),")
            print("network unreachable for `--config auto`, or analyzer-pipeline regression.")
        return 1

    if not quiet:
        print()
        print(f"OK: all {len(CANONICAL)} canonical snippets produced findings")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--quiet", action="store_true", help="suppress stdout, exit code only")
    args = p.parse_args()

    ensure_path()

    missing = check_binaries()
    if missing:
        if not args.quiet:
            print(f"FAIL: analyzer binary not found: {', '.join(missing)}", file=sys.stderr)
            print(f"      Checked .venv/bin and system PATH. Did the venv get rebuilt?", file=sys.stderr)
        return 2

    return warm_up(quiet=args.quiet)


if __name__ == "__main__":
    sys.exit(main())
