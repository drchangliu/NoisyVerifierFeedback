#!/usr/bin/env python3
"""External functional-capability axis for the r-law: HumanEval pass@1 per model.

This breaks the circularity in the capability->r correlation. The within-benchmark
proxies (JP@0, sec@0, CWEval-JP@0) share the `has_vulnerability` signal with r, so
their strong correlations are partly mechanical. HumanEval pass@1 is functional
correctness on DISJOINT items with NO security signal, so it is independent of r;
correlating it with r gives a non-circular test of the law.

Single-shot generation (no feedback loop), temperature 0.2, one sample per task
(pass@1). Resumable: per-model results are appended to data/humaneval/<label>.jsonl
and already-completed task_ids are skipped on re-run.

Usage:
    python scripts/run_humaneval.py                 # all 15 models
    python scripts/run_humaneval.py --models haiku-4.5,sonnet4
    python scripts/run_humaneval.py --limit 2       # smoke test (first 2 tasks)
    python scripts/run_humaneval.py --summary       # just print pass@1 from saved files
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv()  # ANTHROPIC_API_KEY for the Claude models
from nvf.llm.client import LLMClient  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "humaneval"
OLLAMA = "http://localhost:11435"
# Thinking models (qwen3.x, gemma4, deepseek-v4) can burn >4k tokens on
# reasoning alone; at the old 4096 cap the content came back EMPTY and the
# task scored as a failure. 16384 leaves headroom; cap hits are recorded as
# "token-cap" errors so a truncated run can never masquerade as a valid score.
MAX_TOKENS = 16384

# (label, model_id, ollama_base_url|None) -- the 15 distinct models of the r-law.
# Labels match tau_star_prediction / r_capability cohort labels so the
# correlation step can join on them directly.
MODELS = [
    ("haiku-4.5",         "claude-haiku-4-5-20251001", None),
    ("sonnet4",           "claude-sonnet-4-20250514",  None),
    ("sonnet46",          "claude-sonnet-4-6",         None),
    ("opus-4.8",          "claude-opus-4-8",           None),
    ("fable-5",           "claude-fable-5",            None),
    ("gemma4-31b",        "gemma4:31b",                OLLAMA),
    ("gemma3-27b",        "gemma3:27b",                OLLAMA),
    ("qwen3-8b",          "qwen3:8b",                  OLLAMA),
    ("qwen3-14b",         "qwen3:14b",                 OLLAMA),
    ("qwen3-32b",         "qwen3:32b",                 OLLAMA),
    ("qwen3.5-27b",       "qwen3.5:27b",               OLLAMA),
    ("qwen3.6-27b",       "qwen3.6:27b",               OLLAMA),
    ("llama3.1-8b",       "llama3.1:8b",               OLLAMA),
    ("deepseek-v2-16b",   "deepseek-v2:16b",           OLLAMA),
    ("qwen3-coder-480b",  "qwen3-coder:480b-cloud",    OLLAMA),
    ("deepseek-v4-flash", "deepseek-v4-flash:cloud",   OLLAMA),
    ("glm-4.6",           "glm-4.6:cloud",             OLLAMA),
]

SYSTEM = ("You are an expert Python programmer. Complete the requested function. "
          "Respond with ONLY the complete function definition (including any needed "
          "imports) inside a single ```python code block. No explanation.")


def extract_code(text: str) -> str:
    """Pull the code out of a chat reply (prefer the longest fenced block).
    Strip only blank lines, not leading indentation: a body-only reply's
    first line may be legitimately indented, and stripping it would make
    the assembled program a SyntaxError."""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = max(blocks, key=len) if blocks else text
    return code.strip("\n").rstrip()


def build_program(problem: dict, reply: str) -> str:
    """Assemble a runnable program: model's function + HumanEval test harness."""
    entry = problem["entry_point"]
    prompt = problem["prompt"]
    code = extract_code(reply)
    if re.search(rf"def\s+{re.escape(entry)}\b", code):
        # Model returned a full function. Keep the prompt's preamble (imports /
        # helpers that precede the signature) but use the model's definition.
        m = re.search(rf"^(?:async\s+)?def\s+{re.escape(entry)}\b", prompt, re.M)
        preamble = prompt[: m.start()] if m else ""
        body = preamble + "\n" + code
    else:
        # Model returned just the body -> append to the prompt stub.
        body = prompt + "\n" + code
    return body + "\n\n" + problem["test"] + f"\n\ncheck({entry})\n"


def run_one(problem: dict, reply: str, timeout: int = 12) -> tuple[bool, str]:
    program = build_program(problem, reply)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "prog.py"
        f.write_text(program)
        try:
            r = subprocess.run([sys.executable, str(f)], capture_output=True,
                               text=True, timeout=timeout, cwd=tmp)
            if r.returncode == 0:
                return True, ""
            return False, (r.stderr or r.stdout)[-300:]
        except subprocess.TimeoutExpired:
            return False, "timeout"
        except Exception as e:  # pragma: no cover
            return False, f"harness-error: {e}"


def load_done(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["task_id"]] = d
    return done


def gen_and_score(client: LLMClient, problem: dict) -> dict:
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": problem["prompt"]}]
    resp = None
    last_err = ""
    for attempt in range(4):  # survive transient Ollama/API disconnects
        try:
            resp = client.generate(msgs)
            break
        except Exception as e:
            last_err = str(e)
            # 4xx errors (model not found, bad request) are permanent -- a
            # retired model would otherwise burn 164 tasks x 4 backoffs.
            if any(s in last_err for s in ("Error code: 4", "not_found", "404")):
                break
            time.sleep(2 * (attempt + 1))
    if resp is None:
        return {"task_id": problem["task_id"], "passed": False,
                "error": f"gen-error: {last_err}", "cost_usd": 0.0}
    if resp.output_tokens >= MAX_TOKENS or resp.stop_reason in ("length", "max_tokens"):
        return {"task_id": problem["task_id"], "passed": False,
                "error": "token-cap", "cost_usd": resp.cost_usd,
                "out_tokens": resp.output_tokens}
    passed, err = run_one(problem, resp.content)
    return {"task_id": problem["task_id"], "passed": passed, "error": err,
            "cost_usd": resp.cost_usd, "out_tokens": resp.output_tokens}


def purge_capped(path: Path, cap: int) -> int:
    """Drop records whose generation hit the old token cap (empty-content
    truncations misrecorded as failures). Uncapped records are unaffected by
    raising the cap, so keeping them is unbiased."""
    done = load_done(path)
    capped = {t for t, d in done.items()
              if not d["passed"] and (d.get("out_tokens") == cap
                                      or d.get("error") == "token-cap")}
    if capped:
        keep = [json.dumps(d) for t, d in done.items() if t not in capped]
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return len(capped)


def purge_errors(path: Path) -> int:
    """Drop gen-error records (transient outages that survived the retries)
    so the resume pass re-attempts those tasks instead of skipping forever."""
    done = load_done(path)
    bad = {t for t, d in done.items()
           if str(d.get("error", "")).startswith("gen-error")}
    if bad:
        keep = [json.dumps(d) for t, d in done.items() if t not in bad]
        path.write_text("\n".join(keep) + ("\n" if keep else ""))
    return len(bad)


def run_model(label: str, model_id: str, base_url: str | None,
              problems: list, workers: int, redo_capped: bool = False,
              redo_errors: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{label}.jsonl"
    if redo_capped and path.exists():
        n = purge_capped(path, 4096)
        if n:
            print(f"[{label}] purged {n} capped records", flush=True)
    if redo_errors and path.exists():
        n = purge_errors(path)
        if n:
            print(f"[{label}] purged {n} gen-error records", flush=True)
    done = load_done(path)
    todo = [p for p in problems if p["task_id"] not in done]
    client = LLMClient(model=model_id, temperature=0.2, ollama_base_url=base_url,
                       max_tokens=MAX_TOKENS)
    print(f"[{label}] {model_id}: {len(done)} done, {len(todo)} to run", flush=True)
    # Local Ollama is GPU-serialized; only parallelize API/cloud calls.
    nw = workers if (base_url is None or "cloud" in model_id) else 1
    results = []
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(gen_and_score, client, p): p for p in todo}
        with open(path, "a") as fh:
            for fut in as_completed(futs):
                d = fut.result()
                results.append(d)
                fh.write(json.dumps(d) + "\n")
                fh.flush()
    all_d = load_done(path)
    n = len(all_d)
    k = sum(1 for d in all_d.values() if d["passed"])
    cost = sum(d.get("cost_usd", 0.0) for d in all_d.values())
    print(f"[{label}] pass@1 = {k}/{n} = {k/n:.3f}   (cost ${cost:.3f})", flush=True)


def summary() -> None:
    print(f"{'model':<20}{'pass@1':>9}{'n':>5}{'cost$':>9}")
    total = 0.0
    for label, *_ in MODELS:
        d = load_done(OUT / f"{label}.jsonl")
        if not d:
            print(f"{label:<20}{'--':>9}{0:>5}")
            continue
        n = len(d); k = sum(1 for x in d.values() if x["passed"])
        c = sum(x.get("cost_usd", 0.0) for x in d.values())
        total += c
        print(f"{label:<20}{k/n:>9.3f}{n:>5}{c:>9.3f}")
    print(f"\ntotal API cost: ${total:.2f}")


def main() -> None:
    global MAX_TOKENS
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=None, help="comma-separated labels subset")
    ap.add_argument("--limit", type=int, default=None, help="first N tasks (smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent API/cloud calls")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--redo-capped", action="store_true",
                    help="purge failures that hit a token cap and re-run them")
    ap.add_argument("--redo-errors", action="store_true",
                    help="purge gen-error records (transient outages) and re-run them")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="generation budget (last-resort raise for tasks that "
                         "cap even at the default)")
    args = ap.parse_args()
    MAX_TOKENS = args.max_tokens

    if args.summary:
        summary()
        return

    from datasets import load_dataset
    problems = list(load_dataset("openai_humaneval")["test"])
    if args.limit:
        problems = problems[: args.limit]

    wanted = set(args.models.split(",")) if args.models else None
    for label, model_id, base_url in MODELS:
        if wanted and label not in wanted:
            continue
        run_model(label, model_id, base_url, problems, args.workers,
                  redo_capped=args.redo_capped, redo_errors=args.redo_errors)

    print("\n=== summary ===")
    summary()


if __name__ == "__main__":
    main()
