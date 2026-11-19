"""Run the fresh-agent eval: headless Claude Code sessions solve pyfixest tasks.

    python3 scripts/agent_eval/run.py --python <env>/bin/python --out <results dir>

Each session starts in an empty temporary directory holding only the task's
data, without memory, CLAUDE.md, MCP servers, plugins or skills. It may read and
write that directory, read the environment's site-packages and run that
environment's interpreter, nothing else. The runner then executes the agent's
`solution.py` itself and scores it. Needs only the standard library.
"""

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TASKS = Path(__file__).resolve().parent / "tasks.py"
PROMPT = """{prompt}

The data are in data.csv in the current directory. Use the Python package
pyfixest. Run Python only with this interpreter, which has pyfixest installed:
{python}

Write solution.py in the current directory. It must load the data, fit the
model and print one JSON object with exactly the keys {keys}, as unrounded
numbers. Run it to check that it works. You have no internet access.

Create and change files with the Write and Edit tools. The shell accepts only
single commands that start with the interpreter path above."""
# Runs solution.py with deprecations as errors. `-W` cannot name these classes:
# it is parsed before site-packages is importable.
STRICT = """import runpy, warnings
import pandas.errors, pyfixest.errors
for cat in (DeprecationWarning, FutureWarning, pyfixest.errors.PyfixestDeprecationWarning):
    warnings.simplefilter("error", cat)
# pandas 3.1 deprecates `inplace=`, which pyfixest itself still passes.
if hasattr(pandas.errors, "PandasChangeWarning"):
    warnings.simplefilter("ignore", pandas.errors.PandasChangeWarning)
runpy.run_path("solution.py", run_name="__main__")"""
PRIVATE = re.compile(r"\._[A-Za-z]")  # e.g. fit._N, fit._r2


def sh(cmd, cwd=None, env=None, timeout=600):
    kw = {"capture_output": True, "text": True, "timeout": timeout}
    return subprocess.run(cmd, cwd=cwd, env=env, **kw)


def last_json(stdout):
    for line in reversed(stdout.strip().splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def prepare(python, task_id, out):
    """Write the task's data under `out` and return its prompt and reference."""
    (out / task_id / "data").mkdir(parents=True, exist_ok=True)
    res = sh([python, "-I", str(TASKS), task_id, str(out / task_id / "data")])
    if res.returncode:
        raise RuntimeError(f"{task_id}: reference failed\n{res.stderr}")
    task = last_json(res.stdout)
    if not task["pyfixest"].startswith(task["site"]):
        raise RuntimeError(f"pyfixest is not in site-packages: {task['pyfixest']}")
    (out / task_id / "reference.json").write_text(json.dumps(task, indent=1))
    return task


def session(args, task, rep):
    run_dir = args.out / task["id"] / f"rep{rep}"
    run_dir.mkdir(parents=True, exist_ok=True)
    work = Path(os.path.realpath(tempfile.mkdtemp(prefix="pf-agent-eval-")))
    shutil.copy(args.out / task["id"] / "data" / "data.csv", work)
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("CLAUDE", "AI_AGENT"))
    }
    ref, site = task["reference"], os.path.realpath(task["site"])
    prompt = PROMPT.format(
        prompt=task["prompt"], python=args.python, keys=", ".join(ref)
    )
    cmd = [
        "claude", "-p", prompt, "--model", args.model,
        "--max-budget-usd", str(args.budget), "--max-turns", str(args.max_turns),
        "--output-format", "stream-json", "--verbose", "--no-session-persistence",
        "--safe-mode", "--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands",
        "--tools", "Read,Glob,Grep,Write,Edit,Bash", "--permission-mode", "dontAsk",
        "--allowedTools", f"Read(/{work}/**)", f"Edit(/{work}/**)",
        f"Read(/{site}/**)", f"Bash({args.python} *)",
    ]  # fmt: skip
    res = sh(cmd, cwd=work, env=env, timeout=1800)
    (run_dir / "transcript.jsonl").write_text(res.stdout + res.stderr)
    meta = last_json(res.stdout) or {}
    tools = {}  # tool calls by name, e.g. how often it read site-packages
    for line in res.stdout.splitlines():
        msg = json.loads(line) if line.startswith("{") else {}
        # A permission_denied event carries its message as a string.
        message = msg.get("message")
        parts = message.get("content") if isinstance(message, dict) else None
        for part in parts or []:
            if isinstance(part, dict) and part.get("type") == "tool_use":
                tools[part["name"]] = tools.get(part["name"], 0) + 1
    row = {
        "task": task["id"], "rep": rep, "correct": False, "clean": False,
        "turns": meta.get("num_turns"), "cost": meta.get("total_cost_usd", 0.0),
        "end": meta.get("subtype"), "denied": len(meta.get("permission_denials", [])),
        "tools": tools,
    }  # fmt: skip
    solution = work / "solution.py"
    if solution.exists():
        shutil.copy(solution, run_dir)
        plain = sh([args.python, "solution.py"], cwd=work)
        strict = sh([args.python, "-c", STRICT], cwd=work)
        (run_dir / "solution.out").write_text(
            plain.stdout + plain.stderr + strict.stderr
        )
        got = last_json(plain.stdout)
        row["correct"] = isinstance(got, dict) and all(
            isinstance(got.get(k), (int, float))
            and math.isclose(got[k], v, rel_tol=1e-4, abs_tol=1e-10)
            for k, v in ref.items()
        )
        row["clean"] = not strict.returncode and not PRIVATE.search(
            solution.read_text()
        )
    shutil.rmtree(work)
    return row


def summary(rows):
    lines = ["| task | reps | correct | clean | median turns | cost (USD) |"]
    lines.append("|---|---|---|---|---|---|")
    groups = {r["task"]: [] for r in rows}
    for r in rows:
        groups[r["task"]].append(r)
    for name, rs in [*groups.items(), ("**all**", rows)]:
        turns = statistics.median(r["turns"] or 0 for r in rs)
        ok, clean = sum(r["correct"] for r in rs), sum(r["clean"] for r in rs)
        cost = sum(r["cost"] for r in rs)
        lines.append(
            f"| {name} | {len(rs)} | {ok} | {clean} | {turns:g} | {cost:.2f} |"
        )
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--python", required=True, help="interpreter under test")
    p.add_argument("--out", required=True, type=Path, help="results directory")
    p.add_argument("--tasks", help="comma-separated task ids (default: all)")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--model", default="claude-sonnet-5-5")
    p.add_argument("--budget", type=float, default=0.5, help="USD cap per session")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--jobs", type=int, default=4, help="parallel sessions")
    p.add_argument("--summary-only", action="store_true", help="reprint --out's table")
    args = p.parse_args()
    args.python = os.path.abspath(args.python)
    results = args.out / "results.jsonl"
    if not args.summary_only:
        ids = args.tasks.split(",") if args.tasks else None
        ids = ids or last_json(sh([args.python, "-I", str(TASKS)]).stdout)
        tasks = [prepare(args.python, task_id, args.out) for task_id in ids]
        with ThreadPoolExecutor(args.jobs) as pool:
            jobs = [
                pool.submit(session, args, t, i)
                for t in tasks
                for i in range(args.reps)
            ]
            for job in jobs:
                row = job.result()
                print(json.dumps(row), flush=True)
                with results.open("a") as f:
                    f.write(json.dumps(row) + "\n")
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    (args.out / "summary.md").write_text(summary(rows) + "\n")
    print(summary(rows))


if __name__ == "__main__":
    main()
