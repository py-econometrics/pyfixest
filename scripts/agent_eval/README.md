# Fresh-agent eval

Measures how well a coding agent with no project context uses an installed
pyfixest. Each task in `tasks.py` describes an analysis in plain language; a
headless Claude Code session writes `solution.py`, and the runner executes it
and scores:

- **correct**: the printed numbers match the reference (relative tolerance 1e-4);
- **clean**: the script also runs with `DeprecationWarning`, `FutureWarning` and
  `PyfixestDeprecationWarning` raised as errors, and reads no private attribute
  such as `fit._N`;
- **turns** and **cost**, from the session's output.

An agent that can rerun its script fixes whatever raises, so the pass rates
move only on mistakes that run silently or only warn. For the rest, compare
turns. `results.jsonl` also counts each session's tool calls, which shows
whether it read the installed package.

Not part of pytest or CI. A session costs about 0.03 USD with the defaults
(cap 0.50 USD and 20 turns per session).

## Environment under test

Install pyfixest into a fresh virtual environment, not as an editable install,
so that the agent finds the package and anything it ships in `site-packages`:

```bash
python3.12 -m venv /tmp/pf-env
/tmp/pf-env/bin/python -m pip install pyfixest==0.70.0   # or a wheel from `maturin build --release`
cd /tmp && /tmp/pf-env/bin/python -I -c "import pyfixest; print(pyfixest.__file__)"
```

The runner refuses an interpreter whose `pyfixest` is not in `site-packages`.
Keep the environment in a directory of its own, away from `--out`: the prompt
shows the interpreter's path, and the agent's scripts can read anything near it,
including earlier results and references.

## Run

```bash
python3 scripts/agent_eval/run.py --python /tmp/pf-env/bin/python --out /tmp/pf-eval/smoke --tasks quadratic --reps 1
python3 scripts/agent_eval/run.py --python /tmp/pf-env/bin/python --out /tmp/pf-eval/baseline
python3 scripts/agent_eval/run.py --python /tmp/pf-env/bin/python --out /tmp/pf-eval/baseline --summary-only
```

Options: `--model` (default `claude-sonnet-5-5`), `--reps` (3), `--budget` (USD
per session, 0.5), `--max-turns` (20), `--jobs` (parallel sessions, 4). Use a new
`--out` for each run, because `results.jsonl` is appended to. Per task, the
output directory holds the data, the reference and, for each repetition, the
transcript, `solution.py` and its output.

## Isolation

Each session starts in an empty temporary directory that contains only
`data.csv`, with `--safe-mode`, no setting sources, no MCP servers, no slash
commands and no session persistence. It has `Read`, `Glob`, `Grep`, `Write`,
`Edit` and `Bash` in `dontAsk` mode: files may be read and written in the task
directory and read in the environment's `site-packages`, and shell commands
must start with the environment's interpreter (Claude Code also allows
read-only commands such as `head` in the task directory). Anything else is
denied without a prompt. The prompt tells the agent about these limits. The interpreter itself
is not sandboxed, so a script could still reach the network.
