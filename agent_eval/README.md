# PyFixest agent evaluation

This directory defines a provider-neutral, manual evaluation of how well a
fresh coding agent can use and contribute to PyFixest from local information.
It measures documentation discovery as well as answer correctness: an agent
that eventually reverse-engineers an answer from tests has had a worse
experience than one that reaches an authoritative local guide immediately.

Live-agent runs are deliberately **not a blocking CI job**. They are
nondeterministic, consume an external model, and depend on the surrounding
harness. Deterministic checks for documentation generation, links, public API
contracts, skills, and wheel contents belong in CI instead.

## Suite files

- `tasks.yaml` contains frozen consumer and contributor tasks, their required
  answer points, and intended sources.
- `rubric.yaml` defines the 10-point run-level rubric, hard failures, metadata,
  and aggregate acceptance thresholds.
- `prompt-template.txt` is the provider-neutral prompt hashed into every run
  manifest.
- `cli.py` is a provider-neutral utility that validates the suite, creates inert
  trials, validates manually scored JSON records, aggregates runs, and compares
  a candidate with a protocol-compatible baseline. It uses PyYAML from the
  repository tooling environment and adds no runtime package dependency.
- `baseline-summary.yaml` is the committed result shape. It is currently a
  clearly marked **not-run template**, not evidence of baseline performance.

The two profiles answer different questions:

- **Consumer** tasks test formulas, inference, multiple estimation, reporting,
  GLMs, quantile regression, difference-in-differences, and reduced model
  state.
- **Contributor** tasks test whether an agent can locate the correct extension
  points, follow the estimation pipeline, and cross the Python/Rust boundary
  without putting code in the wrong layer.

## Invariants for a comparable run

Treat `tasks.yaml`, `rubric.yaml`, the agent prompt, source-access rules, and
repeat count as frozen between the baseline and candidate revision. If any of
them changes, create a new suite version and rerun both revisions. Do not compare
results produced by different suite versions.

Each task is an independent trial:

- start a new agent session with no memory of other tasks;
- disable network access and web search;
- provide only the source checkout and/or installed wheel declared in the run
  metadata;
- do not mention likely file paths beyond the task prompt;
- do not let one trial's searches or answer enter another trial's context; and
- do not score a run until the agent has produced its final answer and evidence.

For a wheel-focused consumer evaluation, expose the installed package but not
the repository. For a source-focused contributor evaluation, expose a clean
checkout. A full release evaluation should run both modes; never silently
switch modes between baseline and candidate.

## Baseline and candidate protocol

1. Choose an immutable baseline revision and record its full commit SHA. Use a
   clean worktree; record whether the package came from a wheel, editable
   install, or source tree.
2. Record the agent provider, harness, model identifier/version, system prompt
   identifier or hash, available tools, platform, and package version. If the
   harness does not reveal a value, record `unknown` rather than guessing.
3. Run every task **three times**, each time in a fresh, network-disabled agent
   session using the prompt template below.
4. Preserve the final answer plus the ordered file reads/searches. Count one
   search for each search command or tool call, and one read for each distinct
   file-open call; a single command that opens several files counts once per
   file.
5. Have a reviewer score each run with `rubric.yaml`. Record a hard failure even
   when the arithmetic score would otherwise pass.
6. Populate a copy of `baseline-summary.yaml`; do not replace
   `status: not_run` until all expected runs are present and validated.
7. Repeat the identical protocol at the candidate revision. Report task-level
   medians, unsupported-API failures, intended-first-source rate, aggregate
   score, and deltas from baseline.

## Deterministic helper

The helper never imports a provider SDK, launches an agent, or makes a network
request. Preparing a run only writes prompts, blank JSON records, and a manifest
whose fingerprint covers the tasks, rubric, and prompt:

```bash
pixi run python -m agent_eval.cli validate-suite
pixi run python -m agent_eval.cli prepare \
  --output /tmp/pyfixest-agent-baseline \
  --date 2026-07-11 \
  --pyfixest-ref '<full commit SHA>' \
  --pyfixest-version '<package version>' \
  --source-access-mode source_checkout \
  --install-source source \
  --agent-provider '<provider or unknown>' \
  --agent-harness '<harness and version or unknown>' \
  --model '<model and version or unknown>' \
  --system-prompt-id '<identifier or hash>' \
  --tool shell --tool read \
  --platform '<OS and architecture>'
```

Run each generated prompt independently, review and score its blank record, then
aggregate only after all three repeats are complete:

```bash
pixi run python -m agent_eval.cli validate-results \
  --results /tmp/pyfixest-agent-baseline/records
pixi run python -m agent_eval.cli aggregate \
  --manifest /tmp/pyfixest-agent-baseline/manifest.json \
  --results /tmp/pyfixest-agent-baseline/records \
  --output /tmp/pyfixest-agent-baseline-summary.json
pixi run python -m agent_eval.cli compare \
  --baseline /tmp/pyfixest-agent-baseline-summary.json \
  --candidate /tmp/pyfixest-agent-candidate-summary.json
```

`compare` exits 0 only if all prespecified thresholds pass, 1 for a valid
comparison that fails a threshold, and 2 for malformed or incomparable input.
The CLI uses JSON for normalized records and summaries so it can remain
standard-library-only; the YAML below documents the same fields for humans.

An external launcher can explicitly feed a generated prompt to an agent. For
example, a maintainer with an already-installed local model can use `codex exec
--oss --local-provider ollama --ephemeral --ignore-user-config --sandbox
read-only --cd <clean-checkout> -` and pipe one prompt on standard input. The
launcher, not this repository, must enforce network isolation, start a fresh
process for every prompt, capture the discovery trail, and normalize the final
record. Do not add an implicit or CI-triggered provider adapter here.

The recommended smoke tasks are `consumer-inference-cluster` and
`contributor-python-rust-kernel`. Run the full suite before evaluating an
agent-documentation release.

## Provider-neutral agent prompt

`prompt-template.txt` is the single source of truth. The `prepare` command
substitutes only the profile and task prompt, then hashes each generated prompt
into the manifest. Do not copy the template into a provider-specific launcher
or add hints for an individual task.

## Run record

Store raw transcripts outside the repository when they may contain provider
metadata or sensitive environment details. A normalized result can be attached
to a PR or added to a results artifact with this shape:

```yaml
suite_version: 2
suite_fingerprint: "<manifest fingerprint>"
date: YYYY-MM-DD
profile: consumer
task_id: consumer-formula-sections
run_number: 1
pyfixest_ref: "<full git SHA>"
pyfixest_version: "<installed version>"
install_source: wheel
source_access_mode: installed_package
network: disabled
agent_provider: "<provider or unknown>"
agent_harness: "<harness and version or unknown>"
model: "<model identifier/version or unknown>"
system_prompt_id: "<identifier/hash or unknown>"
tools: ["<tool names>"]
platform: "<OS and architecture>"
elapsed_seconds: 0
commands_or_searches_used: []
files_consulted: []
first_relevant_source: null
final_answer: |
  <answer>
score:
  correctness: 0
  source_discovery: 0
  efficiency: 0
  evidence: 0
  epistemic_discipline: 0
  total: 0
hard_failure: null
hard_failure_code: null
notes: null
```

## Interpretation

A candidate passes only when it satisfies every aggregate threshold in
`rubric.yaml`. The most useful diagnostic is the ordered discovery trail: it
shows whether a local breadcrumb led the agent to the right source before it
fell back to implementation archaeology. Do not claim that documentation
improved from an unrun template or from selected successful trials.
