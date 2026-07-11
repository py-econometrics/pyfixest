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

The recommended smoke tasks are `consumer-inference-cluster` and
`contributor-python-rust-kernel`. Run the full suite before evaluating an
agent-documentation release.

## Provider-neutral agent prompt

Replace the braces with the task and profile from `tasks.yaml`. Do not add hints
for an individual task.

```text
You are evaluating PyFixest's local documentation as a fresh {profile} agent.

Rules:
- Use only the local files and installed package made available to you.
- Do not use the network or web search.
- Prefer user documentation, packaged documentation, package docstrings,
  architecture guidance, and explicit repository breadcrumbs over tests.
- Do not modify files or run commands that change repository state.
- Answer the task directly. Do not invent an API or a file you did not verify.
- After the answer, emit the evidence fields below exactly once.

Evidence:
files_consulted:
  - <ordered local path, or []>
first_relevant_source: <local path or null>
commands_or_searches_used:
  - <ordered command/search, or []>
final_answer: |
  <the answer you would give the user>

Task:
{task_prompt}
```

## Run record

Store raw transcripts outside the repository when they may contain provider
metadata or sensitive environment details. A normalized result can be attached
to a PR or added to a results artifact with this shape:

```yaml
suite_version: 2
date: YYYY-MM-DD
profile: consumer
task_id: consumer-formula-sections
run_number: 1
pyfixest_ref: "<full git SHA>"
pyfixest_version: "<installed version>"
install_source: wheel
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
notes: null
```

## Interpretation

A candidate passes only when it satisfies every aggregate threshold in
`rubric.yaml`. The most useful diagnostic is the ordered discovery trail: it
shows whether a local breadcrumb led the agent to the right source before it
fell back to implementation archaeology. Do not claim that documentation
improved from an unrun template or from selected successful trials.
