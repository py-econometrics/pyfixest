---
title: "Agent Skill"
description: "Install the PyFixest skill so that coding agents write correct PyFixest code."
---

PyFixest ships an [Agent Skill](https://agentskills.io): a `SKILL.md` file
that tells a coding agent how to call PyFixest correctly. It covers the
estimation functions, formula syntax, standard errors, post-estimation
methods, regression tables, and the spellings that agents tend to carry over
from R `fixest`, Stata or statsmodels. The skill is part of the installed
package, so it matches the PyFixest version in your environment.

## Install the skill

Agents that support skills read them from a skills directory in your project,
for example `.claude/skills/` for Claude Code or `.agents/skills/` for tools
that use that location. Copy the `pyfixest` skill folder from the installed
package into that directory:

```bash
SKILL_DIR=$(python -c "import pathlib, pyfixest; print(pathlib.Path(pyfixest.__file__).parent / 'skills' / 'pyfixest')")
mkdir -p .claude/skills
cp -r "$SKILL_DIR" .claude/skills/
```

Run the commands with the Python environment of your project, and copy the
folder again after you upgrade PyFixest. To keep the skill in step with
upgrades automatically, link the folder instead of copying it:
`ln -s "$SKILL_DIR" .claude/skills/pyfixest`.

The agent then loads the skill when a task involves PyFixest. You can also
point it at the [llms.txt](llms.txt) index of this site.

## Skill content

This is the `SKILL.md` that ships with this version of PyFixest. The header
between the `---` lines is the skill's metadata: its `name` and the
`description` that the agent reads to decide when to load it.

````markdown
{{< include ../pyfixest/skills/pyfixest/SKILL.md >}}
````
