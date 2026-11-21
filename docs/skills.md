---
title: "Agent Skill"
description: "Use the PyFixest skill so that coding agents write correct PyFixest code."
---

PyFixest ships an [Agent Skill](https://agentskills.io): a `SKILL.md` file
that tells a coding agent how to call PyFixest correctly. It covers the
estimation functions, formula syntax, standard errors, post-estimation
methods, regression tables, and the spellings that agents tend to carry over
from R `fixest`, Stata or statsmodels. The skill is part of the installed
package, so it matches the PyFixest version in your environment.

## Use the skill

The skill is the `skills/pyfixest` folder of the installed package. Make it
available to your coding agent in whichever way that agent loads skills or
project instructions, and point the agent at the [llms.txt](llms.txt) index of
this site for the full documentation.

## Skill content

This is the `SKILL.md` that ships with this version of PyFixest. The header
between the `---` lines is the skill's metadata: its `name` and the
`description` that the agent reads to decide when to load it.

````markdown
{{< include ../pyfixest/skills/pyfixest/SKILL.md >}}
````
