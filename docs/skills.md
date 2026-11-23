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

Next to `SKILL.md`, the skill folder holds a `references/` folder with longer
reference files that the agent opens only when it needs them:

- [`formula-syntax.md`](skills/formula-syntax.qmd): every formula operator,
  including `i()`, varying slopes, IV and multiple estimation.
- [`inference.md`](skills/inference.qmd): every `vcov` option and the models
  that support it, small-sample corrections, the wild bootstrap, randomization
  inference and multiple-testing corrections.
- [`coming-from-fixest.md`](skills/coming-from-fixest.qmd): R `fixest`
  spellings and their PyFixest equivalents, and behaviour that differs.
- [`migrating-to-0.70.md`](skills/migrating-to-0.70.qmd): deprecated and
  removed spellings and attributes, with their replacements.

## Use the skill

The skill is the `skills/pyfixest` folder of the installed package. Make it
available to your coding agent in whichever way that agent loads skills or
project instructions, and point the agent at the [llms.txt](llms.txt) index of
this site for the full documentation. To print the folder's location, run this
in the Python environment of your project:

```python
import pathlib, pyfixest
print(pathlib.Path(pyfixest.__file__).parent / "skills" / "pyfixest")
```

The folder belongs to the installed version of PyFixest, so read it in place
rather than from a stale copy after you upgrade.

## Skill content

This is the `SKILL.md` that ships with this version of PyFixest. The header
between the `---` lines is the skill's metadata: its `name` and the
`description` that the agent reads to decide when to load it. The reference
files have their own pages, linked at the top of this page.

````markdown
{{< include ../pyfixest/skills/pyfixest/SKILL.md >}}
````
