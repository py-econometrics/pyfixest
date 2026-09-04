---
title: "Using PyFixest with AI agents"
description: "What ships with a PyFixest release for coding agents, where the installable agent skill lands after pip install, and how to install it."
---

Coding agents work better with PyFixest when they read PyFixest's own
documentation instead of recalling an API from memory. Every release ships two
things for that: a machine-readable copy of this site, and an installable agent
skill that points at it.

## What ships with a release

**The documentation.** Each wheel contains the rendered site as Markdown, next
to the package. Locate it with:

```bash
python -c "import importlib.resources as r; print(r.files('pyfixest') / 'docs')"
```

Start with `cheatsheet.llms.md` — one page with formula syntax, standard errors,
post-estimation methods, tables, and a table naming the page to read for each
task. `llms.txt` in the same directory lists every page. Because the corpus
travels with the wheel, it always describes the installed version.

**The skill.** `SKILL.md` is a short router: it tells the agent how to find the
directory above, which page to open for the task at hand, and a handful of core
facts (formula grammar, `vcov` spellings, one `etable()` call) that PyFixest's
test suite executes on every run.

## Where the skill lands

After `pip install pyfixest`, the skill sits beside the package, at
`<site-packages>/skills/pyfixest/SKILL.md`. Print the directory with:

```bash
python -c "import pyfixest, pathlib; print(pathlib.Path(pyfixest.__file__).parents[1] / 'skills' / 'pyfixest')"
```

## How to install it

Copy that directory into the place your agent reads skills from. For Claude
Code, that is `~/.claude/skills/pyfixest` for every project, or
`.claude/skills/pyfixest` for one project:

```bash
cp -r "$(python -c "import pyfixest, pathlib; print(pathlib.Path(pyfixest.__file__).parents[1] / 'skills' / 'pyfixest')")" ~/.claude/skills/pyfixest
```

You can also install it straight from the repository, without a local PyFixest:

```bash
npx skills add py-econometrics/pyfixest
```

Or open
[`skills/pyfixest/SKILL.md`](https://github.com/py-econometrics/pyfixest/blob/master/skills/pyfixest/SKILL.md)
on GitHub and paste it into whichever skill file your tool reads.

The skill needs `pyfixest` importable in the environment the agent uses, and
nothing else — no API keys, no extra packages. Without a local PyFixest it falls
back to <https://pyfixest.org>.

## See also

- [PyFixest Cheat Sheet](cheatsheet.qmd) — the same task table, executed, on one
  page.
- [llms.txt](llms.txt) — the machine-readable index of this site.
- [py-econometrics/pyfixest](https://github.com/py-econometrics/pyfixest) — the
  source repository, including `AGENTS.md` for agents working *on* PyFixest.
