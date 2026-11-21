# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase. pyfixest is single-context: one glossary, one set of decision records.

## Before exploring, read these

- **`GLOSSARY.md`** at the repo root.
- **Decision records** that touch the area you're about to work in:
  - [`docs/developer/fixest-compatibility.md`](../fixest-compatibility.md): the compatibility ledger of intentional differences from R's `fixest`.
  - [`docs/developer/adr/`](../adr/): architectural decisions and their reasons.

If a file doesn't exist, **proceed silently**. Don't flag its absence and don't suggest creating it upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) adds terms and decisions lazily as they get resolved.

## Where new entries go

- **Glossary terms**: `GLOSSARY.md` currently covers econometric and estimation concepts. Add project-specific software concepts, such as the demeaner backend or a capability declaration, to the same file under their own subheadings once they are resolved. General programming concepts stay out.
- **A difference from `fixest`**: add a ledger entry to `fixest-compatibility.md` following its "Adding an entry" section, not an ADR.
- **Any other ADR**: `docs/developer/adr/NNNN-<slug>.md`. Never `docs/adr/`: Quarto publishes everything under `docs/` except `docs/developer/`.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `GLOSSARY.md`. Don't drift to synonyms the glossary lists under _Avoid_.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag decision conflicts

If your output contradicts an existing ledger entry or ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (…), but worth reopening because…_
