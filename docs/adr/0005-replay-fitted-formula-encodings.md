# Replay fitted formula encodings on new data

Use the fitted Formulaic specification for prediction and staged estimation.
Reparsing the input formula can relearn categorical references, lose
row-drop information, or interpret a fixed-effect interaction such as `a^b`
as Python XOR. Keep Formulaic-specific representations in the compatibility
adapter so changes to those internals have a single point of entry.

## Formulaic boundary

- Multistage formulas store the parsed first stage in `.deps[0]`. The
  adapter uses the private flatten operation without relying on its output
  order. It reads per-term output columns from `ModelSpec.structure`, factor
  kind and encoder dictionaries from `encoder_state`, and the value wrapped
  by the `FactorValues` proxy for custom encoding. Direct transform calls
  supply the state and specification that Formulaic normally injects.
- Missing names during expression inspection must still allow the walker to
  discover literal quoted lookups in `Q()`. A failed context capture yields an
  empty mapping. Preserve missing-row indices from materialization and add
  drops from later stages. Where Formulaic overwrites equally named columns,
  the surviving column is the last.
- In `i()` interactions, track nulls from both inputs. A complete set of
  categorical cells spans the intercept and needs a dropped cell only when
  the model already spans it. Explicitly dropping a reference removes that
  span. Numeric slopes do not span the intercept. Save the full fitted
  category list so prediction drops the original reference. Factorization
  preserves declared categorical order, and fixed-effect interactions combine
  inputs by position.
- Prediction materializes sub-specifications separately so it can proceed
  without a dependent variable. Intersect their valid rows and restore the
  requested shape with missing predictions and uncertainty where inputs
  were missing or levels were unseen. In particular, an unseen categorical
  level must not silently acquire the reference-level prediction.
- Reuse the fitted model stages when parsing a user-supplied offset expression.
  DID2S replays the first-stage fixed-effect encodings on the full sample.
