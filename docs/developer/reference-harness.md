# Fixest-first reference harness

`scripts/reference/compare_fixest.py` compares one deterministic pyfixest case
with R `fixest`. It is a developer diagnostic and recording tool, not a public
pyfixest API and not a substitute for permanent pytest coverage.

Run it in the R-enabled environment:

```bash
pixi run -e py312-r compare-fixest \
  scripts/reference/cases/feols-smoke.toml
```

The initial adapters support `feols` and `fepois`. Add a small adapter for a
different external package rather than weakening the normalized result contract.

## Case format

Cases are versioned TOML files:

```toml
schema_version = 1
id = "feols-example"
estimator = "feols"
formula = "Y ~ X1 + X2 | f1"
prediction_rows = 5
rtol = 1e-8
atol = 1e-8
prediction_rtol = 1e-5
prediction_atol = 1e-5

[data]
source = "generated"
seed = 9289
n = 500
model = "Feols"

[vcov]
type = "iid"

[ssc]
k_adj = true
k_fixef = "nonnested"
g_adj = true
g_df = "min"
t_df = "min"
```

`data.source` may be `generated` or `csv`. CSV paths are resolved relative
to the case. Vcov types are `iid`, `hetero`, `CRV1`, and `CRV3`; cluster
types also require `vcov.cluster`. A weights column can be supplied with the
top-level `weights` key.

Use tight general tolerances. Separate prediction tolerances are appropriate
when recovered high-dimensional fixed effects follow different iteration paths;
explain them in the case.

## Normalized comparison

The harness aligns coefficients by normalized term name and compares:

- coefficients;
- covariance matrices;
- standard errors;
- inference degrees of freedom;
- observation count;
- dropped variables;
- convergence;
- the requested deterministic prediction subset.

Only understood spellings such as R's `(Intercept)` are normalized. Unknown
structural differences fail instead of being silently reordered or discarded.

## Reports and exit codes

The terminal report shows pass/fail and maximum absolute and relative
differences for every numerical metric.

```bash
# Summary JSON
pixi run -e py312-r compare-fixest <case.toml> \
  --json-output /tmp/comparison.json

# Provenance plus normalized values
pixi run -e py312-r compare-fixest <case.toml> \
  --record /tmp/comparison-record.json
```

Output paths must be explicit, their parent must already exist, and existing
files are never overwritten. Records include the command, UTC creation time,
platform, Python/R/package versions, case hash, data hash, tolerances, metrics,
and normalized results.

Exit status is 0 for parity, 1 for a numerical or structural mismatch, and 2
for an invalid case, missing dependency, or execution/configuration error.

After diagnosing a feature, add the comparison to a permanent
`against_r_core` or `against_r_extended` pytest matrix. Keep the external
software version, deterministic input, and tolerance rationale with that test.
