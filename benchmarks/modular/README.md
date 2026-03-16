# Modular Benchmarks

## 1. Model

The estimating equation is a three-way fixed effects model:

$$Y_{it} = \alpha_i + \psi_{J(i,t)} + \phi_t + X_{it}'\beta + \varepsilon_{it}$$

where $i = 1, \ldots, N_w$ indexes workers, $t = 1, \ldots, T$ indexes time periods,
$J(i,t) \in \{1, \ldots, N_f\}$ maps worker $i$ at time $t$ to a firm, $\alpha_i$ is a
worker fixed effect, $\psi_j$ is a firm fixed effect, and $\phi_t$ is a time fixed
effect.

The standard AKM panel carries three ID columns (`indiv_id`, `firm_id`, `year`)
and the standard sweep benchmarks the three-way specification
($\alpha_i + \psi_j + \phi_t$). Occupation-enabled scenarios add a fourth ID
column, `occ_id`, for the occupation extension benchmark.

Implementation: `akm_dgp.py` → `simulate_akm_panel(config: AKMConfig, ...)`.

---

## 2. DGP Primitives

### 2.1 Industry and firm structure

$S$ industries (`n_industries`). Each firm assigned uniformly: $s(j) \sim \text{Cat}(1/S)$.

**Firm effects:** $\psi_j \sim N(0, \sigma^2_\psi)$ (`var_psi`).

**Firm size:** Pareto-like weights controlled by shape $\gamma$ (`gamma`):

$$L_j \propto U_j^{-1/\gamma}, \quad U_j \sim \text{Uniform}(0,1)$$

$\gamma = 1$ gives Pareto(1) (skewed); $\gamma \to \infty$ gives uniform sizes.
Code: `_firm_size_weights`.

**Size-quality correlation:** Firm effects $\psi_j$ and sizes $L_j$ are coupled
via a Gaussian copula with correlation $\rho_{\text{size}}$ (`rho_size`).
Code: `_couple_by_rank`.

### 2.2 Worker effects

$\alpha_i \sim N(0, \sigma^2_\alpha)$ (`var_alpha`). Each worker has a home industry
$s_i \sim \text{Cat}(1/S)$.

### 2.3 Matching

The probability worker $i$ matches to firm $j$ is:

$$\pi_{ij} \propto \exp\!\left(- \frac{\rho}{2\tau^2} (\alpha_i - \psi_j)^2 \right) \cdot L_j \cdot w^{\text{ind}}_{ij}$$

where $\tau^2 = \sigma^2_\alpha + \sigma^2_\psi$ and:

- **Sorting** $\rho$ (`rho`): $\rho = 0$ is random matching; $\rho \to \infty$ is perfectly assortative.
- **Industry weight** $\lambda$ (`lambda_`):

$$w^{\text{ind}}_{ij} = \begin{cases} \lambda & \text{if } s(j) = s_i \\ (1 - \lambda)/(S - 1) & \text{otherwise} \end{cases}$$

$\lambda = 1/S$: industry irrelevant. $\lambda = 1$: workers never leave their industry.

For efficiency, workers are bucketed by $\alpha$-rank into `n_match_bins` bins
and CDFs are precomputed per (industry, bin) pair. Code: `_alpha_bins`, `_build_assignment_cdfs`.

### 2.4 Mobility

At $t=1$, each worker draws a firm from $\pi_{ij}$. For $t > 1$, the worker
separates with probability $\delta$ (`delta`) and draws a new firm (excluding
the current one). Expected tenure: $1/\delta$ periods. Code: `_sample_firms`.

### 2.5 Occupations (optional extension)

When occupation is enabled, each firm draws a sparse menu of occupations from an
industry-level occupation pool and one primary occupation inside that menu.
Workers draw their initial occupation from their firm's menu, with `occ_lambda`
controlling how concentrated that draw is on the firm's primary occupation.

On a firm move, occupation evolves conditionally on the destination firm:

- If the worker's old occupation is in the destination firm's menu, they keep it
  with probability `occ_delta`.
- Otherwise, they are forced to redraw from the destination firm's occupation menu.

This creates two separate hardness levers for the occupation benchmark:

- High `occ_lambda` makes occupation FE close to firm FE.
- High `occ_delta` makes occupation FE persistent across firm moves.

### 2.6 Outcome

$$Y_{it} = \alpha_i + \psi_{J(i,t)} + \phi_t + \beta \cdot x_{1,it} + \varepsilon_{it}$$

with $\phi_t \sim N(0, \sigma^2_\phi)$ (`var_phi`), $x_1 \sim N(0,1)$,
$\varepsilon_{it} \sim N(0, \sigma^2_\varepsilon)$ (`var_epsilon`), $\beta$ = `beta_x1`.
When occupation is enabled, the occupation extension benchmark adds an
occupation term $\kappa_{O(i,t)}$ with variance `var_occ`.

---

## 3. Parameter Table

| Symbol | Code (`AKMConfig`) | Default | Controls |
|---|---|---|---|
| $N_w$ | `n_workers` | $10^5$ | Number of workers |
| $N_f$ | `n_firms` | $10^4$ | Number of firms |
| $T$ | `n_time` | 10 | Panel length |
| $S$ | `n_industries` | 5 | Number of industries |
| $\sigma^2_\alpha$ | `var_alpha` | 1.0 | Worker effect variance |
| $\sigma^2_\psi$ | `var_psi` | 0.5 | Firm effect variance |
| $\sigma^2_\phi$ | `var_phi` | 0.1 | Time effect variance |
| $\sigma^2_\varepsilon$ | `var_epsilon` | 1.0 | Noise variance |
| $\gamma$ | `gamma` | 1.0 | Firm size Pareto shape |
| $\rho_{\text{size}}$ | `rho_size` | 0.6 | Size-quality rank correlation |
| $\rho$ | `rho` | 1.0 | Sorting (type assortativity) |
| $\delta$ | `delta` | 0.2 | Separation rate per period |
| $\lambda$ | `lambda_` | 0.8 | Within-industry match probability |
| $\beta$ | `beta_x1` | 0.5 | Covariate coefficient |
| $B$ | `n_match_bins` | 64 | Alpha discretization bins |
| $N_o$ | `n_occupations` | 0 | Number of occupations (0 disables occupation) |
| $\sigma^2_\kappa$ | `var_occ` | 0.3 | Occupation effect variance |
| $K_o$ | `occ_menu_size` | 5 | Occupations available per firm |
| $\lambda_o$ | `occ_lambda` | 0.5 | Primary-occupation mass within firm |
| $\delta_o$ | `occ_delta` | 0.3 | Keep old occupation on compatible moves |

Defaults in `dgps.py` → `_AKM_DEFAULTS` for the standard sweep and
`_AKM_OCCUPATION_DEFAULTS` for the occupation extension.

---

## 4. What Drives Solver Performance

| Derived quantity | Determined by | Why it matters |
|---|---|---|
| Mover share $\approx 1-(1-\delta)^{T-1}$ | $\delta$, $T$ | Graph connectivity; identification strength |
| Graph bridge thinness | $\lambda$, $S$, $\delta$ | Near-block-diagonal → slow MAP convergence |
| FE saturation $(N_w + N_f)/N$ | all | Near-saturation → ill-conditioned normal equations |
| Firm size imbalance | $\gamma$ | Unequal leverage → slower convergence |
| Sorting strength | $\rho$ | High sorting → near-collinearity of $\alpha$ and $\psi$ |

Diagnostics computed by `summarize_akm_panel`: mover share, singleton count,
connected components, largest connected set share, cross-industry share, and,
when occupation is enabled, within-firm occupation concentration, realized
occupation change share, compatible move share, and forced occupation change share.

---

## 5. Benchmark Scripts

### `benchmark_main.py`

Basic scaling benchmark. Two DGPs ("simple", "difficult") across 1K–1M obs
with 2-way and 3-way FE specs.

### `benchmark_akm_sweep.py`

Systematic sweep over AKM parameters. Each scenario varies one or two
parameters from defaults, targeting $N = 10^6$ obs (except scale scenarios).
Compares MAP (alternating projections), CG-Schwarz, fixest, and Julia LSMR.

The standard sweep benchmarks the three-way AKM specification
`indiv_id + firm_id + year`.

### `benchmark_akm_occupation.py`

Occupation extension benchmark. Uses occupation-enabled AKM scenarios and
times the occupation-extended specification
`indiv_id + firm_id + year + occ_id`.

---

## 6. AKM Sweep Scenarios

Scenario definitions: `dgps.py` → `_akm_sweep_scenarios`.

The scenarios are organized in four acts that build progressively:

### Act 1: Reference points

| Scenario | Overrides | Purpose |
|---|---|---|
| `akm_baseline` | none (all defaults) | Reference point for all comparisons |
| `akm_easy` | `n_firms=100, delta=0.5, rho=0, n_time=20` | Trivially easy; MAP-favorable, CG overhead is wasted |

### Act 2: Scale

How does runtime scale with $N$, holding structure constant?

| Scenario | $N$ |
|---|---|
| `akm_scale_1` | 10K |
| `akm_scale_2` | 100K |
| `akm_scale_3` | 1M |
| `akm_scale_4` | 10M |

### Act 3: Single-axis sweeps

Each group varies one knob from defaults ($N = 10^6$), isolating its effect.

**Sorting** (vary $\rho$; `n_match_bins=2048` for near-continuous matching, `delta=0.05` so fewer moves create fewer bridges between quality bands, `n_firms=50000` so high $\rho$ creates many fine-grained quality bands)

| Scenario | $\rho$ | Regime |
|---|---|---|
| `akm_sorting_1` | 0 | Random matching |
| `akm_sorting_2` | 5 | Moderate |
| `akm_sorting_3` | 20 | Strong |
| `akm_sorting_4` | 50 | Near-deterministic |
| `akm_sorting_5` | 100 | Extreme |

**Mobility** (vary $\delta$)

| Scenario | $\delta$ | Approx. mover share |
|---|---|---|
| `akm_mobility_1` | 0.5 | 99.8% |
| `akm_mobility_2` | 0.05 | 37% |
| `akm_mobility_3` | 0.01 | 8.6% |
| `akm_mobility_4` | 0.005 | 4.4% |
| `akm_mobility_5` | 0.001 | 0.9% |

### Act 4: Combinations

**Sorting x mobility** (2x2 factorial)

| Scenario | $\rho$ | $\delta$ |
|---|---|---|
| `akm_interaction_1` | 0 | 0.5 |
| `akm_interaction_2` | 20 | 0.5 |
| `akm_interaction_3` | 0 | 0.02 |
| `akm_interaction_4` | 20 | 0.02 |

Runtime of `_4` minus `_2` minus `_3` plus `_1` measures the interaction effect.

### Act 5: Progressive freezing

These scenarios progressively switch off mobility market-by-market:

| Scenario | Frozen markets (of 10) |
|---|---|
| `akm_freeze_1` | 0 |
| `akm_freeze_2` | 2 |
| `akm_freeze_3` | 4 |
| `akm_freeze_4` | 6 |
| `akm_freeze_5` | 8 |
| `akm_freeze_6` | 10 |

### Act 6: Occupation extension

Scenario definitions: `dgps.py` → `_akm_occupation_scenarios`.

- `akm_occlambda_*`: vary only within-firm occupation concentration
- `akm_occsize_*`: vary only occupation dimensionality
