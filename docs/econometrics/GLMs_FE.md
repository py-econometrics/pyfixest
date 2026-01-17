# Estimation of Generalized Linear Models with High-Dimensional Fixed Effects



## A Synthesis of Stammann (2018) and Correia, Guimarães & Zylkin (2019)



This document summarizes the core algorithms for estimating GLMs with high-dimensional k-way fixed effects, synthesizing two foundational papers:



1. **Stammann (2018)**: "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects"

2. **Correia, Guimarães & Zylkin (2019)**: "PPMLHDFE: Fast Poisson Estimation with High-Dimensional Fixed Effects"



---



## 1. Problem Setup



### 1.1 The General GLM Framework



Consider a GLM from the exponential family:



$$f_y(y; \theta, \phi) = \exp\left(\frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi)\right)$$



where:

- $a(\cdot)$, $b(\cdot)$, $c(\cdot)$ are specific functions defining the distribution

- $\phi$ is a dispersion parameter

- $\theta$ is the canonical parameter



Key properties:

- **Mean**: $\mathbb{E}[y] = \mu = b'(\theta)$

- **Variance**: $\text{Var}(y) = b''(\theta) \cdot a(\phi)$



### 1.2 Model with High-Dimensional Fixed Effects



For $n$ independent observations indexed by $i$, we model:



$$\mathbb{E}[y_i] = \mu_i = g^{-1}(\eta_i)$$



where the linear predictor includes k-way fixed effects:



$$\eta_i = \mathbf{x}_i'\boldsymbol{\beta} + \sum_{k=1}^{K} \mathbf{d}_{ik}'\boldsymbol{\alpha}_k$$



Or in matrix form:



$$\boldsymbol{\eta} = \mathbf{X}\boldsymbol{\beta} + \sum_{k=1}^{K} \mathbf{D}_k \boldsymbol{\alpha}_k$$



where:

- $\mathbf{X}$ is the $n \times p$ design matrix of covariates of interest

- $\mathbf{D}_k$ is the $n \times J_k$ dummy matrix for fixed effect category $k$

- $\boldsymbol{\beta}$ are the structural parameters of interest

- $\boldsymbol{\alpha}_k$ are the fixed effect parameters (nuisance parameters)

- $g(\cdot)$ is the link function



### 1.3 Common GLM Families



| Family | Link $g(\mu)$ | Mean $\mu = g^{-1}(\eta)$ | Variance $V(\mu)$ |

|--------|---------------|---------------------------|-------------------|

| Gaussian | $\mu$ (identity) | $\eta$ | $1$ |

| Poisson | $\log(\mu)$ | $\exp(\eta)$ | $\mu$ |

| Binomial (logit) | $\log(\mu/(1-\mu))$ | $\frac{e^\eta}{1+e^\eta}$ | $\mu(1-\mu)$ |

| Binomial (probit) | $\Phi^{-1}(\mu)$ | $\Phi(\eta)$ | $\mu(1-\mu)$ |

| Gamma | $1/\mu$ | $1/\eta$ | $\mu^2$ |

| Negative Binomial | $\log(\mu)$ | $\exp(\eta)$ | $\mu + \mu^2/\theta$ |



---



## 2. The Iteratively Reweighted Least Squares (IRLS) Algorithm



### 2.1 Standard IRLS Update



The MLE for GLMs is typically found via IRLS (equivalent to Fisher Scoring / Newton-Raphson with expected Hessian). The update at iteration $r$ is:



$$\boldsymbol{\beta}^{(r)} = \left(\mathbf{X}'\mathbf{W}^{(r-1)}\mathbf{X}\right)^{-1} \mathbf{X}'\mathbf{W}^{(r-1)} \mathbf{z}^{(r-1)}$$



where:

- **Weight matrix**: $\mathbf{W}^{(r-1)} = \text{diag}\left\{w_i^{(r-1)}\right\}$

- **Working dependent variable**: $\mathbf{z}^{(r-1)}$ (pseudo-response)



### 2.2 IRLS Components for Common Families



The weights and working response for canonical links:



$$w_i = \frac{1}{V(\mu_i) \cdot [g'(\mu_i)]^2}$$



$$z_i = \eta_i + (y_i - \mu_i) \cdot g'(\mu_i)$$



| Family | Weight $w_i$ | Working Response $z_i$ |

|--------|--------------|------------------------|

| **Poisson** | $\mu_i$ | $\eta_i + \frac{y_i - \mu_i}{\mu_i}$ |

| **Logit** | $\mu_i(1-\mu_i)$ | $\eta_i + \frac{y_i - \mu_i}{\mu_i(1-\mu_i)}$ |

| **Probit** | $\frac{\phi(\eta_i)^2}{\Phi(\eta_i)(1-\Phi(\eta_i))}$ | $\eta_i + \frac{y_i - \mu_i}{\phi(\eta_i)} \cdot \frac{1}{\mu_i(1-\mu_i)}$ |

| **Gaussian** | $1$ | $y_i$ |



where $\phi(\cdot)$ is the standard normal PDF and $\Phi(\cdot)$ is the CDF.



---



## 3. The Core Algorithm: Weighted FWL + Method of Alternating Projections



### 3.1 The Computational Challenge



Direct estimation requires computing:



$$(\mathbf{Z}'\mathbf{W}\mathbf{Z})^{-1}$$



where $\mathbf{Z} = [\mathbf{X}, \mathbf{D}_1, \ldots, \mathbf{D}_K]$ may have millions of columns due to the fixed effects dummy matrices. This is memory-prohibitive.



### 3.2 The Frisch-Waugh-Lovell (FWL) Theorem for IRLS



**Key Insight (Stammann 2018, CGZ 2019)**: The FWL theorem extends to weighted least squares. Instead of estimating all parameters jointly, we can:



1. "Partial out" the fixed effects from both $\mathbf{X}$ and $\mathbf{z}$

2. Run weighted regression on the residualized variables



The update equation becomes:



$$\boldsymbol{\delta}^{(r)} = \left(\tilde{\mathbf{X}}'\mathbf{W}^{(r-1)}\tilde{\mathbf{X}}\right)^{-1} \tilde{\mathbf{X}}'\mathbf{W}^{(r-1)} \tilde{\mathbf{z}}^{(r-1)}$$



where $\tilde{\mathbf{X}}$ and $\tilde{\mathbf{z}}$ are **weighted within-transformed** (demeaned) versions.



### 3.3 Weighted Demeaning via Alternating Projections



The within-transformation for each fixed effect category $k$ is:



$$\mathbf{M}_k = \mathbf{I} - \mathbf{D}_k(\mathbf{D}_k'\mathbf{W}\mathbf{D}_k)^{-1}\mathbf{D}_k'\mathbf{W}$$



For a single fixed effect, this reduces to weighted group-mean centering. For $K$ fixed effects, we use the **Method of Alternating Projections (MAP)**:



**Algorithm: Weighted Alternating Projections**



```

Input: variable v, weights W, fixed effect categories D_1, ..., D_K

Output: within-transformed variable ṽ



ṽ ← v

repeat until convergence:

    for k = 1 to K:

        for each level j in category k:

            # Compute weighted group mean

            group_j ← observations where D_k = j

            mean_j ← Σ_{i ∈ group_j} W_i · ṽ_i / Σ_{i ∈ group_j} W_i



            # Subtract weighted group mean

            for i ∈ group_j:

                ṽ_i ← ṽ_i - mean_j

```



This converges to the projection onto the orthogonal complement of the fixed effects space.



### 3.4 The Full HDFE-GLM Algorithm



**Algorithm: GLM with High-Dimensional Fixed Effects**



```

Input: y, X, D_1, ..., D_K, family, tolerance ε

Output: β̂, standard errors



# Initialization

μ^(0) ← (y + ȳ) / 2  # or family-specific initialization

η^(0) ← g(μ^(0))



repeat until ||β^(r) - β^(r-1)|| < ε:

    r ← r + 1



    # 1. Compute IRLS weights and working response

    W^(r-1) ← diag{w_i(μ^(r-1))}  # family-specific weights

    z^(r-1) ← η^(r-1) + (y - μ^(r-1)) · g'(μ^(r-1))



    # 2. Within-transform using weighted alternating projections

    X̃ ← WeightedDemean(X, W^(r-1), D_1, ..., D_K)

    z̃^(r-1) ← WeightedDemean(z^(r-1), W^(r-1), D_1, ..., D_K)



    # 3. Weighted least squares on transformed data

    β^(r) ← (X̃'W^(r-1)X̃)^{-1} X̃'W^(r-1) z̃^(r-1)



    # 4. Compute residuals and update linear predictor

    e^(r) ← z̃^(r-1) - X̃ β^(r)

    η^(r) ← z^(r-1) - e^(r)



    # 5. Update conditional mean

    μ^(r) ← g^{-1}(η^(r))



# Variance estimation (robust/clustered as needed)

V(β̂) ← standard sandwich estimator on transformed data

```



### 3.5 Key Implementation Detail: Residual-Based Update



A crucial insight from both papers: **we do not need to explicitly recover the fixed effect estimates** to continue iteration. The residuals from the within-transformed regression equal the residuals from the full model (FWL property), so:



$$\boldsymbol{\eta}^{(r)} = \mathbf{z}^{(r-1)} - \mathbf{e}^{(r)}$$



where $\mathbf{e}^{(r)}$ are residuals from the weighted regression on demeaned data.



---



## 4. Acceleration Techniques (ppmlhdfe)



CGZ (2019) introduce several acceleration techniques:



### 4.1 Progressive Within-Transformation



Instead of fully within-transforming from scratch each iteration:



1. Store $\tilde{\mathbf{z}}^{(r-1)}$ from previous iteration

2. Update: $\tilde{\mathbf{z}}^{*(r)} = \tilde{\mathbf{z}}^{(r-1)} + \mathbf{z}^{(r)} - \mathbf{z}^{(r-1)}$

3. Use $\tilde{\mathbf{z}}^{*(r)}$ as starting point for alternating projections



**Intuition**: If fixed effects change slowly between iterations (they do), the previous within-transformation provides an excellent warm start.



### 4.2 Adaptive Tolerance Tightening



The inner loop (alternating projections) tolerance can be relaxed early and tightened as the outer loop approaches convergence:



```

if criterion^(r) < 10 × inner_tol:

    inner_tol ← inner_tol / 10

```



**Impact**: CGZ report 50%+ reduction in total alternating projection iterations.



### 4.3 One-Time X Transformation



Since $\mathbf{X}$ doesn't change between iterations, it only needs to be within-transformed **once** in the first iteration. Subsequent iterations reuse $\tilde{\mathbf{X}}$ with progressive updates.



---



## 5. Separation and Existence of MLE



### 5.1 The Separation Problem



For GLMs (especially Poisson and Logit), MLE may not exist when:

- **Poisson**: Perfect collinearity among regressors for the subsample where $y_i > 0$

- **Logit**: Perfect prediction (separation) — some covariate perfectly predicts $y=1$ or $y=0$



### 5.2 Detection and Handling (CGZ 2019)



ppmlhdfe implements multiple methods to detect separation:



1. **FE method**: Check for fixed effects that perfectly predict zeros

2. **Simplex method**: Linear programming to identify separated observations

3. **IR method**: Iteratively check for observations with $\mu_i \to 0$ or $\mu_i \to 1$



**Resolution**: Drop separated observations (they contribute no information to the likelihood), then drop collinear regressors.



---



## 6. Poisson-Specific Details



For Poisson/PPML with log link:



$$\mathbb{E}[y_i] = \mu_i = \exp(\mathbf{x}_i'\boldsymbol{\beta} + \text{fixed effects})$$



**Weights**:

$$W^{(r-1)} = \text{diag}\left\{\exp(\mathbf{x}_i\boldsymbol{\beta}^{(r-1)})\right\} = \text{diag}\{\mu_i^{(r-1)}\}$$



**Working response**:

$$z_i^{(r-1)} = \frac{y_i - \mu_i^{(r-1)}}{\mu_i^{(r-1)}} + \mathbf{x}_i\boldsymbol{\beta}^{(r-1)} = \frac{y_i - \mu_i^{(r-1)}}{\mu_i^{(r-1)}} + \eta_i^{(r-1)}$$



---



## 7. Logit-Specific Details (for PyFixest Implementation)



For binomial logit with logit link:



$$\mathbb{E}[y_i] = \mu_i = \frac{\exp(\eta_i)}{1 + \exp(\eta_i)} = \Lambda(\eta_i)$$



where $\Lambda(\cdot)$ is the logistic CDF.



**Weights**:

$$W^{(r-1)} = \text{diag}\left\{\mu_i^{(r-1)}(1-\mu_i^{(r-1)})\right\}$$



**Working response**:

$$z_i^{(r-1)} = \eta_i^{(r-1)} + \frac{y_i - \mu_i^{(r-1)}}{\mu_i^{(r-1)}(1-\mu_i^{(r-1)})}$$



**Separation detection**: Critical for logit! Use linear programming or iterative checking for observations where $\mu_i \to 0$ or $\mu_i \to 1$.



---



## 8. Variance Estimation



### 8.1 Standard Errors



The FWL theorem guarantees that standard errors from the within-transformed regression are correct for $\boldsymbol{\beta}$. Robust and clustered standard errors follow standard sandwich formulas on the transformed data.



### 8.2 Degrees of Freedom



Careful accounting is needed:

- Total parameters = $p + \sum_k J_k - \text{redundant FE}$

- Redundant FE arise from: singletons, collinearity between FE categories



---

## References



1. Stammann, A. (2018). "Fast and Feasible Estimation of Generalized Linear Models with High-Dimensional k-way Fixed Effects." arXiv:1707.01815.



2. Correia, S., Guimarães, P., & Zylkin, T. (2019). "PPMLHDFE: Fast Poisson Estimation with High-Dimensional Fixed Effects." arXiv:1903.01690.



3. Guimarães, P., & Portugal, P. (2010). "A Simple Feasible Procedure to Fit Models with High-Dimensional Fixed Effects." Stata Journal, 10(4), 628-649.



4. Gaure, S. (2013). "OLS with Multiple High Dimensional Category Variables." Computational Statistics & Data Analysis, 66, 8-18.



5. Correia, S., Guimarães, P., & Zylkin, T. (2019). "Verifying the Existence of Maximum Likelihood Estimates for Generalized Linear Models." Working paper.



---



## Appendix: Family-Specific Formulas



### A.1 Gaussian (Identity Link)

- $\mu_i = \eta_i$

- $w_i = 1$

- $z_i = y_i$

- Reduces to standard OLS with fixed effects



### A.2 Poisson (Log Link)

- $\mu_i = \exp(\eta_i)$

- $w_i = \mu_i$

- $z_i = \eta_i + (y_i - \mu_i)/\mu_i$



### A.3 Binomial Logit (Logit Link)

- $\mu_i = 1/(1 + \exp(-\eta_i))$

- $w_i = \mu_i(1-\mu_i)$

- $z_i = \eta_i + (y_i - \mu_i)/[\mu_i(1-\mu_i)]$



### A.4 Binomial Probit (Probit Link)

- $\mu_i = \Phi(\eta_i)$

- $w_i = \phi(\eta_i)^2/[\Phi(\eta_i)(1-\Phi(\eta_i))]$

- $z_i = \eta_i + (y_i - \mu_i)\cdot[\Phi(\eta_i)(1-\Phi(\eta_i))]/\phi(\eta_i)$



### A.5 Negative Binomial (Log Link)

- $\mu_i = \exp(\eta_i)$

- $w_i = \mu_i/[1 + \mu_i/\theta]$ where $\theta$ is the dispersion

- $z_i = \eta_i + (y_i - \mu_i)/\mu_i$
