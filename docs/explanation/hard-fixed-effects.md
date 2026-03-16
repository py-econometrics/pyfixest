# "When Are Fixed Effects Estimations Hard?"

If you have ever fitted a fixed regression model, then you might have noticed that fixed effects regressions with the same number of observations and fixed effects levels can take orders of magnitudes to run. The runtime of a fixed effects problem is not only determined by the sheer size of the data, but of properties of the fixed effects. Problems that are known to be particularly "hard" are ubiqitous in economics, and arise for example in matched employer-employee data, patient-doctor panels, or trade networks.

In this guide, we explain *why* some fixed effects problems are harder to estimate than others, and benchmark different strategies to fit fixed effects regressions in a range of scenarios. 

The key insight is that
fixed-effects estimation is a **graph problem**: the structure of who-works-where (or who-sees-which-doctor which-brand-in-which-store) determines how hard the problem is. After reading this guide, you should have a good idea if you can speed up your own fixed effects problem by choosing a different solver. 

## Fixed Effects as a Network

As our workhorse example throughout, we will consider the Abowd-Kramarz-Margolis (AKM) wage model with worker fixed effects $\alpha_i$, firm fixed effects $\psi_{J(i,t)}$ and time fixed effects $\phi_t$.

$$y_{it} = \alpha_i + \psi_{J(i,t)} + \phi_t + x'_{it}\beta + \varepsilon_{it}$$

Workers and firms form a **bipartite graph**: workers are one set of nodes, firms
are the other, and each employment spell is an edge. Movers - workers who change
firms - are the edges that connect different parts of the graph. Without movers,
worker and firm effects are not separately identified.

![](figures/akm-benchmarks/bipartite_graph.png)

*A dense graph (left) has many movers connecting all firms, making worker and firm effects easy to separate. A sparse graph (right) has a single mover bridging two clusters - demeaning must propagate information through that thin bridge, which is slow.*

This bipartite structure is ubiqituous in applied economics. In AKM
wage decompositions, workers and firms are the two sides of the graph,
and job changers are the movers that connect them. The same pattern
arises in mover designs, where families / students move across schools or neighborhoods. In health economics, we have problems of similar structure with doctor-patient fixed effects. And in trade and industrial organisation,
products sold across multiple markets or brands stocked in different
stores play the role of movers.

In all these settings, estimation requires solving the same underlying
linear algebra problem, which we introduce in the following section.

## From FWL to Demeaning

Before we dive into algorithmic strategies, we first want to (re-) introduce the **Frisch-Waugh-Lovell (FWL) theorem**: in a
regression of $y$ on covariates $X$ and a set of dummy variables $D$
(the fixed effects), the coefficient $\hat{\beta}$ on $X$ is identical
whether we estimate the full model or first project both $y$ and $X$
onto the orthogonal complement of $D$'s column space in two separate regressions and then regress the two resulting residuals on each other.

Fitting the two initial regression and forming a residual is often referred to as a **demeaning** step. 

For a single factor (e.g., only worker FEs), this demeaning step is trivial -
we can simply subtract the average wage of each worker from $y$ and $X$ and we are done. 

For two-way FEs in balanced panels, closed-form solutions exist (e.g., the Mundlak
approach), but as soon as panels are unbalanced - which is the norm in
matched employer-employee data and most real-world applications -
these methods break down and we need specialised algorithms/solvers. 


## Algorithms for the FWL Demeaning Step

Several algorithms have been proposed for this multi-factor demeaning
problem:

- **Method of Alternating Projections (MAP).** Introduced by
  Guimarães & Portugal (2010) as the "Zig-Zag" and Gaure (2013), this is the workhorse algorithm in most FE packages (`reghdfe`, `lfe`, `fixest`). It sequentially sweeps through each factor, demeaning the target variable by the current factor's group means. Usually, this approach is implemented with accelerations. For example, R's `fixest` uses MAP with Irons-Tuck acceleration and other convergence tricks. In PyFixest, the `"rust"` backend implements MAP without acceleration. One key advantage of the MAP algorithm is that the fixed effects do not have to be encoded as a (sparse) one-hot encoded design matrix.

- **LSMR.** Julia's `FixedEffectModels.jl` uses LSMR. tba.

- **CG-Schwarz (Conjugate Gradients with Additive Schwarz Preconditioner).**
  The [`within`](https://github.com/py-econometrics/within) crate, used by
  PyFixest's `"rust-cg"` backend, takes a different approach: it
  explicitly builds and exploits the block structure of the normal
  equations. We explain this structure below.

## The Normal Equations and Their Block Structure

The FWL projection - removing fixed effects from the data - amounts to
solving a linear system. Specifically, we must solve the **normal
equations**:

$$G \, \hat{\mu} = D^\top y$$

where $D$ is the $n \times m$ dummy matrix that encodes all FE levels
and $G = D^\top D$ is the **Gramian** - a symmetric positive
semi-definite matrix of dimension $m \times m$, where $m$ is the total
number of FE levels across all factors. 

The Gramian has a natural **block structure**. To illustrate this, we will consider a small example (adapted from the
[`within` documentation](https://github.com/py-econometrics/within)) of
a worker-firm panel with $n = 6$ observations and $Q = 3$ factors
(worker, firm, year). Worker W1 moves from Firm F1 to F2 - this
mobility is what connects the two firms in the estimation graph.
Workers W2 (at F1) and W3 (at F2) stay at their firms.

| Obs | Worker | Firm | Year | $y$ |
|-----|----------------|--------------|--------------|------|
| 1 | W1 | F1 | Y1 | 3.2 |
| 2 | W1 | F2 | Y2 | 4.1 |
| 3 | W2 | F1 | Y1 | 2.8 |
| 4 | W2 | F1 | Y2 | 3.9 |
| 5 | W3 | F2 | Y1 | 5.0 |
| 6 | W3 | F2 | Y2 | 4.5 |

Factor 1 (workers) has $m_1 = 3$ levels, factor 2 (firms) has $m_2 = 2$
levels, factor 3 (years) has $m_3 = 2$ levels, giving $m = 7$ total FE
levels. The Gramian has $Q = 3$ **diagonal blocks** and
$\binom{3}{2} = 3$ **cross-tabulation blocks**:

$$
G = \begin{pmatrix}
{\color{royalblue}G_{WW}} & {\color{gray}G_{WF}} & {\color{gray}G_{WY}} \\
{\color{gray}G_{WF}^\top} & {\color{crimson}G_{FF}} & {\color{gray}G_{FY}} \\
{\color{gray}G_{WY}^\top} & {\color{gray}G_{FY}^\top} & {\color{forestgreen}G_{YY}}
\end{pmatrix}
= \left(\begin{array}{ccc|cc|cc}
{\color{royalblue}2} & {\color{royalblue}0} & {\color{royalblue}0} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} \\
{\color{royalblue}0} & {\color{royalblue}2} & {\color{royalblue}0} & {\color{gray}2} & {\color{gray}0} & {\color{gray}1} & {\color{gray}1} \\
{\color{royalblue}0} & {\color{royalblue}0} & {\color{royalblue}2} & {\color{gray}0} & {\color{gray}2} & {\color{gray}1} & {\color{gray}1} \\
\hline
{\color{gray}1} & {\color{gray}2} & {\color{gray}0} & {\color{crimson}3} & {\color{crimson}0} & {\color{gray}2} & {\color{gray}1} \\
{\color{gray}1} & {\color{gray}0} & {\color{gray}2} & {\color{crimson}0} & {\color{crimson}3} & {\color{gray}1} & {\color{gray}2} \\
\hline
{\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}2} & {\color{gray}1} & {\color{forestgreen}3} & {\color{forestgreen}0} \\
{\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}1} & {\color{gray}2} & {\color{forestgreen}0} & {\color{forestgreen}3}
\end{array}\right)
$$

The **diagonal blocks** ${\color{royalblue}G_{WW}}$,
${\color{crimson}G_{FF}}$, ${\color{forestgreen}G_{YY}}$ are each
diagonal matrices whose entries are the group counts (how many
observations belong to each worker, firm, or year). We note that inverting these
blocks is computationally cheap because it amounts to dividing by group sizes, i.e.,
computing group means.

The **cross-tabulation blocks** ${\color{gray}G_{WF}}$,
${\color{gray}G_{WY}}$, ${\color{gray}G_{FY}}$ encode the
bipartite graph structure. For example, $G_{WF} = D_W^\top D_F$ is the
worker-firm cross-tabulation: entry $(i, j)$ counts how many times
worker $i$ is observed at firm $j$. This is where the mover information
lives. Worker W1's row in $G_{WF}$ is $(1, 1)$ while W2's row is $(2, 0)$. In other words, W1 works in both firms, while W2 workins in F2 in both period. In a labour market with little mobility, the off-diagonal blocks will be sparse as most
workers stay within a single firm.

## The Method of Alternating Projections

Recall that we need to solve $G \hat{\mu} = D^\top y$ for the FE
coefficients $\hat{\mu}$, or equivalently, find the residual
$r = y - D \hat{\mu}$ that has all fixed effects projected out. The method of alternating projections (MAP) approaches this iteratively: it sweeps through each factor and subtracts
group means from the current residual. In terms of the Gramian, this is
**block Gauss-Seidel** - each sweep solves one diagonal block of $G$ at
a time. Writing $D_W, D_F, D_Y$ for the $n \times m_q$ dummy
sub-matrices (column blocks of $D$), the steps are:

1. Start with $r = y$
2. Subtract worker means from $r$: $r \leftarrow r - D_W {\color{royalblue}G_{WW}^{-1}} D_W^\top r$ 
3. Subtract firm means from $r$: $r \leftarrow r - D_F {\color{crimson}G_{FF}^{-1}} D_F^\top r$ 
4. Subtract year means from $r$: $r \leftarrow r - D_Y {\color{forestgreen}G_{YY}^{-1}} D_Y^\top r$ 
5. Repeat steps 2-4 until convergence

Each of these steps is individually cheap because $G_{WW}$, $G_{FF}$, $G_{YY}$. are diagonal matrices. 

We also note that the MAP algorithm **never directly touches the cross-tabulation blocks** $G_{WF}$, $G_{FY}$, $G_{WY}$. It can only extract information about the
relationship between workers and firms *indirectly*, through the
residuals that get passed from one sweep to the next.

When the graph structure of workers and firms is dense (many movers), each sweep of the MAP algorithm makes good progress. If workers move frequently between firms, high-ability workers are observed at both good and bad firms, producing different outcomes in
each. The algorithm can easily discriminate the worker effect from
the firm effect, because the same worker provides a direct
comparison across firms. Subtracting worker means already removes most
of the worker effect, and what remains is a clean signal about firms.

When the graph is sparse (few movers), workers are essentially
**nested** within firms - nearly every worker is observed at a single
firm only. In this regime, a high outcome could be due to a good worker
*or* a good firm, and the data provides almost no variation to tell
them apart. Subtracting worker means barely helps, because each worker's
mean is contaminated by the firm effect they are stuck in. The algorithm
must iterate many times, slowly propagating the little cross-factor
information that exists through the few movers that bridge different
firms.

## How CG-Schwarz Works (and Why It Uses the Graph)

The [`within`](https://github.com/py-econometrics/within) crate takes
the opposite approach: it works with the **full Gramian** $G$, including
the cross-tabulation blocks, and solves the normal equations using
**preconditioned conjugate gradients (CG)**.

Because CG directly works with the cross-tabulation structure, it can
propagate information across the graph in a single matrix-vector
multiply. Its convergence rate depends on the **condition number** of
the preconditioned system, not the Friedrichs angle. This makes it much
more robust to sparse graphs.

## When Does Each Solver Win?

Here is some first-order intuition: 
- **MAP wins on dense graphs.** When the graph is well-connected with
  many movers and no fragmentation, the vanilla MAP algorithm converges in a handful of iterations.
  Each sweep is extremely cheap because it only computes group means, so
  the total cost is low. The CG algorithm in stead has overhead from forming the preconditioner, which might not amortize for dense graphs. 

- **CG-Schwarz wins on sparse graphs.** When the graph has few movers,
  MAP's convergence stalls because it cannot propagate information across
  thin bridges. CG uses the cross-tabulation blocks directly, so it does
  not suffer from this bottleneck. The overhead of building $G$ is repaid
  many times over by needing far fewer iterations.

The intuition above is deliberately simplified. In practice, fixed-point accelerations
such as Irons-Tuck (used in R's `fixest`) can significantly speed up
MAP convergence, narrowing the gap on moderately sparse graphs. This is
why the benchmarks below compare four backends: 
- `pyfixest (rust-map)` (vanilla MAP without acceleration),
-  `fixest-map` (R's `fixest` with Irons-Tuck
acceleration, non-vanilla MAP)
- `pyfixest (rust-cg)` (CG-Schwarz via `within`)
- `FEM.jl` (Julia's `FixedEffectModels.jl` via LSMR). 

All benchmarks time the full estimation function call (e.g., `fixest::feols()` or
`pf.feols()`). We want to stress that no benchmark is perfect - the different packages all run different pre-processing routines, e.g. for dropping singletons or multicollinear variables. In addition, they use slightly different convergence criteria. 

The benchmark scripts and DGP documentation live in
[`benchmarks/modular/`](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks/modular). You should be able to reproduce all results by running `pixi r benchmark`, `pixi r benchmark-akm` and `pixi run benchmark-akm-occupation`. 

## Well-Connected Graphs

Before looking at problems with sparse connections, it is worth confirming the
baseline: when the bipartite graph is dense, the MAP algorithm converges really quickly. 

In this initial scenario, we simulate 100 firms, a separation rate of $\delta = 0.5$ (workers switch firms every other period on average), no sorting of workers to firms ($\rho = 0$), and a 20-period panel, which produces a deliberately well-connected graph with ~1M observations.

On this easy problem, the standard worker + firm + year specification is
solved in well under a second by all backends. MAP converges in very few
sweeps because each sweep already removes most of the variation, and
CG-Schwarz's overhead from building the Gramian does not pay off.

## Benchmark Parametrization

We now introduce our baseline benchmark parametrization. All subsequent
sweeps start from these defaults and vary one parameter at a time.

| Parameter | Symbol | Default | Meaning |
|:----------|:------:|--------:|:--------|
| Panel length | $T$ | 10 | Number of time periods each worker is observed |
| Number of firms | $m_F$ | 10,000 | Total firms in the economy |
| Separation rate | $\delta$ | 0.20 | Per-period probability that a worker switches firms |
| Sorting | $\rho$ | 1.0 | Correlation between worker ability and firm quality in the matching process ($\rho = 0$: random, high $\rho$: strong assortative matching) |
| Within-industry match prob. | $\lambda$ | 0.80 | Probability that a mover's next firm is in the same industry ($\lambda = 1$: no cross-industry moves) |
| Number of industries | $S$ | 5 | Number of industry clusters |
| Firm size shape | $\gamma$ | 1.0 | Pareto shape parameter for the firm size distribution (high $\gamma$: equal sizes, low $\gamma$: extreme dispersion) |

With these defaults, the economy has ~1M worker-year observations,
a 20% per-period mover probability, moderate sorting, and a
well-connected graph, which forms a relatively "easy" baseline.

## Scaling with Dataset Size

The first question is how runtimes grow with the number of observations.
We hold the graph structure fixed at the defaults and increase the number of workers $N$ from 10K to 10M.

![](figures/akm-benchmarks/bench_scale.png)

*Benchmark: scale sweep. Runtime as a function of dataset size on a well-connected graph with default parameters.*

On a well-connected graph, all solvers scale roughly linearly. At 1M
observations, `fixest` and `rust-cg` finish in under half a second,
while `rust-map` (MAP without acceleration) takes about 2 seconds.
The absolute differences remain moderate because the graph structure is
benign - MAP converges quickly even without acceleration. The gap
between vanilla MAP and other solvers only blows up when the graph becomes sparse, as shown in the sections below.

## Complex Fixed Effects Structures

All benchmarks below hold the estimation dataset at ~1M observations and vary
one structural parameter at a time from the defaults, isolating the effect of
each graph property on solver performance.


### (a) Low worker mobility $\delta$

The separation rate $\delta$ controls how likely a worker is to change
firms in each period. With $\delta = 0.5$, a worker has a 50% chance of
switching firms every period, so after a 10-period panel nearly everyone
has moved at least once. With $\delta = 0.01$, workers separate only 1%
of the time, so the expected tenure at a firm is 100 periods, which is far
longer than the panel itself. In a 10-period panel with $\delta = 0.01$,
only about 9% of workers are ever observed at more than one firm.

This matters because movers are the only source of information that
lets the algorithm distinguish worker effects from firm effects. When
$\delta$ is high, the bipartite graph is dense with edges and every
firm is connected to many others through shared workers. When $\delta$
is low, most workers sit at a single firm, and the graph thins out to
a near-nested structure where worker and firm effects are almost
collinear.

![](figures/akm-benchmarks/bench_mobility.png)

*Benchmark: mobility sweep. MAP-based solvers (rust-map, fixest) degrade sharply as mobility decreases, while CG-Schwarz (rust-cg) remains stable.*

### (b) Progressive freezing

The uniform mobility sweep above turns one global knob. But real labour
markets are not uniformly sparse — some sectors are fluid while others
are ossified. The progressive-freezing benchmark captures this by
simulating 10 industry markets and progressively switching off
mobility in 2-market increments. Active markets keep the baseline
separation rate ($\delta = 0.2$), while frozen markets drop to
$\delta = 0.005$.

![](figures/akm-benchmarks/bench_freeze.png)

*Benchmark: progressive freezing. As more of the 10 industry markets
are frozen (delta drops from 0.2 to 0.005), MAP runtimes rise
progressively. CG-Schwarz remains stable throughout.*

Even a few frozen markets degrade MAP noticeably, because the frozen
industries contribute stayers that dilute the cross-factor information
available per sweep. The effect is cumulative: each additional pair of
frozen markets adds another near-disconnected block to the Gramian,
and MAP must propagate information through increasingly thin bridges
between the remaining active industries.

### (c) Strong assortative matching $\rho$

The sorting parameter $\rho$ controls how strongly worker ability
$\alpha_i$ and firm quality $\psi_j$ are correlated in the matching
process. When $\rho = 0$, workers are randomly assigned to firms
regardless of type. When $\rho$ is large, high-ability workers
systematically sort into high-quality firms and low-ability workers end
up at low-quality firms.

Strong sorting changes the structure of the bipartite graph even if the
total number of movers stays the same. With random matching, movers
create edges that criss-cross the entire graph, connecting firms of all
quality levels. With strong sorting, movers tend to switch between firms
of similar quality, so the graph develops a near-block-diagonal
structure where each "quality band" becomes an almost self-contained
subgraph. The cross-tabulation blocks become sparse because there are
few edges between quality bands, making worker and firm effect columns
nearly collinear.

![](figures/akm-benchmarks/bench_sorting.png)

*Benchmark: sorting sweep. Increasing sorting ($\rho$) inflates MAP runtime, with wide variance indicating unstable convergence.*


### (d) Multiple sources of difficulty

The single-axis benchmarks above vary one parameter at a time from the
baseline. In practice, real datasets often combine multiple "sources of
difficulty" simultaneously. The sorting $\times$ mobility factorial
benchmark tests this by crossing two levels of sorting ($\rho \in
\{0, 20\}$) with two levels of mobility ($\delta \in \{0.5, 0.02\}$).

![](figures/akm-benchmarks/bench_interaction.png)

*Benchmark: sorting x mobility interaction. The combination of high sorting and low mobility produces super-additive slowdowns - far worse than either factor alone.*

The combination of $\rho = 20$ and $\delta = 0.02$ produces runtimes far
exceeding what either factor alone would predict. Sorting thins out the
bridges between quality bands by making movers switch only between
similar firms, while low mobility means there are few movers to begin
with. Together, they produce an extremely sparse graph where MAP has
almost no cross-factor information to work with.

The general principle is that **graph connectivity is necessary and
sufficient.** Every other axis matters only insofar as it reduces the
effective connectivity of the bipartite graph.


## Adding a Third Nesting Structure: Occupation Fixed Effects

The benchmarks above all involve the standard worker + firm + year
specification - essentially a bipartite graph between workers and firms
(year is low-dimensional and trivially absorbed). But many empirical
applications add a third high-dimensional factor. In AKM wage
regressions, occupation is the natural candidate: researchers want to
separate how much of a worker's wage comes from *who they are*
(worker FE), *where they work* (firm FE), and *what they do*
(occupation FE).

Adding occupation turns the bipartite graph into a **tripartite**
structure. The new factor can be "easy" or "hard" depending on how it
relates to the existing factors. If occupation cross-cuts firms and
workers - different occupations appear at each firm, and workers switch
occupations when they move - then the occupation dimension adds
independent variation and is cheap to absorb. But if occupation is
*nested* within one of the existing factors, the new fixed effects
become nearly collinear with an existing set, and the problem gets
harder for the same reasons we saw above: the effective graph loses
edges.

There are two ways occupation can nest:

- **Firm nesting:** Each firm concentrates on a single occupation
  (think a law firm where everyone is a lawyer). Then knowing the firm
  almost perfectly predicts the occupation, and the occupation FE
  column is nearly a copy of the firm FE column.
- **Worker nesting:** Workers carry their occupation across job changes
  (a nurse remains a nurse regardless of which hospital they join).
  Then the occupation FE column is nearly a copy of the worker FE
  column.

In either case, the solver must separate two nearly identical columns -
exactly the collinearity problem that makes MAP slow.

### The occupation DGP

The occupation extension keeps the standard worker + firm + year AKM
panel and adds `occ_id` as a fourth fixed effect. Each firm draws a
sparse menu of occupations from an industry-level pool, with one
primary occupation. Workers draw their initial occupation from that
menu. On a firm move, workers keep their old occupation only if the
destination firm supports it; otherwise they switch.

![](figures/akm-benchmarks/tripartite_occ_graph.png)

*Tripartite view of the occupation extension. In the cross-cutting case (left), occupation links connect workers and firms in multiple directions - the graph stays well mixed. High `occ_delta` (centre) makes workers carry the same occupation across firm moves. High `occ_lambda` (right) makes each firm concentrate on one primary occupation.*

Three parameters control the occupation structure:

| Parameter | Symbol | Default | Meaning |
|:----------|:------:|--------:|:--------|
| Firm–occ concentration | `occ_lambda` | 0.50 | Share of a firm's workers in the primary occupation ($\approx 1$: occupation is a deterministic function of firm) |
| Occupation persistence | `occ_delta` | 0.30 | Probability a mover keeps the same occupation at the new firm ($= 1$: occupation travels with the worker) |
| Number of occupations | `n_occupations` | 200 | Total occupation levels in the economy |

Each firm also has a menu size of 5 occupations. These defaults give a
moderately cross-cutting baseline on top of the standard three-way AKM
panel (~1M observations).

We also swept `occ_delta` (occupation persistence) but found no measurable effect on runtimes — persistence reshuffles labels without thinning the graph.

### (a) Firm–occupation nesting (`occ_lambda`)

The parameter `occ_lambda` controls how concentrated each firm's
occupation menu is. At `occ_lambda = 0.01`, a firm's workers are
spread roughly evenly across the menu occupations. At
`occ_lambda = 1.0`, virtually all workers at a given firm hold the
firm's primary occupation, so occupation becomes an almost
deterministic function of firm identity.

![](figures/akm-occupation-benchmarks/bench_occlambda.png)

*Benchmark: firm–occupation concentration sweep. This family varies only `occ_lambda`, making occupations progressively more nested within firms.*

### (b) Occupation dimensionality (`n_occupations`)

Increasing the number of occupation levels makes the problem harder
through two channels. First, the Gramian grows: each new occupation
adds a row and column that every iteration must touch, so the per-iteration
cost rises with $N_o$. Second, the occupation sub-graph thins out.
With 10 occupations, each level is observed at many firms and held by
many workers, so the occupation cross-tabulation blocks are dense and
well-conditioned. With 5,000 occupations, each level appears in far
fewer observations, and the cross-tabulation blocks become sparse —
the same graph-thinning mechanism that makes low mobility hard, but
now in the occupation dimension. The sweep varies `n_occupations` from
10 to 5,000 while holding all other parameters at their defaults.

![](figures/akm-occupation-benchmarks/bench_occsize.png)

*Benchmark: occupation dimensionality sweep. This family varies only the number of occupation levels from 10 to 5,000.*


## Conclusion

Graph connectivity is the fundamental driver of fixed-effects solver
performance. Among the individual axes we tested, **mobility** ($\delta$)
has the largest effect: reducing the separation rate from 0.5 to 0.001
can increase MAP runtimes by orders of magnitude while leaving CG-Schwarz
largely unaffected. **Sorting** ($\rho$) operates through a distinct
mechanism — movers still exist but are wasted within quality bands — and
the two interact **super-additively**: the combination of high sorting
and low mobility is far worse than either alone.

The **progressive-freezing** benchmark confirms that the effect is local:
even a few frozen industry markets degrade MAP performance, and each
additional frozen pair compounds the slowdown. This mirrors real labour
markets where some sectors are fluid and others ossified.

When a fourth high-dimensional factor is added (occupations), the main
hardness channel is **firm–occupation nesting** (`occ_lambda`): as
occupation becomes a near-deterministic function of firm identity, the
new FE column is almost collinear with the existing firm FE, recreating
the sparse-graph problem in a higher-dimensional space. Occupation
**dimensionality** (`n_occupations`) adds cost through sheer scale, while
occupation **persistence** (`occ_delta`) has negligible impact on
runtimes.

The practical recommendation is straightforward: for well-connected
graphs (high mobility, low sorting, cross-cutting factors), MAP with
acceleration (e.g., `fixest`) is fast and hard to beat. For sparse
graphs — low mobility, strong sorting, nested structures, or any
combination thereof — CG-Schwarz (`rust-cg` via the `within` crate)
is the more robust choice.