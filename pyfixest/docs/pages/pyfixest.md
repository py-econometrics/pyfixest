<!-- Generated from docs/pyfixest.md; do not edit. -->

# PyFixest: Fast High-Dimensional Fixed Effects Regression in Python

[*Image omitted: License*](https://opensource.org/license/mit)
*Image omitted: Python Versions*
[*Image omitted: PyPI Version*](https://pypi.org/project/pyfixest/)
[*Image omitted: Coverage*](https://codecov.io/gh/py-econometrics/pyfixest)
[*Image omitted: Downloads*](https://pepy.tech/project/pyfixest)
[*Image omitted: Downloads*](https://pepy.tech/project/pyfixest)

*Image omitted: Project Chat*

[chat-badge]: https://img.shields.io/discord/1259933360726216754.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2&style=flat-square
[chat-url]: https://discord.gg/gBAydeDMVK

[Docs](pyfixest.md) · [Quickstart](quickstart.md) · [Function & API Reference](https://pyfixest.org/reference/) · [DeepWiki](https://deepwiki.com/py-econometrics/pyfixest) · [Benchmarks](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks) · [Contributing](contributing.md) · [Changelog](changelog.md)

`PyFixest` is a Python package for fast high-dimensional fixed effects regression.

The package aims to mimic the syntax and functionality of [Laurent Bergé's](https://sites.google.com/site/laurentrberge/) formidable [fixest](https://github.com/lrberge/fixest) package as closely as Python allows. If you know `fixest` well, the goal is that you won't have to read the docs to get started! In particular, this means that all of `fixest's` defaults are mirrored by `PyFixest`.

For questions on `PyFixest`, head over to our [GitHub discussions](https://github.com/py-econometrics/pyfixest/discussions), or join our [Discord server](https://discord.gg/gBAydeDMVK).

## Features

- **Estimation**
  - **OLS**, **WLS**, **IV**, and **GLMs** (Poisson, logit, probit, gaussian) with high-dimensional fixed effects
  - Different **demeaning backends** (MAP, [within](https://github.com/py-econometrics/within) LSMR, torch LSMR) on CPU and GPU
  - Fast **quantile regression** via an interior-point solver
  - **Difference-in-differences** estimators, including TWFE, `Did2s`, local projections, and Sun-Abraham event studies
  - Regression **decomposition** following [Gelbach (2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1425737)
  - Multiple estimation syntax
- **Inference**
  - Several **robust**, **cluster-robust**, and **HAC variance-covariance** estimators
  - **Wild cluster bootstrap** inference via [wildboottest](https://github.com/py-econometrics/wildboottest)
  - **Multiple hypothesis corrections** and simultaneous confidence intervals
  - Fast **randomization inference**
  - The **causal cluster variance estimator (CCV)**
- **Post-Estimation & Reporting**
  - **Publication-ready tables** with [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html) or LaTeX booktabs via the [maketables library](https://github.com/py-econometrics/maketables)

## Installation

You can install the release version from `PyPI` by running

```bash
# inside an active virtual environment
python -m pip install pyfixest
```

or the development version from github by running

```bash
python -m pip install git+https://github.com/py-econometrics/pyfixest
```

<details>
<summary>Optional dependencies</summary>

For visualization features using the `lets-plot` backend, install:

```bash
python -m pip install pyfixest[plots]
```

`matplotlib` is included by default, so plotting works without this extra.

To run the LSMR demeaner via `PyTorch` (CPU and GPU), you need to install `PyTorch`, which you can do via

```bash
python -m pip install pyfixest[torch]
```

For GPU acceleration on CUDA, you additionally need to install a CUDA-enabled torch build. See the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for details.

Then use the typed `demeaner` API:

```python
# CPU
pf.feols(
    "Y ~ X1 | f1 + f2",
    data=data,
    demeaner=pf.LsmrDemeaner(backend="torch", device="cpu"),
)

# CUDA GPU
pf.feols(
    "Y ~ X1 | f1 + f2",
    data=data,
    demeaner=pf.LsmrDemeaner(backend="torch", device="cuda"),
)
```

</details>

## Quickstart

```python
import pyfixest as pf

data = pf.get_data()
pf.feols("Y ~ X1 | f1 + f2", data=data).summary()
```

    ###

    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997

    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5% |   97.5% |
    |:--------------|-----------:|-------------:|----------:|-----------:|-------:|--------:|
    | X1            |     -0.919 |        0.065 |   -14.057 |      0.000 | -1.053 |  -0.786 |
    ---
    RMSE: 1.441   R2: 0.609   R2 Within: 0.2

`PyFixest` also supports multiple estimation syntax:

```python
fit = pf.feols("Y + Y2 ~ X1 | csw0(f1, f2)", data=data, vcov={"CRV1": "group_id"})
fit.etable()
```

For more examples, see the [quickstart](quickstart.md), the [formula syntax tutorial](https://pyfixest.org/formula-syntax.html), and the [Poisson & GLMs tutorial](https://pyfixest.org/poisson-glm.html).

## Benchmarks

The DGPs follow the "simple" and "difficult" designs from the [fixest benchmarks](https://github.com/kylebutts/fixest_benchmarks). The figure timings for regressions with `k=10` covariates and plots the median runtime across three runs for PyFixest MAP, PyFixest within, PyFixest torch on CUDA GPU, fixest, and FixedEffectModels.jl.

*Image omitted.*

To reproduce the benchmarks, run the modular benchmark script:

```bash
python benchmarks/modular/benchmark_main.py
```

For the full benchmark suite, see the [`benchmarks/`](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks) directory and the note on [difficult fixed effects problems](https://github.com/py-econometrics/pyfixest/blob/master/docs/explanation/difficult-fixed-effects.md).

## Learn More

- [Quickstart](quickstart.md)
- [Function & API Reference](https://pyfixest.org/reference/)
- [Difference-in-Differences](https://pyfixest.org/difference-in-differences.html)
- [Quantile Regression](https://pyfixest.org/quantile-regression.html)
- [Changelog](changelog.md)
- [Contributing](contributing.md)

## Acknowledgements

First and foremost, we want to acknowledge [Laurent Bergé's](https://sites.google.com/site/laurentrberge/) formidable [fixest](https://github.com/lrberge/fixest), which [is so good we decided to stick to its API and conventions](https://youtu.be/kSQxGGA7Rr4?si=8-wTbzLPnIZQ7lYI&t=576) as closely as Python allows. Without `fixest`, `PyFixest` likely wouldn't exist - or at the very least, it would look very different.

For a full list of software packages and papers that have influenced PyFixest, please take a look at the [Acknowledgements page](acknowledgements.md).

We thank all institutions that have funded or supported work on PyFixest!

*Image omitted.*

## How to Cite

If you want to cite PyFixest, you can use the following BibTeX entry:

```bibtex
@software{pyfixest,
  author  = {{The PyFixest Authors}},
  title   = {{pyfixest: Fast high-dimensional fixed effect estimation in Python}},
  year    = {2025},
  url     = {https://github.com/py-econometrics/pyfixest}
}
```

## Support PyFixest

If you enjoy using `PyFixest`, please consider donating to [GiveDirectly](https://donate.givedirectly.org/dedicate) and dedicating your donation to `pyfixest.dev@gmail.com`.
You can also leave a message through the donation form; your support and encouragement mean a lot to the developers.

## Call for Contributions

Thanks for showing interest in contributing to `pyfixest`! We appreciate all
contributions and constructive feedback, whether that be reporting bugs, requesting
new features, or suggesting improvements to documentation.

If you'd like to get involved, but are not yet sure how, please feel free to send us an [email](mailto:alexander-fischer1801@t-online.de). Some familiarity with
either Python or econometrics will help, but you really don't need to be a `numpy` core developer or have published in [Econometrica](https://onlinelibrary.wiley.com/journal/14680262) =) We'd be more than happy to invest time to help you get started!

## Contributors ✨

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/styfenschaer">*Image omitted.*<br /><sub><b>styfenschaer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=styfenschaer" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.nkeleher.com/">*Image omitted.*<br /><sub><b>Niall Keleher</b></sub></a><br /><a href="#infra-NKeleher" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=NKeleher" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://wenzhi-ding.com">*Image omitted.*<br /><sub><b>Wenzhi Ding</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Wenzhi-Ding" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://apoorvalal.github.io/">*Image omitted.*<br /><sub><b>Apoorva Lal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=apoorvalal" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aapoorvalal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://juanitorduz.github.io">*Image omitted.*<br /><sub><b>Juan Orduz</b></sub></a><br /><a href="#infra-juanitorduz" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=juanitorduz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://s3alfisc.github.io/">*Image omitted.*<br /><sub><b>Alexander Fischer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=s3alfisc" title="Code">💻</a> <a href="#infra-s3alfisc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.aeturrell.com">*Image omitted.*<br /><sub><b>aeturrell</b></sub></a><br /><a href="#tutorial-aeturrell" title="Tutorials">✅</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=aeturrell" title="Documentation">📖</a> <a href="#promotion-aeturrell" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/leostimpfle">*Image omitted.*<br /><sub><b>leostimpfle</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=leostimpfle" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aleostimpfle" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/baggiponte">*Image omitted.*<br /><sub><b>baggiponte</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=baggiponte" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sanskriti2005">*Image omitted.*<br /><sub><b>Sanskriti</b></sub></a><br /><a href="#infra-sanskriti2005" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jayhyung">*Image omitted.*<br /><sub><b>Jaehyung</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Jayhyung" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://alexstephenson.me">*Image omitted.*<br /><sub><b>Alex</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=asteves" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/greenguy33">*Image omitted.*<br /><sub><b>Hayden Freedman</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/saidamir">*Image omitted.*<br /><sub><b>Aziz Mamatov</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=saidamir" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/rafimikail">*Image omitted.*<br /><sub><b>rafimikail</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=rafimikail" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/benjamin-knight/">*Image omitted.*<br /><sub><b>Benjamin Knight</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=b-knight" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://dsliwka.github.io/">*Image omitted.*<br /><sub><b>Dirk Sliwka</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/daltonm-bls">*Image omitted.*<br /><sub><b>daltonm-bls</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Adaltonm-bls" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/marcandre259">*Image omitted.*<br /><sub><b>Marc-André</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=marcandre259" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarcandre259" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kylebutts">*Image omitted.*<br /><sub><b>Kyle F Butts</b></sub></a><br /><a href="#data-kylebutts" title="Data">🔣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://fosstodon.org/@marcogorelli">*Image omitted.*<br /><sub><b>Marco Edward Gorelli</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/pulls?q=is%3Apr+reviewed-by%3AMarcoGorelli" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://arelbundock.com">*Image omitted.*<br /><sub><b>Vincent Arel-Bundock</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=vincentarelbundock" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/IshwaraHegde97">*Image omitted.*<br /><sub><b>IshwaraHegde97</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=IshwaraHegde97" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/RoyalTS">*Image omitted.*<br /><sub><b>Tobias Schmidt</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=RoyalTS" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/escherpf">*Image omitted.*<br /><sub><b>escherpf</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aescherpf" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=escherpf" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.ivanhigueram.com">*Image omitted.*<br /><sub><b>Iván Higuera Mendieta</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=ivanhigueram" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adamvig96">*Image omitted.*<br /><sub><b>Ádám Vig</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=adamvig96" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://szymon.info">*Image omitted.*<br /><sub><b>Szymon Sacher</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=elchorro" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/AronNemeth">*Image omitted.*<br /><sub><b>AronNemeth</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=AronNemeth" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DTchebotarev">*Image omitted.*<br /><sub><b>Dmitri Tchebotarev</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=DTchebotarev" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/FuZhiyu">*Image omitted.*<br /><sub><b>FuZhiyu</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3AFuZhiyu" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=FuZhiyu" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.marceloortizm.com">*Image omitted.*<br /><sub><b>Marcelo Ortiz M.</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=mortizm1988" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jestover">*Image omitted.*<br /><sub><b>Joseph Stover</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jestover" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/JaapCTJ">*Image omitted.*<br /><sub><b>JaapCTJ</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=JaapCTJ" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://shapiromh.com">*Image omitted.*<br /><sub><b>Matt Shapiro</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=shapiromh" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/schroedk">*Image omitted.*<br /><sub><b>Kristof Schröder</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=schroedk" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/WiktorTheScriptor">*Image omitted.*<br /><sub><b>Wiktor </b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=WiktorTheScriptor" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://damandhaliwal.me">*Image omitted.*<br /><sub><b>Daman Dhaliwal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=damandhaliwal" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://markkaj.github.io/">*Image omitted.*<br /><sub><b>Jaakko Markkanen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarkkaj" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://jsr-p.github.io/">*Image omitted.*<br /><sub><b>Jonas Skjold Raaschou-Pedersen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="http://bobbyho.me">*Image omitted.*<br /><sub><b>Bobby Ho</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=bobby1030" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Erica-Ryan">*Image omitted.*<br /><sub><b>Erica Ryan</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Erica-Ryan" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/souhil25">*Image omitted.*<br /><sub><b>Souhil Abdelmalek Louddad</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=souhil25" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://dpananos.github.io">*Image omitted.*<br /><sub><b>Demetri Pananos</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Dpananos" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/pulls?q=is%3Apr+reviewed-by%3ADpananos" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pdoupe">*Image omitted.*<br /><sub><b>Patrick Doupe</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=pdoupe" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.linkedin.com/in/ariadnaaz/">*Image omitted.*<br /><sub><b>Ariadna Albors Zumel</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Ariadnaaz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/blucap">*Image omitted.*<br /><sub><b>Martien Lubberink</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=blucap" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jonpcohen">*Image omitted.*<br /><sub><b>jonpcohen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jonpcohen" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/rhstanton">*Image omitted.*<br /><sub><b>rhstanton</b></sub></a><br /><a href="#userTesting-rhstanton" title="User Testing">📓</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Arhstanton" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/mdizadi">*Image omitted.*<br /><sub><b>mdizadi</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=mdizadi" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://tmensinger.com">*Image omitted.*<br /><sub><b>Tim Mensinger</b></sub></a><br /><a href="#infra-timmens" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://nicholasjunge.com">*Image omitted.*<br /><sub><b>Nicholas Junge</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=nicholasjng" title="Code">💻</a> <a href="#infra-nicholasjng" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://portfolio-abelaba.vercel.app/">*Image omitted.*<br /><sub><b>Abel Abate</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=abelaba" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://janfb.github.io">*Image omitted.*<br /><sub><b>Jan Teusen (né Boelts)</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=janfb" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
