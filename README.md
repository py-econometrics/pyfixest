![](figures/pyfixest-logo.png)

# PyFixest: Fast High-Dimensional Fixed Effects Regression in Python

[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit)
![Python Versions](https://img.shields.io/badge/Python-3.9–3.14-blue)
[![PyPI Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
[![Coverage](https://codecov.io/gh/py-econometrics/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/py-econometrics/pyfixest)
[![Downloads](https://static.pepy.tech/badge/pyfixest)](https://pepy.tech/project/pyfixest)
[![Downloads](https://static.pepy.tech/badge/pyfixest/month)](https://pepy.tech/project/pyfixest)

[![Project Chat][chat-badge]][chat-url]

[chat-badge]: https://img.shields.io/discord/1259933360726216754.svg?label=&logo=discord&logoColor=ffffff&color=7389D8&labelColor=6A7EC2&style=flat-square
[chat-url]: https://discord.gg/gBAydeDMVK

[Docs](https://pyfixest.org/pyfixest.html) · [Quickstart](https://pyfixest.org/quickstart.html) · [Function & API Reference](https://pyfixest.org/reference/) · [DeepWiki](https://deepwiki.com/py-econometrics/pyfixest) · [Benchmarks](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks) · [Contributing](https://pyfixest.org/contributing.html) · [Changelog](https://pyfixest.org/changelog.html)

`PyFixest` is a Python package for fast high-dimensional fixed effects regression.

The package aims to mimic the syntax and functionality of [Laurent Bergé's](https://sites.google.com/site/laurentrberge/) formidable [fixest](https://github.com/lrberge/fixest) package as closely as Python allows. If you know `fixest` well, the goal is that you won't have to read the docs to get started! In particular, this means that all of `fixest's` defaults are mirrored by `PyFixest`.

For questions on `PyFixest`, head over to our [GitHub discussions](https://github.com/py-econometrics/pyfixest/discussions), or join our [Discord server](https://discord.gg/gBAydeDMVK).

## Features

- **Estimation**
  - **OLS**, **WLS**, **IV**, and **GLMs** (Poisson, logit, probit, gaussian) with high-dimensional fixed effects
  - Different **demeaning backends** (MAP, [within](https://github.com/py-econometrics/within), LSMR) on CPU and GPU
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

For GPU-accelerated fixed-effects demeaning via CuPy, install the build matching your CUDA version:

```bash
pip install cupy-cuda11x
pip install cupy-cuda12x
pip install cupy-cuda13x
```

Then use the typed `demeaner` API for GPU execution:

```python
pf.feols(
    "Y ~ X1 | f1 + f2",
    data=data,
    demeaner=pf.LsmrDemeaner(backend="cupy", precision="float32"),
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

For more examples, see the [quickstart](https://pyfixest.org/quickstart.html), the [formula syntax tutorial](https://pyfixest.org/formula-syntax.html), and the [Poisson & GLMs tutorial](https://pyfixest.org/poisson-glm.html).

## Benchmarks

The dgps follow the "simple" and "difficult" dgps from the [fixest benchmarks](https://github.com/kylebutts/fixest_benchmarks).

### Simple DGP

![](docs/explanation/figures/base-benchmarks/bench_simple.png)

### Difficult DGP

![](docs/explanation/figures/base-benchmarks/bench_difficult.png)

For the full benchmark suite, see the [`benchmarks/`](https://github.com/py-econometrics/pyfixest/tree/master/benchmarks) directory and the note on [difficult fixed effects problems](https://github.com/py-econometrics/pyfixest/blob/master/docs/explanation/difficult-fixed-effects.md).

## Learn More

- [Quickstart](https://pyfixest.org/quickstart.html)
- [Function & API Reference](https://pyfixest.org/reference/)
- [Difference-in-Differences](https://pyfixest.org/difference-in-differences.html)
- [Quantile Regression](https://pyfixest.org/quantile-regression.html)
- [Changelog](https://pyfixest.org/changelog.html)
- [Contributing](https://pyfixest.org/contributing.html)

## Acknowledgements

First and foremost, we want to acknowledge [Laurent Bergé's](https://sites.google.com/site/laurentrberge/) formidable [fixest](https://github.com/lrberge/fixest), which [is so good we decided to stick to its API and conventions](https://youtu.be/kSQxGGA7Rr4?si=8-wTbzLPnIZQ7lYI&t=576) as closely as Python allows. Without `fixest`, `PyFixest` likely wouldn't exist - or at the very least, it would look very different.

For a full list of software packages and papers that have influenced PyFixest, please take a look at the [Acknowledgements page](https://pyfixest.org/acknowledgements.html).

We thank all institutions that have funded or supported work on PyFixest!

<img src="figures/aai-institute-logo.svg" width="185">

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

If you'd like to get involved, but are not yet sure how, please feel free to send us an [email](alexander-fischer1801@t-online.de). Some familiarity with
either Python or econometrics will help, but you really don't need to be a `numpy` core developer or have published in [Econometrica](https://onlinelibrary.wiley.com/journal/14680262) =) We'd be more than happy to invest time to help you get started!

## Contributors ✨

Thanks goes to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/styfenschaer"><img src="https://avatars.githubusercontent.com/u/79762922?v=4?s=40" width="40px;" alt="styfenschaer"/><br /><sub><b>styfenschaer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=styfenschaer" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.nkeleher.com/"><img src="https://avatars.githubusercontent.com/u/5607589?v=4?s=40" width="40px;" alt="Niall Keleher"/><br /><sub><b>Niall Keleher</b></sub></a><br /><a href="#infra-NKeleher" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=NKeleher" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://wenzhi-ding.com"><img src="https://avatars.githubusercontent.com/u/30380959?v=4?s=40" width="40px;" alt="Wenzhi Ding"/><br /><sub><b>Wenzhi Ding</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Wenzhi-Ding" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://apoorvalal.github.io/"><img src="https://avatars.githubusercontent.com/u/12086926?v=4?s=40" width="40px;" alt="Apoorva Lal"/><br /><sub><b>Apoorva Lal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=apoorvalal" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aapoorvalal" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://juanitorduz.github.io"><img src="https://avatars.githubusercontent.com/u/22996444?v=4?s=40" width="40px;" alt="Juan Orduz"/><br /><sub><b>Juan Orduz</b></sub></a><br /><a href="#infra-juanitorduz" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=juanitorduz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://s3alfisc.github.io/"><img src="https://avatars.githubusercontent.com/u/19531450?v=4?s=40" width="40px;" alt="Alexander Fischer"/><br /><sub><b>Alexander Fischer</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=s3alfisc" title="Code">💻</a> <a href="#infra-s3alfisc" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.aeturrell.com"><img src="https://avatars.githubusercontent.com/u/11294320?v=4?s=40" width="40px;" alt="aeturrell"/><br /><sub><b>aeturrell</b></sub></a><br /><a href="#tutorial-aeturrell" title="Tutorials">✅</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=aeturrell" title="Documentation">📖</a> <a href="#promotion-aeturrell" title="Promotion">📣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/leostimpfle"><img src="https://avatars.githubusercontent.com/u/31652181?v=4?s=40" width="40px;" alt="leostimpfle"/><br /><sub><b>leostimpfle</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=leostimpfle" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aleostimpfle" title="Bug reports">🐛</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/baggiponte"><img src="https://avatars.githubusercontent.com/u/57922983?v=4?s=40" width="40px;" alt="baggiponte"/><br /><sub><b>baggiponte</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=baggiponte" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/sanskriti2005"><img src="https://avatars.githubusercontent.com/u/150411024?v=4?s=40" width="40px;" alt="Sanskriti"/><br /><sub><b>Sanskriti</b></sub></a><br /><a href="#infra-sanskriti2005" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Jayhyung"><img src="https://avatars.githubusercontent.com/u/40373774?v=4?s=40" width="40px;" alt="Jaehyung"/><br /><sub><b>Jaehyung</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Jayhyung" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://alexstephenson.me"><img src="https://avatars.githubusercontent.com/u/24926205?v=4?s=40" width="40px;" alt="Alex"/><br /><sub><b>Alex</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=asteves" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/greenguy33"><img src="https://avatars.githubusercontent.com/u/8525718?v=4?s=40" width="40px;" alt="Hayden Freedman"/><br /><sub><b>Hayden Freedman</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=greenguy33" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/saidamir"><img src="https://avatars.githubusercontent.com/u/20246711?v=4?s=40" width="40px;" alt="Aziz Mamatov"/><br /><sub><b>Aziz Mamatov</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=saidamir" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/rafimikail"><img src="https://avatars.githubusercontent.com/u/61386867?v=4?s=40" width="40px;" alt="rafimikail"/><br /><sub><b>rafimikail</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=rafimikail" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://www.linkedin.com/in/benjamin-knight/"><img src="https://avatars.githubusercontent.com/u/12180931?v=4?s=40" width="40px;" alt="Benjamin Knight"/><br /><sub><b>Benjamin Knight</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=b-knight" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://dsliwka.github.io/"><img src="https://avatars.githubusercontent.com/u/49401450?v=4?s=40" width="40px;" alt="Dirk Sliwka"/><br /><sub><b>Dirk Sliwka</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=dsliwka" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/daltonm-bls"><img src="https://avatars.githubusercontent.com/u/78225214?v=4?s=40" width="40px;" alt="daltonm-bls"/><br /><sub><b>daltonm-bls</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Adaltonm-bls" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/marcandre259"><img src="https://avatars.githubusercontent.com/u/19809475?v=4?s=40" width="40px;" alt="Marc-André"/><br /><sub><b>Marc-André</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=marcandre259" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarcandre259" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/kylebutts"><img src="https://avatars.githubusercontent.com/u/19961439?v=4?s=40" width="40px;" alt="Kyle F Butts"/><br /><sub><b>Kyle F Butts</b></sub></a><br /><a href="#data-kylebutts" title="Data">🔣</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://fosstodon.org/@marcogorelli"><img src="https://avatars.githubusercontent.com/u/33491632?v=4?s=40" width="40px;" alt="Marco Edward Gorelli"/><br /><sub><b>Marco Edward Gorelli</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/pulls?q=is%3Apr+reviewed-by%3AMarcoGorelli" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://arelbundock.com"><img src="https://avatars.githubusercontent.com/u/987057?v=4?s=40" width="40px;" alt="Vincent Arel-Bundock"/><br /><sub><b>Vincent Arel-Bundock</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=vincentarelbundock" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/IshwaraHegde97"><img src="https://avatars.githubusercontent.com/u/187858441?v=4?s=40" width="40px;" alt="IshwaraHegde97"/><br /><sub><b>IshwaraHegde97</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=IshwaraHegde97" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/RoyalTS"><img src="https://avatars.githubusercontent.com/u/702580?v=4?s=40" width="40px;" alt="Tobias Schmidt"/><br /><sub><b>Tobias Schmidt</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=RoyalTS" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/escherpf"><img src="https://avatars.githubusercontent.com/u/3789736?v=4?s=40" width="40px;" alt="escherpf"/><br /><sub><b>escherpf</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Aescherpf" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=escherpf" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.ivanhigueram.com"><img src="https://avatars.githubusercontent.com/u/9004403?v=4?s=40" width="40px;" alt="Iván Higuera Mendieta"/><br /><sub><b>Iván Higuera Mendieta</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=ivanhigueram" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/adamvig96"><img src="https://avatars.githubusercontent.com/u/52835042?v=4?s=40" width="40px;" alt="Ádám Vig"/><br /><sub><b>Ádám Vig</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=adamvig96" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://szymon.info"><img src="https://avatars.githubusercontent.com/u/13162607?v=4?s=40" width="40px;" alt="Szymon Sacher"/><br /><sub><b>Szymon Sacher</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=elchorro" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/AronNemeth"><img src="https://avatars.githubusercontent.com/u/96979880?v=4?s=40" width="40px;" alt="AronNemeth"/><br /><sub><b>AronNemeth</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=AronNemeth" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/DTchebotarev"><img src="https://avatars.githubusercontent.com/u/6100882?v=4?s=40" width="40px;" alt="Dmitri Tchebotarev"/><br /><sub><b>Dmitri Tchebotarev</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=DTchebotarev" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/FuZhiyu"><img src="https://avatars.githubusercontent.com/u/11523699?v=4?s=40" width="40px;" alt="FuZhiyu"/><br /><sub><b>FuZhiyu</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3AFuZhiyu" title="Bug reports">🐛</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=FuZhiyu" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.marceloortizm.com"><img src="https://avatars.githubusercontent.com/u/70643379?v=4?s=40" width="40px;" alt="Marcelo Ortiz M."/><br /><sub><b>Marcelo Ortiz M.</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=mortizm1988" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jestover"><img src="https://avatars.githubusercontent.com/u/11298484?v=4?s=40" width="40px;" alt="Joseph Stover"/><br /><sub><b>Joseph Stover</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jestover" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/JaapCTJ"><img src="https://avatars.githubusercontent.com/u/157970664?v=4?s=40" width="40px;" alt="JaapCTJ"/><br /><sub><b>JaapCTJ</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=JaapCTJ" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://shapiromh.com"><img src="https://avatars.githubusercontent.com/u/2327833?v=4?s=40" width="40px;" alt="Matt Shapiro"/><br /><sub><b>Matt Shapiro</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=shapiromh" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/schroedk"><img src="https://avatars.githubusercontent.com/u/38397037?v=4?s=40" width="40px;" alt="Kristof Schröder"/><br /><sub><b>Kristof Schröder</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=schroedk" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/WiktorTheScriptor"><img src="https://avatars.githubusercontent.com/u/162815872?v=4?s=40" width="40px;" alt="Wiktor "/><br /><sub><b>Wiktor </b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=WiktorTheScriptor" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://damandhaliwal.me"><img src="https://avatars.githubusercontent.com/u/48694192?v=4?s=40" width="40px;" alt="Daman Dhaliwal"/><br /><sub><b>Daman Dhaliwal</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=damandhaliwal" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://markkaj.github.io/"><img src="https://avatars.githubusercontent.com/u/24573803?v=4?s=40" width="40px;" alt="Jaakko Markkanen"/><br /><sub><b>Jaakko Markkanen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Amarkkaj" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://jsr-p.github.io/"><img src="https://avatars.githubusercontent.com/u/49307119?v=4?s=40" width="40px;" alt="Jonas Skjold Raaschou-Pedersen"/><br /><sub><b>Jonas Skjold Raaschou-Pedersen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/commits?author=jsr-p" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="http://bobbyho.me"><img src="https://avatars.githubusercontent.com/u/10739091?v=4?s=40" width="40px;" alt="Bobby Ho"/><br /><sub><b>Bobby Ho</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=bobby1030" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/Erica-Ryan"><img src="https://avatars.githubusercontent.com/u/27149581?v=4?s=40" width="40px;" alt="Erica Ryan"/><br /><sub><b>Erica Ryan</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Erica-Ryan" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/souhil25"><img src="https://avatars.githubusercontent.com/u/185626340?v=4?s=40" width="40px;" alt="Souhil Abdelmalek Louddad"/><br /><sub><b>Souhil Abdelmalek Louddad</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=souhil25" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://dpananos.github.io"><img src="https://avatars.githubusercontent.com/u/20362941?v=4?s=40" width="40px;" alt="Demetri Pananos"/><br /><sub><b>Demetri Pananos</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Dpananos" title="Code">💻</a> <a href="https://github.com/py-econometrics/pyfixest/pulls?q=is%3Apr+reviewed-by%3ADpananos" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/pdoupe"><img src="https://avatars.githubusercontent.com/u/215010049?v=4?s=40" width="40px;" alt="Patrick Doupe"/><br /><sub><b>Patrick Doupe</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=pdoupe" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://www.linkedin.com/in/ariadnaaz/"><img src="https://avatars.githubusercontent.com/u/35080247?v=4?s=40" width="40px;" alt="Ariadna Albors Zumel"/><br /><sub><b>Ariadna Albors Zumel</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=Ariadnaaz" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/blucap"><img src="https://avatars.githubusercontent.com/u/614800?v=4?s=40" width="40px;" alt="Martien Lubberink"/><br /><sub><b>Martien Lubberink</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=blucap" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/jonpcohen"><img src="https://avatars.githubusercontent.com/u/48955223?v=4?s=40" width="40px;" alt="jonpcohen"/><br /><sub><b>jonpcohen</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=jonpcohen" title="Documentation">📖</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/rhstanton"><img src="https://avatars.githubusercontent.com/u/2153614?v=4?s=40" width="40px;" alt="rhstanton"/><br /><sub><b>rhstanton</b></sub></a><br /><a href="#userTesting-rhstanton" title="User Testing">📓</a> <a href="https://github.com/py-econometrics/pyfixest/issues?q=author%3Arhstanton" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://github.com/mdizadi"><img src="https://avatars.githubusercontent.com/u/16979129?v=4?s=40" width="40px;" alt="mdizadi"/><br /><sub><b>mdizadi</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=mdizadi" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://tmensinger.com"><img src="https://avatars.githubusercontent.com/u/39708667?v=4?s=40" width="40px;" alt="Tim Mensinger"/><br /><sub><b>Tim Mensinger</b></sub></a><br /><a href="#infra-timmens" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://nicholasjunge.com"><img src="https://avatars.githubusercontent.com/u/54248170?v=4?s=40" width="40px;" alt="Nicholas Junge"/><br /><sub><b>Nicholas Junge</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=nicholasjng" title="Code">💻</a> <a href="#infra-nicholasjng" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a></td>
      <td align="center" valign="top" width="12.5%"><a href="https://portfolio-abelaba.vercel.app/"><img src="https://avatars.githubusercontent.com/u/61546411?v=4?s=40" width="40px;" alt="Abel Abate"/><br /><sub><b>Abel Abate</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=abelaba" title="Code">💻</a></td>
      <td align="center" valign="top" width="12.5%"><a href="http://janfb.github.io"><img src="https://avatars.githubusercontent.com/u/11921527?v=4?s=40" width="40px;" alt="Jan Teusen (né Boelts)"/><br /><sub><b>Jan Teusen (né Boelts)</b></sub></a><br /><a href="https://github.com/py-econometrics/pyfixest/commits?author=janfb" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
