## PyFixest

[![PyPI - Version](https://img.shields.io/pypi/v/pyfixest.svg)](https://pypi.org/project/pyfixest/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyfixest.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dm/pyfixest)
[![image](https://codecov.io/gh/s3alfisc/pyfixest/branch/master/graph/badge.svg)](https://codecov.io/gh/s3alfisc/pyfixest)

`PyFixest` is a Python implementation of the formidable [fixest](https://github.com/lrberge/fixest) package. The package aims to mimic `fixest` syntax and functionality as closely as Python allows. For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/) or take a look at the regression chapter of [Arthur Turrell's](https://github.com/aeturrell) book on [Coding for Economists](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#imports).

`PyFixest` supports

- OLS and IV Regression
- Poisson Regression
- Multiple Estimation Syntax
- Several Robust and Cluster Robust Variance-Covariance Types
- Wild Cluster Bootstrap Inference (via [wildboottest](https://github.com/s3alfisc/wildboottest))
- Difference-in-Difference Estimators:
  - The canonical Two-Way Fixed Effects Estimator
  - [Gardner's two-stage ("`Did2s`")](https://jrgcmu.github.io/2sdd_current.pdf) estimator
  - Basic Versions of the Local Projections estimator following [Dube et al (2023)](https://www.nber.org/papers/w31184)

## Installation

You can install the release version from `PyPi` by running

```py
pip install pyfixest
```
or the development version from github by running
```py
pip install git+https://github.com/s3alfisc/pyfixest.git
```

## News

`PyFixest` `0.13` adds support for the local projections "DID2s" estimator:




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Estimate</th>
      <th>Std. Error</th>
      <th>t value</th>
      <th>Pr(&gt;|t|)</th>
      <th>2.5 %</th>
      <th>97.5 %</th>
      <th>N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>treat_diff</th>
      <td>31.794381</td>
      <td>0.755459</td>
      <td>42.086191</td>
      <td>0.0</td>
      <td>30.312812</td>
      <td>33.27595</td>
      <td>28709.0</td>
    </tr>
  </tbody>
</table>
</div>






<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" class="plt-container" width="500.0" height="300.0">
  <style type="text/css">
  .plt-container {
   font-family: Lucida Grande, sans-serif;
   user-select: none;
   -webkit-user-select: none;
   -moz-user-select: none;
   -ms-user-select: none;
}
text {
   text-rendering: optimizeLegibility;
}
#peQKOGG .plot-title {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 16.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .plot-subtitle {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 15.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .plot-caption {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .legend-title {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 15.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .legend-item {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .axis-title-x {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 15.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .axis-text-x {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#dMP4Xhp .axis-tooltip-text-x {
   fill: #ffffff;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .axis-title-y {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 15.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .axis-text-y {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#dMP4Xhp .axis-tooltip-text-y {
   fill: #ffffff;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .facet-strip-text-x {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#peQKOGG .facet-strip-text-y {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#dMP4Xhp .tooltip-text {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: normal;
   font-style: normal;   
}
#dMP4Xhp .tooltip-title {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: bold;
   font-style: normal;   
}
#dMP4Xhp .tooltip-label {
   fill: #474747;
   font-family: Lucida Grande, sans-serif;
   font-size: 13.0px;
   font-weight: bold;
   font-style: normal;   
}

  </style>
  <g id="peQKOGG">
    <path fill-rule="evenodd" fill="rgb(255,255,255)" fill-opacity="1.0" d="M0.0 0.0 L0.0 300.0 L500.0 300.0 L500.0 0.0 Z">
    </path>
    <g transform="translate(23.0 34.0 ) ">
      <g transform="translate(21.961210910936405 0.0 ) ">
        <line x1="156.7923229319766" y1="220.0" x2="156.7923229319766" y2="-2.842170943040401E-14" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
        <line x1="307.9174534688215" y1="220.0" x2="307.9174534688215" y2="-2.842170943040401E-14" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
      </g>
      <g transform="translate(21.961210910936405 220.0 ) ">
        <g transform="translate(5.667192395131685 0.0 ) ">
          <line stroke-width="1.0" stroke="rgb(71,71,71)" stroke-opacity="1.0" x2="0.0" y2="4.0">
          </line>
          <g transform="translate(0.0 13.5 ) ">
            <text class="axis-text-x" text-anchor="middle" dy="0.35em">
              <tspan>time_to_treatment::-5</tspan>
            </text>
          </g>
        </g>
        <g transform="translate(156.7923229319766 0.0 ) ">
          <line stroke-width="1.0" stroke="rgb(71,71,71)" stroke-opacity="1.0" x2="0.0" y2="4.0">
          </line>
          <g transform="translate(0.0 13.5 ) ">
            <text class="axis-text-x" text-anchor="middle" dy="0.35em">
              <tspan>time_to_treatment::2</tspan>
            </text>
          </g>
        </g>
        <g transform="translate(307.9174534688215 0.0 ) ">
          <line stroke-width="1.0" stroke="rgb(71,71,71)" stroke-opacity="1.0" x2="0.0" y2="4.0">
          </line>
          <g transform="translate(0.0 13.5 ) ">
            <text class="axis-text-x" text-anchor="middle" dy="0.35em">
              <tspan>time_to_treatment::8</tspan>
            </text>
          </g>
        </g>
        <line x1="0.0" y1="0.0" x2="363.95968937623485" y2="0.0" stroke-width="1.0" stroke="rgb(71,71,71)" stroke-opacity="1.0">
        </line>
      </g>
      <g transform="translate(21.961210910936405 0.0 ) ">
        <line x1="0.0" y1="205.05339139443123" x2="363.95968937623485" y2="205.05339139443123" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
        <line x1="0.0" y1="153.0222511621569" x2="363.95968937623485" y2="153.0222511621569" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
        <line x1="0.0" y1="100.99111092988254" x2="363.95968937623485" y2="100.99111092988254" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
        <line x1="0.0" y1="48.959970697608185" x2="363.95968937623485" y2="48.959970697608185" stroke="rgb(233,233,233)" stroke-opacity="1.0" stroke-width="1.0" fill="none">
        </line>
      </g>
      <g transform="translate(21.961210910936405 0.0 ) ">
        <g transform="translate(0.0 205.05339139443123 ) ">
          <g transform="translate(-3.0 0.0 ) ">
            <text class="axis-text-y" text-anchor="end" dy="0.35em">
              <tspan>0</tspan>
            </text>
          </g>
        </g>
        <g transform="translate(0.0 153.0222511621569 ) ">
          <g transform="translate(-3.0 0.0 ) ">
            <text class="axis-text-y" text-anchor="end" dy="0.35em">
              <tspan>20</tspan>
            </text>
          </g>
        </g>
        <g transform="translate(0.0 100.99111092988254 ) ">
          <g transform="translate(-3.0 0.0 ) ">
            <text class="axis-text-y" text-anchor="end" dy="0.35em">
              <tspan>40</tspan>
            </text>
          </g>
        </g>
        <g transform="translate(0.0 48.959970697608185 ) ">
          <g transform="translate(-3.0 0.0 ) ">
            <text class="axis-text-y" text-anchor="end" dy="0.35em">
              <tspan>60</tspan>
            </text>
          </g>
        </g>
      </g>
      <g transform="translate(21.961210910936405 0.0 ) " clip-path="url(#ctiPkEn)" clip-bounds-jfx="[rect (0.0, 0.0), (363.95968937623485, 220.0)]">
        <defs>
          <clipPath id="ctiPkEn">
            <rect x="0.0" y="0.0" width="363.95968937623485" height="220.0">
            </rect>
          </clipPath>
        </defs>
        <g>
          <g >
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="5.667192395131685" cy="205.16412910374248" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="30.854714151272507" cy="203.3875252637302" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="56.04223590741333" cy="202.24414877876814" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="81.22975766355414" cy="201.25863045471667" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="106.41727941969496" cy="195.58332470893762" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="131.6048011758358" cy="186.54987013569072" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="156.7923229319766" cy="179.6887028637255" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="181.97984468811742" cy="166.85391037734996" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="207.16736644425825" cy="153.3382809069135" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="232.35488820039907" cy="130.90803021258316" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="257.5424099565399" cy="114.74188814271155" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="282.7299317126807" cy="93.51143580325065" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="307.9174534688215" cy="66.61911430320365" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="333.10497522496235" cy="42.10699515906421" r="3.3000000000000003" />
            <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="358.29249698110317" cy="16.087129215735388" r="3.3000000000000003" />
          </g>
        </g>
      </g>
      <g transform="translate(21.961210910936405 0.0 ) " clip-path="url(#c0yluou)" clip-bounds-jfx="[rect (0.0, 0.0), (363.95968937623485, 220.0)]">
        <defs>
          <clipPath id="c0yluou">
            <rect x="0.0" y="0.0" width="363.95968937623485" height="220.0">
            </rect>
          </clipPath>
        </defs>
        <g>
          <line x1="5.037504351228164" y1="210.0" x2="6.296880439035205" y2="210.0" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="5.037504351228164" y1="200.32825820748496" x2="6.296880439035205" y2="200.32825820748496" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="5.667192395131685" y1="210.0" x2="5.667192395131685" y2="200.32825820748496" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="30.225026107368986" y1="208.2771605939079" x2="31.484402195176024" y2="208.2771605939079" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="30.225026107368986" y1="198.4978899335525" x2="31.484402195176024" y2="198.4978899335525" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="30.854714151272507" y1="208.2771605939079" x2="30.854714151272507" y2="198.4978899335525" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="55.41254786350981" y1="206.81688528960768" x2="56.67192395131685" y2="206.81688528960768" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="55.41254786350981" y1="197.6714122679286" x2="56.67192395131685" y2="197.6714122679286" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="56.04223590741333" y1="206.81688528960768" x2="56.04223590741333" y2="197.6714122679286" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="80.60006961965063" y1="205.47303028859116" x2="81.85944570745767" y2="205.47303028859116" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="80.60006961965063" y1="197.0442306208422" x2="81.85944570745767" y2="197.0442306208422" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="81.22975766355414" y1="205.47303028859116" x2="81.22975766355414" y2="197.0442306208422" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="105.78759137579145" y1="199.63686988856244" x2="107.04696746359849" y2="199.63686988856244" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="105.78759137579145" y1="191.5297795293128" x2="107.04696746359849" y2="191.5297795293128" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="106.41727941969496" y1="199.63686988856244" x2="106.41727941969496" y2="191.5297795293128" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="130.97511313193226" y1="191.1869928829804" x2="132.2344892197393" y2="191.1869928829804" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="130.97511313193226" y1="181.91274738840104" x2="132.2344892197393" y2="181.91274738840104" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="131.6048011758358" y1="191.1869928829804" x2="131.6048011758358" y2="181.91274738840104" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="156.1626348880731" y1="184.5707774624443" x2="157.4220109758801" y2="184.5707774624443" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="156.1626348880731" y1="174.80662826500665" x2="157.4220109758801" y2="174.80662826500665" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="156.7923229319766" y1="184.5707774624443" x2="156.7923229319766" y2="174.80662826500665" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="181.3501566442139" y1="171.7999898785958" x2="182.60953273202094" y2="171.7999898785958" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="181.3501566442139" y1="161.90783087610416" x2="182.60953273202094" y2="161.90783087610416" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="181.97984468811742" y1="171.7999898785958" x2="181.97984468811742" y2="161.90783087610416" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="206.5376784003547" y1="158.50439591461702" x2="207.7970544881618" y2="158.50439591461702" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="206.5376784003547" y1="148.17216589920997" x2="207.7970544881618" y2="148.17216589920997" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="207.16736644425825" y1="158.50439591461702" x2="207.16736644425825" y2="148.17216589920997" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="231.72520015649553" y1="136.08029080040563" x2="232.9845762443026" y2="136.08029080040563" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="231.72520015649553" y1="125.7357696247607" x2="232.9845762443026" y2="125.7357696247607" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="232.35488820039907" y1="136.08029080040563" x2="232.35488820039907" y2="125.7357696247607" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="256.91272191263636" y1="119.94976528431263" x2="258.17209800044344" y2="119.94976528431263" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="256.91272191263636" y1="109.53401100111044" x2="258.17209800044344" y2="109.53401100111044" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="257.5424099565399" y1="119.94976528431263" x2="257.5424099565399" y2="109.53401100111044" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="282.1002436687772" y1="98.78587008078298" x2="283.35961975658427" y2="98.78587008078298" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="282.1002436687772" y1="88.23700152571831" x2="283.35961975658427" y2="88.23700152571831" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="282.7299317126807" y1="98.78587008078298" x2="282.7299317126807" y2="88.23700152571831" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="307.287765424918" y1="72.19857010725542" x2="308.5471415127251" y2="72.19857010725542" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="307.287765424918" y1="61.03965849915187" x2="308.5471415127251" y2="61.03965849915187" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="307.9174534688215" y1="72.19857010725542" x2="307.9174534688215" y2="61.03965849915187" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="332.47528718105883" y1="47.75694694252479" x2="333.73466326886586" y2="47.75694694252479" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="332.47528718105883" y1="36.457043375603604" x2="333.73466326886586" y2="36.457043375603604" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="333.10497522496235" y1="47.75694694252479" x2="333.10497522496235" y2="36.457043375603604" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
        <g>
          <line x1="357.66280893719966" y1="22.174258431470804" x2="358.9221850250067" y2="22.174258431470804" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="357.66280893719966" y1="9.999999999999972" x2="358.9221850250067" y2="9.999999999999972" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
          <line x1="358.29249698110317" y1="22.174258431470804" x2="358.29249698110317" y2="9.999999999999972" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(17,142,216)" fill-opacity="1.0" stroke-width="1.6500000000000001">
          </line>
        </g>
      </g>
    </g>
    <g transform="translate(44.9612109109364 15.2 ) ">
      <text class="plot-title" y="0.0">
        <tspan>LPDID Event Study Estimate</tspan>
      </text>
    </g>
    <g transform="translate(14.5 144.0 ) rotate(-90.0 ) ">
      <text class="axis-title-y" y="0.0" text-anchor="middle">
        <tspan>Estimate and 95% Confidence Interval</tspan>
      </text>
    </g>
    <g transform="translate(226.9410555990538 291.5 ) ">
      <text class="axis-title-x" y="0.0" text-anchor="middle">
        <tspan>Coefficient</tspan>
      </text>
    </g>
    <g transform="translate(418.92090028717126 111.25 ) ">
      <rect x="5.0" y="5.0" height="55.5" width="71.07909971282872" stroke="rgb(71,71,71)" stroke-opacity="1.0" stroke-width="0.0" fill="rgb(255,255,255)" fill-opacity="1.0">
      </rect>
      <g transform="translate(10.0 10.0 ) ">
        <g transform="translate(0.0 10.5 ) ">
          <text class="legend-title" y="0.0">
            <tspan>Model</tspan>
          </text>
        </g>
        <g transform="translate(0.0 22.5 ) ">
          <g transform="">
            <g>
              <rect x="0.0" y="0.0" height="23.0" width="23.0" stroke-width="0.0" fill="rgb(255,255,255)" fill-opacity="1.0">
              </rect>
              <g transform="translate(1.0 1.0 ) ">
                <g>
                  <g >
                    <circle fill="#e41a1c" stroke="#e41a1c" stroke-opacity="0.0" stroke-width="0.0" cx="10.5" cy="10.5" r="5.5" />
                  </g>
                </g>
                <g>
                  <line x1="6.146249999999999" y1="0.8250000000000001" x2="14.853750000000002" y2="0.8250000000000001" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(255,255,255)" fill-opacity="1.0" stroke-width="1.6500000000000001">
                  </line>
                  <line x1="6.146249999999999" y1="20.175" x2="14.853750000000002" y2="20.175" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(255,255,255)" fill-opacity="1.0" stroke-width="1.6500000000000001">
                  </line>
                  <line x1="10.5" y1="0.8250000000000001" x2="10.5" y2="20.175" stroke="rgb(228,26,28)" stroke-opacity="1.0" fill="rgb(255,255,255)" fill-opacity="1.0" stroke-width="1.6500000000000001">
                  </line>
                </g>
              </g>
              <rect x="0.0" y="0.0" height="23.0" width="23.0" stroke="rgb(255,255,255)" stroke-opacity="1.0" stroke-width="1.0" fill-opacity="0.0">
              </rect>
            </g>
            <g transform="translate(26.9903027277341 16.05 ) ">
              <text class="legend-item" y="0.0">
                <tspan>lpdid</tspan>
              </text>
            </g>
          </g>
        </g>
      </g>
    </g>
    <path fill="rgb(0,0,0)" fill-opacity="0.0" stroke="rgb(71,71,71)" stroke-opacity="1.0" stroke-width="0.0" d="M0.0 0.0 L0.0 300.0 L500.0 300.0 L500.0 0.0 Z">
    </path>
  </g>
  <g id="dMP4Xhp">
  </g>
</svg>



## Benchmarks

All benchmarks follow the [fixest benchmarks](https://github.com/lrberge/fixest/tree/master/_BENCHMARK). All non-pyfixest timings are taken from the `fixest` benchmarks.

![](./benchmarks/lets-plot-images/benchmarks_ols.svg)
![](./benchmarks/lets-plot-images/benchmarks_poisson.svg)

## Quickstart

You can estimate a linear regression models just as you would in `fixest` - via `feols()`:

    ###
    
    Estimation:  OLS
    Dep. var.: Y, Fixed effects: f1+f2
    Inference:  CRV1
    Observations:  997
    
    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | X1            |     -0.919 |        0.065 |   -14.057 |      0.000 |  -1.053 |   -0.786 |
    ---
    RMSE: 1.441   R2: 0.609   R2 Within: 0.2
    

You can estimate multiple models at once by using [multiple estimation syntax](https://aeturrell.github.io/coding-for-economists/econmt-regression.html#multiple-regression-models):

    Model:  Y~X1
    Model:  Y~X1|f1
    Model:  Y~X1|f1+f2
                              est1               est2               est3
    ------------  ----------------  -----------------  -----------------
    depvar                       Y                  Y                  Y
    --------------------------------------------------------------------
    Intercept     0.919*** (0.121)
    X1             -1.0*** (0.117)  -0.949*** (0.087)  -0.919*** (0.069)
    --------------------------------------------------------------------
    f1                           -                  x                  x
    f2                           -                  -                  x
    --------------------------------------------------------------------
    R2                       0.123              0.437              0.609
    S.E. type         by: group_id       by: group_id       by: group_id
    Observations               998                997                997
    --------------------------------------------------------------------
    Significance levels: * p < 0.05, ** p < 0.01, *** p < 0.001
    

Standard Errors can be adjusted after estimation, "on-the-fly":

    Model:  Y~X1
    ###
    
    Estimation:  OLS
    Dep. var.: Y
    Inference:  hetero
    Observations:  998
    
    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | Intercept     |      0.919 |        0.112 |     8.223 |      0.000 |   0.699 |    1.138 |
    | X1            |     -1.000 |        0.082 |   -12.134 |      0.000 |  -1.162 |   -0.838 |
    ---
    RMSE: 2.158   R2: 0.123
    

You can estimate Poisson Regressions via the `fepois()` function:




    '<table border="1" class="dataframe">\n  <thead>\n    <tr style="text-align: right;">\n      <th></th>\n      <th>Estimate</th>\n      <th>Std. Error</th>\n      <th>t value</th>\n      <th>Pr(&gt;|t|)</th>\n      <th>2.5 %</th>\n      <th>97.5 %</th>\n    </tr>\n    <tr>\n      <th>Coefficient</th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n      <th></th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>X1</th>\n      <td>-0.008268</td>\n      <td>0.034600</td>\n      <td>-0.238967</td>\n      <td>0.811131</td>\n      <td>-0.076083</td>\n      <td>0.059547</td>\n    </tr>\n    <tr>\n      <th>X2</th>\n      <td>-0.015107</td>\n      <td>0.010269</td>\n      <td>-1.471148</td>\n      <td>0.141251</td>\n      <td>-0.035234</td>\n      <td>0.005020</td>\n    </tr>\n  </tbody>\n</table>'



Last, `PyFixest` also supports IV estimation via three part formula syntax:

    ###
    
    Estimation:  IV
    Dep. var.: Y, Fixed effects: f1
    Inference:  CRV1
    Observations:  997
    
    | Coefficient   |   Estimate |   Std. Error |   t value |   Pr(>|t|) |   2.5 % |   97.5 % |
    |:--------------|-----------:|-------------:|----------:|-----------:|--------:|---------:|
    | X1            |     -1.025 |        0.115 |    -8.930 |      0.000 |  -1.259 |   -0.790 |
    ---
    
