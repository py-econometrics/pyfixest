## PyFixest

<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="99" height="20">
    <linearGradient id="b" x2="0" y2="100%">
        <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
        <stop offset="1" stop-opacity=".1"/>
    </linearGradient>
    <mask id="a">
        <rect width="99" height="20" rx="3" fill="#fff"/>
    </mask>
    <g mask="url(#a)">
        <path fill="#555" d="M0 0h63v20H0z"/>
        <path fill="#a4a61d" d="M63 0h36v20H63z"/>
        <path fill="url(#b)" d="M0 0h99v20H0z"/>
    </g>
    <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
        <text x="31.5" y="15" fill="#010101" fill-opacity=".3">coverage</text>
        <text x="31.5" y="14">coverage</text>
        <text x="80" y="15" fill="#010101" fill-opacity=".3">79%</text>
        <text x="80" y="14">79%</text>
    </g>
</svg>

This is a draft package (highly experimental!) for a Python clone of the excellent [fixest](https://github.com/lrberge/fixest) package.

Fixed effects are projected out via the [PyHDFE](https://github.com/jeffgortmaker/pyhdfe) package.

For a quick introduction, see the [tutorial](https://s3alfisc.github.io/pyfixest/tutorial/).

```python
import pyfixest as pf
from pyfixest.utils import get_data

data = get_data()

fixest = pf.Fixest(data = data)
fixest.feols("Y~X1 | X2", vcov = "HC1")
fixest.summary()

# ###
# 
# model: feols()
# fml: Y~X1 | X2
# ---
# ###
# 
# Fixed effects:  X2
# Dep. var.:  Y 
# Inference:  HC1
# Observations:  998
# 
#     Estimate  Std. Error   t value  Pr(>|t|)
# X1 -0.142274    0.210556 -0.675707  0.499383
```

