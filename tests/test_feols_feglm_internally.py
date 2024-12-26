import numpy as np
import pytest

import pyfixest as pf

fml_list = [
    ("Y ~ X1 + C(f1)", "Y~X1 | f1"),
    ("Y ~ X1 + C(f1) + C(f2)", "Y~X1 | f1 + f2"),
]


@pytest.mark.skip("Fixed effects are not yet supported.")
@pytest.mark.parametrize("fml", fml_list)
@pytest.mark.parametrize("family", ["gaussian"])
def test_feols_feglm_internally(fml, family):
    data = pf.get_data()
    data["Y"] = np.where(data["Y"] > 0, 1, 0)

    fml1, fml2 = fml

    fit1 = pf.feglm(
        fml=fml1, data=data, family=family, ssc=pf.ssc(adj=False, cluster_adj=False)
    )
    fit2 = pf.feglm(
        fml=fml2, data=data, family=family, ssc=pf.ssc(adj=False, cluster_adj=False)
    )

    assert fit1.coef().xs("X1") == fit2.coef().xs(
        "X1"
    ), f"Test failed for fml = {fml} and family = gaussian"
    assert fit1.se().xs("X1") == fit2.se().xs(
        "X1"
    ), f"Test failed for fml = {fml} and family = gaussian"
