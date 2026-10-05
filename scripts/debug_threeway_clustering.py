"""Run with: pixi run -e py312 python scripts/debug_threeway_clustering.py."""

from __future__ import annotations

import pyfixest as pf


def main() -> None:
    data = pf.get_data(N=1000, seed=221).dropna()

    fit = pf.feols(
        "Y ~ X1 + X2 | f1 + f2",
        data=data,
        vcov={"CRV1": "f1+f2+group_id"},
    )
    print(fit.tidy())


if __name__ == "__main__":
    main()
