from pyfixest.estimation import literals
from pyfixest.estimation.demean_ import (
    demean,
)
from pyfixest.estimation.detect_singletons_ import (
    detect_singletons,
)
from pyfixest.estimation.estimation import (
    feglm,
    feols,
    fepois,
    quantreg,
)
from pyfixest.estimation.fegaussian_ import Fegaussian
from pyfixest.estimation.feiv_ import (
    Feiv,
)
from pyfixest.estimation.felogit_ import Felogit
from pyfixest.estimation.feols_ import (
    Feols,
)
from pyfixest.estimation.fepois_ import (
    Fepois,
)
from pyfixest.estimation.feprobit_ import Feprobit
from pyfixest.estimation.FixestMulti_ import (
    FixestMulti,
)
from pyfixest.estimation.model_matrix_fixest_ import (
    model_matrix_fixest,
)
from pyfixest.estimation.multcomp import (
    bonferroni,
    rwolf,
    wyoung,
)
from pyfixest.estimation.quantreg.quantreg_ import Quantreg

__all__ = [
    "Fegaussian",
    "Feiv",
    "Felogit",
    "Feols",
    "Fepois",
    "Feprobit",
    "FixestMulti",
    "Quantreg",
    "bonferroni",
    "demean",
    "detect_singletons",
    "feglm",
    "feols",
    "fepois",
    "literals",
    "model_matrix_fixest",
    "quantreg",
    "rwolf",
    "wyoung",
]


# monkey patch formulaic to emulate https://github.com/matthewwardrop/formulaic/pull/263
from formulaic.transforms.contrasts import TreatmentContrasts

if "drop" not in TreatmentContrasts.__dataclass_fields__:
    from functools import wraps
    _orig_init = TreatmentContrasts.__init__

    @wraps(_orig_init)
    def _patched_init(self, *args, drop=False, **kwargs):
        self.drop = drop
        kwargs.pop("drop", None)
        _orig_init(self, *args, **kwargs)

    TreatmentContrasts.__init__ = _patched_init

    methods: list[str] = [
        "_get_coding_matrix",
        "_apply",
        "get_coding_column_names",
        "get_coefficient_row_names",
    ]

    def _make_patch(orig):
        @wraps(orig)
        def _patched(self, *args, **kwargs):
            if "reduced_rank" in kwargs:
                kwargs["reduced_rank"] |= self.drop
            return orig(self, *args, **kwargs)

        return _patched

    for method in methods:
        setattr(
            TreatmentContrasts, method, _make_patch(getattr(TreatmentContrasts, method))
        )
