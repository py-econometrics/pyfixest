from pyfixest.estimation.feols_ import Feols
from pyfixest.estimation.feols_ import Feols, _drop_multicollinear_variables
from pyfixest.estimation.FormulaParser import FixestFormula
import pandas as pd
import numpy as np
from typing import Union, Optional, Mapping, Any, Literal
import jax.numpy as jnp
from pyfixest.estimation.jax.OLSJAX import OLSJAX

class OLSJAX_API(Feols):

    def __init__(
        self,
        FixestFormula: FixestFormula,
        data: pd.DataFrame,
        ssc_dict: dict[str, Union[str, bool]],
        drop_singletons: bool,
        drop_intercept: bool,
        weights: Optional[str],
        weights_type: Optional[str],
        collin_tol: float,
        fixef_tol: float,
        lookup_demeaned_data: dict[str, pd.DataFrame],
        solver: Literal[
            "np.linalg.lstsq", "np.linalg.solve", "scipy.sparse.linalg.lsqr", "jax"
        ] = "np.linalg.solve",
        demeaner_backend: Literal["numba", "jax"] = "numba",
        store_data: bool = True,
        copy_data: bool = True,
        lean: bool = False,
        context: Union[int, Mapping[str, Any]] = 0,
        sample_split_var: Optional[str] = None,
        sample_split_value: Optional[Union[str, int]] = None,
    ) -> None:
        super().__init__(
            FixestFormula=FixestFormula,
            data=data,
            ssc_dict=ssc_dict,
            drop_singletons=drop_singletons,
            drop_intercept=drop_intercept,
            weights=weights,
            weights_type=weights_type,
            collin_tol=collin_tol,
            fixef_tol=fixef_tol,
            lookup_demeaned_data=lookup_demeaned_data,
            solver=solver,
            store_data=store_data,
            copy_data=copy_data,
            lean=lean,
            sample_split_var=sample_split_var,
            sample_split_value=sample_split_value,
            context=context,
            demeaner_backend=demeaner_backend,
        )

        self.prepare_model_matrix()
        self.to_jax_array()

        # later to be set in multicoll method
        self._N, self._k = self._X_jax.shape

        self.olsjax = OLSJAX(
            X=self._X_jax,
            Y=self._Y_jax,
            fe=self._fe_jax,
            weights=self._weights_jax,
            vcov="iid",
        )
        #import pdb; pdb.set_trace()
        self.olsjax.Y, self.olsjax.X = self.olsjax.demean(Y = self._Y_jax, X = self._X_jax, fe = self._fe_jax, weights = self._weights_jax.flatten())

    def to_jax_array(self):

        self._X_jax = jnp.array(self._X)
        self._Y_jax = jnp.array(self._Y)
        self._fe_jax = jnp.array(self._fe)
        self._weights_jax = jnp.array(self._weights)


    def get_fit(self):

        self.olsjax.get_fit()
        self._beta_hat = self.olsjax.beta.flatten()
        self._u_hat = self.olsjax.residuals
        self._scores = self.olsjax.scores

    def vcov(self, type: str):

        self._vcov_type = type
        self.olsjax.vcov(vcov_type=type)
        self._vcov = self.olsjax.vcov

        return self

    def convert_attributes_to_numpy(self):
        "Convert core attributes from jax to numpy arrays."
        attr = ["_beta_hat", "_u_hat", "_scores", "_vcov"]
        for a in attr:
            # convert to numpy
            setattr(self, a, np.array(getattr(self, a)))









