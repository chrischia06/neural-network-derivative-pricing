import numpy as np
from scipy.integrate import quad
from typing import Callable

def diagnosis_pred(true, pred, lower_bound = None) -> dict:
    """
    L1, L2, and L_inf errors
    """
    results =  {
        "l1": np.mean(np.abs(true - pred)),
        "l2": np.sqrt(np.mean((true - pred) ** 2)),
        "l_inf": np.max((np.abs(true-pred))),
        
    }
    if lower_bound is not None:
        results["lower_bound_violation"] = np.mean((pred < lower_bound)),
    return results

def PDE_calc(model, f_to_i:Callable, **kwargs):
    """
    model: One of `['heston', 'heston_log', 'sabr', 'sabr_log', 'bs', 'bs_log']`
    f_to_i: A function that maps parameter names to corresponding indices
    """
    if model == "heston":
        """
        heston PDE for Moneyness
        """
        PDE_err = (
        grads[:, f_to_i("ttm")]
        - (grads[:, f_to_i("V")] * Xs[:, f_to_i("kappa")] * (Xs[:, f_to_i("vbar")] - Xs[:, f_to_i("V")])
        + Xs[:, f_to_i("rho")] * Xs[:, f_to_i("vol_of_vol")] * Xs[:, f_to_i("V")] * Xs[:, f_to_i("S/K")] * hessian1[:, f_to_i("V")]
        + 0.5 * (Xs[:, f_to_i("S/K")] ** 2) * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("S/K")]
        + 0.5 * Xs[:, f_to_i("V")] * (Xs[:, f_to_i("vol_of_vol")] ** 2) * hessian2[:, f_to_i("V")]
        ))
        
    if model == "heston_log":
        """
        Heston PDE for log-moneyness
        """
        PDE_err = (
                grads[:, f_to_i("ttm")]
                - (grads[:, f_to_i("V")] * Xs[:, f_to_i("kappa")] * (Xs[:, f_to_i("vbar")] - Xs[:, f_to_i("V")])
                + Xs[:, f_to_i("rho")] * Xs[:, f_to_i("vol_of_vol")] * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("V")]
                + 0.5 * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("log(S/K)")]
                - Xs[:, f_to_i("V")] * grads[:, f_to_i("log(S/K)")]
                + 0.5 * Xs[:, f_to_i("V")] * (Xs[:, f_to_i("vol_of_vol")] ** 2) * hessian2[:, f_to_i("V")]
                ))
        
    if model == "SABR":
        PDE_err = (
        grads[:, f_to_i("ttm")]
        - (grads[:, f_to_i("V")] * Xs[:, f_to_i("kappa")] * (Xs[:, f_to_i("vbar")] - Xs[:, f_to_i("V")])
        + Xs[:, f_to_i("rho")] * Xs[:, f_to_i("vol_of_vol")] * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("V")]
        + 0.5 * (Xs[:, f_to_i("S/K")] ** 2) * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("S/K")]
        + 0.5 * Xs[:, f_to_i("V")] * (Xs[:, f_to_i("vol_of_vol")] ** 2) * hessian2[:, f_to_i("V")]
        ))
    return PDE_err

def diagnosis_pde(PDE_err):
    """
    Errors in PDE
    PDE_err: PDE Differential for each sample, as a numpy array
    """
    return {
        "mean": np.mean(PDE_err),
        "l1": np.mean(np.abs(PDE_err)),
        "l2": np.sqrt(np.mean(PDE_err ** 2)),
        "l_inf": np.max(np.abs(PDE_err))
    }


def diagnosis_grads(hessian, grads, f_to_i: Callable) -> dict:
    """
    Errors in gradients for call
    """
    return {
        "monotonicity_error": np.mean(grads[:, f_to_i("S/K")].numpy() < 0),
        "time_value_error": np.mean(grads[:, f_to_i("ttm")].numpy() < 0),
        "convex_error": np.mean(hessian[:, f_to_i("S/K")].numpy() < 0)
     }

