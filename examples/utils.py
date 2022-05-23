import numpy as np
from scipy.integrate import quad
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns

def diagnosis_pred(true, pred, lower_bound = None, upper_bound = None) -> dict:
    """
    L1, L2, and L_inf errors
    """
    results =  {
        "l1": np.mean(np.abs(true - pred)),
        "l2": np.sqrt(np.mean((true - pred) ** 2)),
        "l_inf": np.max((np.abs(true-pred))),
        
    }
    if lower_bound is not None:
        results["lower_bound_violation"] = np.mean((pred < lower_bound))
    if upper_bound is not None:
        results["lower_bound_violation"] = np.mean((pred < lower_bound))
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


def diagnosis_grads(hessian, grads, f_to_i: Callable, var_ttm:str, var_money: str) -> dict:
    """
    Errors in gradients for call
    """
    return {
        "monotonicity_error": np.mean(grads[:, f_to_i(var_money)] < 0),
        "time_value_error": np.mean(grads[:, f_to_i(var_ttm)] < 0),
        "convex_error": np.mean(hessian[:, f_to_i(var_money)] < 0)
     }

# def sigmoid(a, x):
#     return 1 / (1 + np.exp(-a * x))
# xs = np.linspace(-3, 3)
# fig, ax = plt.subplots()
# for a in [1, 2, 3, 10, 20, 999]:
#     ax.plot(xs, sigmoid(a, xs), label=f"{a}")
# ax.legend()

"""
Plotting utilitiies
"""
def plot_preds(moneyness, ttm, true, preds, lower_bound = None, upper_bound = None):
    # plot predictions vs lower bound
    fig, ax = plt.subplots(ncols = 2)
    sns.scatterplot(moneyness, preds, hue = ttm, label=None, ax = ax[0])
    if lower_bound is not None:
        sns.scatterplot(x = moneyness, y = lower_bound, label = "No-arb-bound", ax = ax[0]);
    if upper_bound is not None:
        sns.scatterplot(x = moneyness, y = upper_bound, label = "No-arb-bound", ax = ax[0]);
    sns.scatterplot(moneyness, true - preds, ax = ax[1])
    ax[1].set_title("Error v Moneyness")

def visualise_surface(moneyness, ttm, preds):
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(111, projection='3d')

    a,b = np.meshgrid(moneyness, ttm)
    preds = preds.reshape((moneyness.shape[0], ttm.shape[0]))
    ax.plot_surface(a, b, preds)