import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import diagnosis_pred


def payoff(XT, K):
    from jax.numpy import maximum

    return maximum(XT - K, 0.0)


def abm_step(S0: np.array, WT: np.array, w: np.array) -> np.array:
    """
    Exact simulation of ABM via euler scheme
    """
    return (S0 + WT) @ w


def bachelier_solution(F: float, K: float, sigma: float, tau: float) -> float:
    """
    F: Forward Value
    K: Strike
    sigma: normal volatility
    tau: time-to-maturity
    """
    from jax.numpy import sqrt
    from jax.scipy.stats import norm

    sigma_tau = sigma * sqrt(tau)
    d1 = (F - K) / sigma_tau
    return sigma_tau * (norm.pdf(d1) + d1 * norm.cdf(d1))


def bachelier_eval_wrapper(
    X_df: pd.DataFrame,
    true_val: np.array,
    preds: np.array,
    grads: np.array,
    lower_bound: np.array = None,
    upper_bound: np.array = None,
    METHOD: str = "standard_ffn",
) -> pd.DataFrame:
    """
    Prediction Errors
    """
    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))
    sns.scatterplot(
        X_df["basket"], X_df["call_payoff"], label="Sample Sample Payoffs", ax=ax[0]
    )
    sns.scatterplot(X_df["basket"], preds, label="Predicted", ax=ax[0])
    sns.scatterplot(X_df["basket"], X_df["call_analytic"], label="Analytic", ax=ax[0])
    sns.scatterplot(
        X_df["basket"],
        np.maximum(X_df["basket"] - 1.0, 0),
        label="Lower Bound",
        ax=ax[0],
    )
    ax[0].legend()
    ax[0].set_title("Predictions vs Basket value")

    sns.scatterplot(X_df["basket"], preds - X_df["call_analytic"], ax=ax[1])
    ax[1].set_title("Prediction Error vs Basket Value")

    upper_bound = None
    lower_bound = np.maximum(X_df["basket"] - 1.0, 0)

    pred_stats = pd.DataFrame(
        [diagnosis_pred(X_df["call_analytic"].values, preds, lower_bound, upper_bound)],
        index=[METHOD],
    ).add_prefix("pred_")

    #     """
    #     Error in PDE operator (Dynamic Arbitrage)
    #     """

    #     PDE_err = bs_log_pde_err(moneyness, ttm,
    #                          grads[:, f_to_i("ttm")],
    #                          grads[:, f_to_i("log(S/K)")],
    #                          hessian_moneyness[:, f_to_i("log(S/K)")])
    #     pde_stats = pd.DataFrame(diagnosis_pde(PDE_err), index = [METHOD]).add_prefix("PDE_")
    """
    Error in Greeks
    """
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    ax[0].scatter(X_df["basket"], grads)
    ax[0].set_title("Predicted Gradient")
    N_ASSETS = len([x for x in X_df.columns if x.find("asset_") == 0])
    ax[1].scatter(X_df["basket"], X_df["call_analytic_delta"] / N_ASSETS - grads)
    ax[1].set_title("Gradient Error vs Basket value")
    true_factor = X_df["call_analytic_delta"] / N_ASSETS
    grad_stats = pd.DataFrame(
        diagnosis_pred(true_factor, grads, lower_bound=0), index=[METHOD]
    ).add_prefix("grad_")
    """
    Display Statistics
    """
    res = pd.concat([pred_stats, grad_stats], axis=1)
    return res
