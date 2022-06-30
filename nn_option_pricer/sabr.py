from jax import grad
import jax.numpy as jnp
import pandas as pd
import numpy as np
from typing import Callable, List


def cev_step(
    F0: float, V0: float, beta: float, dt: float, Z1: float, rho: float, Z2: float
):

    """
    F0: value of the forward process
    V0: value of the volatility process
    beta: value of the CEV beta
    dt: timestep
    Z2: value of the brownian motion associated with the volatility process
    Z1: a brownian motion independent of Z2
    rho: corrrelation between brownian motions
    """
    return jnp.abs(
        F0
        + V0
        * (F0 ** (beta))
        * (jnp.sqrt(dt) * (rho * Z2 + jnp.sqrt(1 - rho**2) * Z1))
    )

    return jnp.abs(
        F0
        + V0
        * jnp.exp((beta - 1) * F0)
        * jnp.sqrt(dt)
        * (rho * Z2 + jnp.sqrt(1 - rho**2) + Z1)
        - 0.5 * (V0**2) * dt * jnp.exp(2 * (beta - 1) * F0)
    )


def sabr_expansion(F, K, vol, beta, rho, vol_of_vol, ttm):
    """
    rho: correlation between brownian motions
    vol_of_vol: vol of vol
    vol: spot vol
    ttm: time to maturity
    """
    import jax.numpy as np
    F_K = (F * K) ** (1 - beta)
    mon = np.log(F / K)
    z = (vol_of_vol / vol) * (F_K**0.5) * mon
    x = np.log((np.sqrt(1 - 2 * rho * z + (z**2)) + z - rho) / (1 - rho + 1e-6))
    num = 1 + ttm * (
        ((1 - beta) ** 2) * ((vol**2) / F_K) / 24
        + 0.25 * rho * beta * vol_of_vol * vol / (F_K * 0.5)
        + (2 - 3 * (rho**2)) * ((vol_of_vol) ** 2) / 24
    )
    denom = (F_K**0.5) * (
        1 + ((1 - beta) ** 2) / 24 * mon**2 + ((1 - beta) ** 4) / 1920 * mon**4
    )
    if F == K:
        return  (np.nan_to_num(z / x) * vol * num / denom) + vol * num / denom
    else:
        return (np.nan_to_num(z / x) * vol * num / denom)



def sabr_pde_err(
    X_df: pd.DataFrame,
    grads: np.array,
    hessian_moneyness: np.array,
    hessian_vol: np.array,
    f_to_i: Callable,
) -> np.array:
    """
    Takes in the inputs, gradients, hessian, and returns the

    X_df:
    """
    vol_of_vol = X_df["vol_of_vol"].values
    rho = X_df["rho"].values
    moneyness = X_df["S/K"].values
    beta = X_df["beta"].values
    vol = X_df["V"].values

    PDE_err = (
        -grads[:, f_to_i("ttm")]
        + 0.5 * (moneyness ** (2 * beta)) * vol * hessian_moneyness[:, f_to_i("V")]
        + rho * vol_of_vol * (moneyness**beta)
        + 0.5 * (vol_of_vol**2) * vol * hessian_vol[:, f_to_i("V")]
    )

    return PDE_err


def sabr_eval_wrapper(
    X_df: pd.DataFrame,
    true_val: np.array,
    preds: np.array,
    grads: np.array,
    hessian_moneyness: np.array,
    feat_names: List[str] = ["log(S/K)", "ttm"],
    lower_bound: np.array = None,
    upper_bound: np.array = None,
    METHOD: str = "standard_ffn",
):
    """
    Prediction Error
    """
    f_to_i = lambda x: feat_names.index(x)

    moneyness = X_df["S/K"]
    ttm = X_df["ttm"]

    plot_preds(
        moneyness=moneyness,
        ttm=ttm,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        true=true_val,
        preds=preds,
    )
    pred_stats = pd.DataFrame(
        [diagnosis_pred(true_val, preds, lower_bound, upper_bound)], index=[METHOD]
    ).add_prefix("pred_")

    """
    Error in PDE operator (Dynamic Arbitrage)
    """

    PDE_err = sabr_pde_err(X_df, grads, hessian_moneyness, hessian_vol, f_to_i)
    pde_stats = pd.DataFrame(diagnosis_pde(PDE_err), index=[METHOD]).add_prefix("PDE_")
    """
    Error in Greeks
    """
    N_FEATS = len(feat_names)
    true_first_order = X_df[[f"true_d_{x}" for x in feat_names]].values
    fig, ax = plt.subplots(ncols=N_FEATS, figsize=(5 * N_FEATS, 10), nrows=2)
    for i in range(N_FEATS):
        sns.scatterplot(x=X_df[feat_names[i]], y=grads[:, i], ax=ax[0, i])
        ax[0, i].set_title(feat_names[i])
        # sns.scatterplot(x = X_df[feat_names[i]], y = true_first_order[:, i] - grads[:, i], ax = ax[1, i])
        # ax[1, i].set_title(f"Error - {feat_names[i]}")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    true_second_order = X_df["true_d2_log(S/K)"].values
    ax[0].scatter(X_df["log(S/K)"], hessian_moneyness[:, f_to_i("log(S/K)")])
    ax[0].set_title("Gamma")
    # ax[1].scatter(X_df["log(S/K)"], true_second_order - hessian_moneyness[:, f_to_i("log(S/K)")])
    # ax[1].set_title(f"Error - Gamma")

    # grad_stats = pd.DataFrame(diagnosis_grads(hessian_moneyness, grads, f_to_i, "ttm", "log(S/K)"), index=[METHOD])
    """
    Display Statistics
    """
    # res = pd.concat([pred_stats, pde_stats, grad_stats], axis = 1)
    res = pd.concat([pred_stats, pde_stats], axis=1)
    return res
