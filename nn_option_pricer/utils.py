import numpy as np
from scipy.integrate import quad
from scipy.stats import skew, kurtosis
from typing import Callable
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List
import plotly.graph_objects as go
import tensorflow as tf

def diagnosis_pred(
    true, pred, lower_bound=None, upper_bound=None, method: str = ""
) -> pd.DataFrame:
    """
    L1, L2, and L_inf errors
    """
    results = {
        "l1": np.mean(np.abs(true - pred)),
        "l2": np.sqrt(np.mean((true - pred) ** 2)),
        "l_inf": np.max((np.abs(true - pred))),
    }
    if lower_bound is not None:
        results["lower_bound_violation"] = np.mean((pred < lower_bound))
    if upper_bound is not None:
        results["upper_bound_violation"] = np.mean((pred > upper_bound))
    return pd.DataFrame(results, index=[method])


def PDE_calc(model, f_to_i: Callable, **kwargs):
    """
    model: One of `['heston', 'heston_log', 'sabr', 'sabr_log', 'bs', 'bs_log']`
    f_to_i: A function that maps parameter names to corresponding indices
    """
    if model == "heston":
        """
        heston PDE for Moneyness
        """
        PDE_err = grads[:, f_to_i("ttm")] - (
            grads[:, f_to_i("V")]
            * Xs[:, f_to_i("kappa")]
            * (Xs[:, f_to_i("vbar")] - Xs[:, f_to_i("V")])
            + Xs[:, f_to_i("rho")]
            * Xs[:, f_to_i("vol_of_vol")]
            * Xs[:, f_to_i("V")]
            * Xs[:, f_to_i("S/K")]
            * hessian1[:, f_to_i("V")]
            + 0.5
            * (Xs[:, f_to_i("S/K")] ** 2)
            * Xs[:, f_to_i("V")]
            * hessian1[:, f_to_i("S/K")]
            + 0.5
            * Xs[:, f_to_i("V")]
            * (Xs[:, f_to_i("vol_of_vol")] ** 2)
            * hessian2[:, f_to_i("V")]
        )

    if model == "heston_log":
        """
        Heston PDE for log-moneyness
        """
        PDE_err = grads[:, f_to_i("ttm")] - (
            grads[:, f_to_i("V")]
            * Xs[:, f_to_i("kappa")]
            * (Xs[:, f_to_i("vbar")] - Xs[:, f_to_i("V")])
            + Xs[:, f_to_i("rho")]
            * Xs[:, f_to_i("vol_of_vol")]
            * Xs[:, f_to_i("V")]
            * hessian1[:, f_to_i("V")]
            + 0.5 * Xs[:, f_to_i("V")] * hessian1[:, f_to_i("log(S/K)")]
            - Xs[:, f_to_i("V")] * grads[:, f_to_i("log(S/K)")]
            + 0.5
            * Xs[:, f_to_i("V")]
            * (Xs[:, f_to_i("vol_of_vol")] ** 2)
            * hessian2[:, f_to_i("V")]
        )

    return PDE_err


def diagnosis_pde(PDE_err: np.array, method: str = "") -> pd.DataFrame:
    """
    Errors in PDE
    PDE_err: PDE Differential for each sample, as a numpy array
    """
    return pd.DataFrame(
        {
            "mean": np.abs(np.mean(PDE_err)),
            "l1": np.mean(np.abs(PDE_err)),
            "l2": np.sqrt(np.mean(PDE_err**2)),
            "l_inf": np.max(np.abs(PDE_err)),
        },
        index=[method],
    )


def diagnosis_grads(
    hessian, grads, f_to_i: Callable, var_ttm: str, var_money: str, method: str = ""
) -> dict:
    """
    Errors in gradients for call
    """
    return pd.DataFrame(
        {
            "monotonicity_error": np.mean(grads[:, f_to_i(var_money)] < 0),
            "time_value_error": np.mean(grads[:, f_to_i(var_ttm)] < 0),
            "convex_error": np.mean(hessian[:, f_to_i(var_money)] < 0),
        },
        index=[method],
    )


def diagnosis_hedge(pnl: np.array, method: str) -> pd.DataFrame:
    """
    Takes in an array, where each entry is the PNL for one sample path
    Returns hedging related statistics
    """
    fig, ax = plt.subplots()
    sns.histplot(pnl, ax=ax)
    ax.set_title(f"Distribution of Hedging PNL - {method}")
    return pd.DataFrame(
        {
            "mean_pnl": pnl.mean(),
            "l1": pnl.abs().mean(),
            "l2": np.sqrt((pnl**2).mean()),
            "linf": np.max(pnl.abs()),
            "skew": skew(pnl),
            "kurtosis": kurtosis(pnl),
            "CVaR_10": -np.nanmean(pnl[pnl < np.quantile(pnl, 0.1)]),
            "CVaR_5": -np.nanmean(pnl[pnl < np.quantile(pnl, 0.05)]),
            "CVaR_2": -np.nanmean(pnl[pnl < np.quantile(pnl, 0.02)]),
            "CVaR_1": -np.nanmean(pnl[pnl < np.quantile(pnl, 0.01)]),
            "CVaR_01": -np.nanmean(pnl[pnl < np.quantile(pnl, 0.001)]),
        },
        index=[method],
    )


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


def plot_preds(
    moneyness, ttm, true, preds, lower_bound=None, upper_bound=None, method: str = ""
):
    """
    Plot Predictions vs Moneyness and time-to-maturity, and masrk upper and lower bounds if they are known
    """
    # plot predictions vs lower bound
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
    sns.scatterplot(x = moneyness, y = preds, hue=ttm, label=None, ax=ax[0])
    idx = np.argsort(moneyness)
    if lower_bound is not None:
        sns.lineplot(x=moneyness[idx], y=lower_bound[idx], label="No-arb lower-bound", ax=ax[0], linestyle="--", color='red')
    if upper_bound is not None:
        sns.lineplot(x=moneyness[idx], y=upper_bound[idx], label="No-arb upper-bound", ax=ax[0], linestyle="--", color='blue')
    sns.scatterplot(x = moneyness, y = true - preds, ax=ax[1], hue = ttm)
    ax[0].set_title(f"{method} - Predictions\nColour, time-to-maturity")
    ax[1].set_title(f"{method} Prediction Error vs Moneyness\nColour, time-to-maturity")
    for i in range(2):
        ax[i].set_xlabel("Moneyness")
    ax[0].set_ylabel("Value")
    ax[1].set_ylabel("Pricing Error")


def visualise_surface(
    moneyness: np.array,
    ttm: np.array,
    preds: np.array,
    x_label: str = "Moneyness",
    y_label: str = "ttm",
    title: str = "Surface",
):
    """
    Utility Function to plot a vanilla surface

    moneyness: numpy array containing sample moneyness (e.g. S/K, log(S/K)) grid points
    ttm: numpy array containing sample maturities (e.g. $\tau = T - t$) grid points
    x_label: label for x-axis
    """
    # fig = plt.figure(figsize=(25, 10))
    # ax = fig.add_subplot(111, projection="3d")

    a, b = np.meshgrid(moneyness, ttm)
    preds = preds.reshape((moneyness.shape[0], ttm.shape[0]))
    fig = go.Figure(
        go.Surface(
            x=a,
            y=b,
            z=preds.T,
            colorbar=dict(lenmode="fraction", len=0.75, thickness=20),
        )
    )
    fig.update_scenes(
        xaxis_title_text=x_label,
        yaxis_title_text=y_label,
        zaxis_title_text=f"Price",
        aspectratio={"x": 1, "y": 1, "z": 1},
    )
    fig.update_layout(width=500, height=600, title=f"{title}")

    # fig = plt.figure(figsize=(25, 10))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.plot_surface(b, a, preds)
    # ax.set_xlabel(x_label)
    # ax.set_ylabel(y_label)
    # ax.set_zlabel("Price")
    # ax.set_title(title)
    return fig


def plot_loss(history, metrics: List[str], method: str = ""):
    fig, ax = plt.subplots(figsize=(15, 5), ncols=2)
    for i, metric in enumerate(metrics):
        ax[i].plot(history.history[metric], label="train")
        ax[i].plot(history.history[f"val_{metric}"], label="val")
        ax[i].legend()
        ax[i].set_yscale("log")
        ax[i].set_title(f"{method} - Training Error:{metric} vs no. epochs")
        ax[i].set_ylabel(f"{metric} (log-scale)")
        ax[i].set_xlabel(f"Epochs")
    return ax

def trainable_params(model):
    return int(np.sum([tf.keras.backend.count_params(p) for p in model.trainable_variables]))