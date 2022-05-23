import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax import grad, hessian
import seaborn as sns
import pandas as pd
from utils import plot_preds, diagnosis_grads, diagnosis_pde, diagnosis_pred
from typing import List

def jax_BS_call(SK, sigma_tau):
    """
    SK: Moneyness $log(S/K)$ , log(forward / strike)
    sigma_tau: Time-scaled implied volatility $\sigma \sqrt{\tau} = \sigma \sqrt{T - t}$
    
    return: jax numpy array of BS call price
    """
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return jnp.exp(SK) * norm.cdf(d1) - norm.cdf(d2)

def jax_BS_digital(SK, sigma_tau):
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return norm.cdf(d2)

def BS_pdf(SK, sigma_tau):
    d1 = (SK / sigma_tau) + 0.5 * sigma_tau
    d2 = d1 - sigma_tau
    return norm.pdf(d2)

def first_order_greeks(moneyness, ttm):
    vec_1 = lambda x, y: grad(jax_BS_call, argnums=(0, 1))(x, y)
    delta, vega =  jnp.vectorize(vec_1)(moneyness, ttm)
    return delta, vega

def second_order_greek(moneyness, ttm):
    vec_2 = lambda x, y : grad(grad(jax_BS_call))(x, y)
    gamma = jnp.vectorize(vec_2)(moneyness, ttm)
    return gamma

def bs_log_pde_err(moneyness, ttm, d_ttm, d_x, d2_x):
    fig, ax = plt.subplots()
    PDE_err = -d_ttm + ttm * (-d_x + d2_x)
    ax.scatter(moneyness, PDE_err)
    ax.set_title("PDE Error")
    return PDE_err

def bs_pde_err(moneyness, ttm, d_ttm, d_x, d2_x):
    fig, ax = plt.subplots()
    PDE_err = -d_ttm + 0.5 * ttm * (moneyness ** 2) * d2_x
    ax.scatter(moneyness, PDE_err)
    ax.set_title("PDE Error")
    return PDE_err

def gbm_step(F, dt, sigma, Z):
    return F - (0.5 * (sigma ** 2) * dt) + sigma * Z



def bs_eval_wrapper(X_df: pd.DataFrame, 
                    true_val:np.array, 
                    preds:np.array,
                    grads:np.array,
                    hessian_moneyness:np.array,
                    feat_names:List[str] = ["log(S/K)", "ttm"],
                    lower_bound:np.array = None, 
                    upper_bound:np.array = None,
                    METHOD:str = "standard_ffn"
                    ):
    """
    Prediction Error
    """
    f_to_i = lambda x: feat_names.index(x)
    
    moneyness = X_df["log(S/K)"]
    ttm = X_df["ttm"]
    
    plot_preds(moneyness = moneyness, 
               ttm = ttm, 
               lower_bound = lower_bound,
               upper_bound = upper_bound,
               true = true_val, 
               preds = preds)
    pred_stats = pd.DataFrame([diagnosis_pred(true_val, preds, lower_bound, upper_bound)], 
                              index=[METHOD]).add_prefix("pred_")



    """
    Error in PDE operator (Dynamic Arbitrage)
    """
    
    PDE_err = bs_log_pde_err(moneyness, ttm, 
                         grads[:, f_to_i("ttm")], 
                         grads[:, f_to_i("log(S/K)")], 
                         hessian_moneyness[:, f_to_i("log(S/K)")])
    pde_stats = pd.DataFrame(diagnosis_pde(PDE_err), index = [METHOD]).add_prefix("PDE_")
    """
    Error in Greeks
    """
    N_FEATS = len(feat_names)
    true_first_order = X_df[[f"true_d_{x}" for x in feat_names]].values
    fig, ax = plt.subplots(ncols = N_FEATS, figsize=(5 * N_FEATS, 10), nrows=2)
    for i in range(N_FEATS):
        sns.scatterplot(x = X_df[feat_names[i]], y = grads[:, i], ax = ax[0, i])
        ax[0, i].set_title(feat_names[i])
        sns.scatterplot(x = X_df[feat_names[i]], y = true_first_order[:, i] - grads[:, i], ax = ax[1, i])
        ax[1, i].set_title(f"Error - {feat_names[i]}")
    
    fig, ax = plt.subplots(ncols = 2, figsize=(10, 5))
    true_second_order = X_df['true_d2_log(S/K)'].values
    ax[0].scatter(X_df["log(S/K)"], hessian_moneyness[:, f_to_i("log(S/K)")])
    ax[0].set_title("Gamma")
    ax[1].scatter(X_df["log(S/K)"], true_second_order - hessian_moneyness[:, f_to_i("log(S/K)")])
    ax[1].set_title(f"Error - Gamma")

    grad_stats = pd.DataFrame(diagnosis_grads(hessian_moneyness, grads, f_to_i, "ttm", "log(S/K)"), index=[METHOD])
    """
    Display Statistics
    """
    res = pd.concat([pred_stats, pde_stats, grad_stats], axis = 1)
    return res