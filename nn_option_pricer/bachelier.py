import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.random import default_rng
from utils import diagnosis_pred
from numpy.linalg import cholesky
from jax import grad
from jax.numpy import vectorize


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
    plots = ["prediction", "gradient"]
) -> pd.DataFrame:
    
    N_ASSETS = len([x for x in X_df.columns if x.find("asset_") == 0])
    """
    1. Prediction Errors
    """
    if "prediction" in plots:
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        # sns.scatterplot(
        #     X_df["basket"], X_df["call_payoff"], label="Sample Payoffs", ax=ax[0]
        # )
        sns.scatterplot(X_df["basket"], preds, label="Predicted", ax=ax[0], alpha = 0.8)
        idx = np.argsort(X_df['basket'])
        sns.lineplot(X_df["basket"].iloc[idx], X_df["call_analytic"].iloc[idx], label="Analytic", ax=ax[0], color = "purple", linewidth=3.0)
        if lower_bound is not None:
            sns.lineplot(
                X_df["basket"].iloc[idx],
                lower_bound[idx],
                label="Lower Bound",
                linestyle="--",
                color="blue",
                ax=ax[0],
            )
        if upper_bound is not None:
            sns.lineplot(
                X_df["basket"].iloc[idx],
                upper_bound[idx],
                label="Upper Bound",
                linestyle="--",
                color="blue",
                ax=ax[0],
            )

        ax[0].legend()
        ax[0].set_title(f"{METHOD} - Predicted Basket Call Option\n vs Basket value ({N_ASSETS} Assets)")
        ax[0].set_xlabel("Basket Value (Average of {N_ASSETS} Assets)")
        ax[0].set_ylabel(f"Basket Call Value")

        sns.scatterplot(X_df["basket"], preds - X_df["call_analytic"], ax=ax[1], alpha = 0.8)

        ax[1].set_title(f"{METHOD} - Price Prediction Error vs Basket Value")
        ax[1].set_xlabel("Basket Value")
        ax[1].set_ylabel("Pricing Error")

    pred_stats = diagnosis_pred(
        X_df["call_analytic"].values, preds, lower_bound, upper_bound, method=METHOD
    ).add_prefix("pred_")


    """
    2. Error in Greeks
    """
    
    true_factor = X_df["call_analytic_delta"] / N_ASSETS
    grad_stats = diagnosis_pred(
        true_factor, grads, lower_bound=0, method=METHOD
    ).add_prefix("grad_")
    
    if "gradient" in plots:
        ## Greek Plots
        fig, ax = plt.subplots(ncols=2, figsize=(12, 5))
        sns.scatterplot(X_df["basket"], grads, ax = ax[0], alpha = 0.8)
        ax[0].set_title("Predicted Gradient")

        sns.scatterplot(x = X_df["basket"], 
                        y = X_df["call_analytic_delta"] / N_ASSETS - grads,
                        ax = ax[1],
                       alpha = 0.8)
        ax[1].set_title(f"{METHOD} Gradient Error vs Basket value")
        ax[1].set_xlabel(f"Basket Value (Average of {N_ASSETS} underlyings)")
        ax[1].set_ylabel(f"Gradient Error")
        
#     """
#     3. Error in PDE operator (Dynamic Arbitrage)
#     """

#     PDE_err = bs_log_pde_err(moneyness, ttm,
#                          grads[:, f_to_i("ttm")],
#                          grads[:, f_to_i("log(S/K)")],
#                          hessian_moneyness[:, f_to_i("log(S/K)")])
#     pde_stats = pd.DataFrame(diagnosis_pde(PDE_err), index = [METHOD]).add_prefix("PDE_")
    """
    Display Statistics
    """
    res = pd.concat([pred_stats, grad_stats], axis=1)
    return res


def make_bachelier_dataset(N_SAMPLES, N_ASSETS, F, SEED, T1, T, K, S0, L, w):
    """
    Define Brownian Motion
    """
    # Simulate St, ST
    rng = default_rng(SEED)
    Wt = np.sqrt(T1) * rng.standard_normal((N_SAMPLES, F)) @ L.T
    St = S0 + Wt
    WT = (np.sqrt(T) * rng.standard_normal((N_SAMPLES, F))) @ L.T

    """
    Calculate Sample Payoffs and Gradients
    """
    XT = (St + WT) @ w
    ys = payoff(XT, K)
    payoff_grad = vectorize(grad(payoff))(XT, K)

    grads = np.zeros((N_SAMPLES, N_ASSETS))
    for i in tqdm(range(N_SAMPLES)):
        grads[i, :] = grad(abm_step)(St[i, :], WT[i, :], w)
    grads = grads * payoff_grad.reshape((-1, 1))

    assert (ys.shape[0] == grads.shape[0]) & (grads.shape[0] == St.shape[0])

    X_df = pd.concat(
        [
            pd.DataFrame(St).add_prefix("asset_"),
            pd.DataFrame(grads).add_prefix("grad_asset"),
        ],
        axis=1,
    )
    X_df["call_payoff"] = ys

    """
    Compute analytic price and all greeks
    """
    X_df["basket"] = St @ w
    sigma = np.sqrt(w @ L @ L.T @ w.T)

    X_df["call_analytic"] = bachelier_solution(X_df["basket"].values, K, sigma, T)
    X_df["call_analytic_theta"] = vectorize(grad(bachelier_solution, argnums=3))(
        X_df["basket"].values, K, sigma, T
    )
    X_df["call_analytic_vega"] = vectorize(grad(bachelier_solution, argnums=2))(
        X_df["basket"].values, K, sigma, T
    )
    X_df["call_analytic_delta"] = vectorize(grad(bachelier_solution, argnums=0))(
        X_df["basket"].values, K, sigma, T
    )
    X_df["call_analytic_gamma"] = vectorize(grad(grad(bachelier_solution, argnums=0)))(
        X_df["basket"].values, K, sigma, T
    )
    
    return X_df, WT, St

# dataset = tf.data.Dataset.from_tensor_slices((Xs, ys, grads))
# l = 1
# batched_dataset = dataset.batch(BATCH_SIZE)
# losses = {"grad": [], "total": [], "pred": []}

# """
# Differential
# """
# # loss_fn = tf.keras.losses.MeanSquaredError()

# # METRICS = [tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.RootMeanSquaredError()]
# # VAL_SPLIT = 0.2
# # CALLBACKS = [EarlyStopping(patience=5)]


# opt = Adam(learning_rate=LR)
# grad_model = make_model(
#     N_FEATS, HIDDEN_UNITS, LAYERS, DROPOUT_RATIO, HIDDEN_ACT, OUTPUT_ACT, BATCH_NORM
# )


# @tf.function
# def train_loop(x_batch, y_batch, grads_batch):
#     with tf.GradientTape() as loss_tape:
#         with tf.GradientTape() as model_tape:
#             model_tape.watch(x_batch)
#             output = grad_model(x_batch)
#             model_grads = model_tape.gradient(output, x_batch)
#         loss_weight = 1.0
#         pred_loss = tf.reduce_mean(
#             tf.keras.losses.MeanSquaredError()(
#                 output * loss_weight, y_batch * loss_weight
#             )
#         )
#         grad_loss = tf.reduce_mean(
#             tf.math.reduce_sum((grads_batch - model_grads) ** 2, axis=1) * loss_weight
#         )
#         total_loss = pred_loss + l * grad_loss

#         loss_grad = loss_tape.gradient(total_loss, grad_model.trainable_variables)
#         opt.apply_gradients(zip(loss_grad, grad_model.trainable_variables))
#     return grad_loss, pred_loss, total_loss


# start = time.time()
# for epoch in tqdm(range(EPOCHS)):
#     temp_grad = []
#     temp_pred = []
#     temp_total = []
#     for (x_batch, y_batch, grads_batch) in batched_dataset:
#         # x_batch = tf.Variable(x_batch)

#         grad_loss, pred_loss, total_loss = train_loop(x_batch, y_batch, grads_batch)
#         temp_grad += [grad_loss.numpy()]
#         temp_pred += [pred_loss.numpy()]
#         temp_total += [total_loss.numpy()]

#     losses["grad"] += [np.mean(temp_grad)]
#     losses["pred"] += [np.mean(temp_pred)]
#     losses["total"] += [np.mean(temp_total)]
# end = time.time()
# """
# Plot loss curves
# """
# fig, ax = plt.subplots(ncols=3, figsize=(15, 4))
# for i, x in enumerate(["grad", "pred", "total"]):
#     ax[i].plot(losses[x])
#     ax[i].set_title(f"Loss - {x}")
#     ax[i].set_yscale("log")

"""
TODO: investigate QMC based sampling and Hessian
"""
# from numpy.linalg import cholesky
# SEED = 42
# rng = default_rng(SEED)

# """
# Define Parameters
# """
# N_ASSETS = 100
# F = N_ASSETS
# N_SAMPLES = 10 ** 4
# T = 1.0
# K = 1.0

# # Covariance matrix
# L = 0.2 * rng.standard_normal((N_ASSETS, F))
# cov = (L @ L.T)
# assert np.linalg.det(cov) > 0
# L = cholesky(cov)

# with tf.GradientTape() as tape2:
#     with tf.GradientTape() as model_tape:
#         model_tape.watch(x_batch)
#         output = grad_model(x_batch)
#         grads = model_tape.gradient(output, x_batch)
#     jacobian = tape2.batch_jacobian(grads, x_batch)

# j_sum = tf.reduce_sum(jacobian, axis=2)
# print(j_sum.shape)
# j_select = tf.einsum('bxby->bxy', jacobian)


# with tf.GradientTape() as tape2:
#     with tf.GradientTape() as model_tape:
#         output = model(X_tensor)
#         model_grads = model_tape.gradient(output, X_tensor)
#     jacobian = tape2.batch_jacobian(model_grads, X_tensor)
# j_sum = tf.reduce_sum(jacobian, axis=2)
# hessian_det = tf.linalg.trace(L.T @ jacobian @ L)
# factor_grad = tf.math.reduce_mean(model_grads, axis = 1).numpy()
# sns.scatterplot(X_df['basket'], hessian_det)
