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

    pred_stats = diagnosis_pred(
        X_df["call_analytic"].values, preds, lower_bound, upper_bound, method=METHOD
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
    grad_stats = diagnosis_pred(
        true_factor, grads, lower_bound=0, method=METHOD
    ).add_prefix("grad_")
    """
    Display Statistics
    """
    res = pd.concat([pred_stats, grad_stats], axis=1)
    return res


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
