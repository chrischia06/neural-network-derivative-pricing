import numpy as np
import pandas as pd
from typing import List
from utils import diagnosis_pred, diagnosis_grads


def bergomi_eval_wrapper(
    X_df: pd.DataFrame,
    true_val: np.array,
    preds: np.array,
    grads: np.array,
    hessian_moneyness: np.array,
    feat_names: List[str] = ["log-strike", "ttm", "alpha", "rho", "vol-of-vol"],
    lower_bound: np.array = None,
    upper_bound: np.array = None,
    METHOD: str = "standard_ffn",
):
    f_to_i = lambda x: feat_names.index(x)
    temp = pd.concat(
        [
            diagnosis_pred(true_val, preds, lower_bound=lower_bound, upper_bound = upper_bound, method=METHOD),
            diagnosis_grads(
                hessian_moneyness, grads, f_to_i, "ttm", "log-strike", method=METHOD
            ),
        ],
        axis=1,
    )
    
    """
    Pricing Error
    """
    fig, ax = plt.subplots(figsize=(12, 5), ncols=2)
    sns.scatterplot(
        X_df["log-strike"],
        true - preds,
        hue=X_df["ttm"],
        ax=ax[1],
    )
    sns.scatterplot(
        X_df["log-strike"], preds, hue=X_df["ttm"], ax=ax[0]
    )

    ax[0].set_xlabel("log-moneyness")
    ax[1].set_xlabel("log-moneyness")
    ax[0].set_ylabel("Predicted Price")
    ax[1].set_ylabel("Pricing Error vs MC Price")
    ax[0].set_title(f"{METHOD}: Predicted Price against Moneyness\nColour: time-to-maturity)")
    ax[1].set_title(f"{METHOD}: Pricing Error against Moneyness\nColour: time-to-maturity)")
    
    """
    Greeks
    """
    fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
    for i, x in enumerate(["log-strike", "ttm"]):
        sns.scatterplot(
            X_df["log-strike"],
            grads[:, f_to_i(x)],
            hue=X_df["ttm"],
            ax=ax[i],
        )
        ax[i].set_title(f"{METHOD}\nSensitivity wrt {x}\nagainst log-moneyness\nColour: time-to-maturity")
        ax[i].set_ylabel(f"Sensitivity wrt {x}")
    sns.scatterplot(
        X_df["log-strike"],
        hessian_moneyness[:, f_to_i("log-strike")],
        ax=ax[2],
        hue=X_df["ttm"],
    )
    ax[2].set_title(f"{METHOD}\nHessian (Gamma) wrt log-moneyness\nagainst log-moneyness\nColour: time-to-maturity")
    ax[2].set_ylabel("Gamma")
    return temp

# """
# Define Neural Network
# """
# dataset = tf.data.Dataset.from_tensor_slices((Xs_train, ys_train, true_grads_train))
# opt = Adam(learning_rate=LR)
# model = make_model(
#     N_FEATS, HIDDEN_UNITS, LAYERS, DROPOUT_RATIO, HIDDEN_ACT, OUTPUT_ACT, BATCH_NORM
# )

# batched_dataset = dataset.batch(BATCH_SIZE)
# METHOD = "differential"


# @tf.function
# def train(y, true_grad, x_var):
#     with tf.GradientTape() as model_tape:
#         with tf.GradientTape() as grad_tape:
#             output = model(x_var)
#         gradients = grad_tape.gradient(output, x_var)
#         grad_loss = tf.keras.losses.MeanSquaredError()(true_grad, gradients[:, 0])
#         pred_loss = tf.keras.losses.MeanSquaredError()(output, y)
#         loss = grad_loss + pred_loss
#         model_grad = model_tape.gradient(loss, model.trainable_variables)
#         opt.apply_gradients(zip(model_grad, model.trainable_variables))
#     return loss, pred_loss, grad_loss


# losses = {
#     "grad": [None for i in range(EPOCHS)],
#     "loss": [None for i in range(EPOCHS)],
#     "pred": [None for i in range(EPOCHS)],
# }
# start = time.time()
# for epoch in tqdm(range(EPOCHS)):
#     temp_pred = []
#     temp_grad = []
#     temp_loss = []
#     for step, (x, y_true, true_grad) in enumerate(batched_dataset):
#         x_var = tf.Variable(x)
#         loss, pred_loss, grad_loss = train(y_true, true_grad, x_var)
#         temp_pred += [pred_loss.numpy()]
#         temp_grad += [grad_loss.numpy()]
#         temp_loss += [loss.numpy()]
#     losses["grad"][epoch] = [np.mean(temp_grad)]
#     losses["pred"][epoch] = [np.mean(temp_pred)]
#     losses["loss"][epoch] = [np.mean(temp_loss)]
# train_time = time.time() - start

# all_models[METHOD] = model
# fig, ax = plt.subplots(figsize=(15, 5), ncols=3)
# for i, metric in enumerate(losses.keys()):
#     ax[i].plot(losses[metric])
#     ax[i].set_yscale("log")
#     ax[i].set_title(metric)