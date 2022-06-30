import numpy as np
import pandas as pd
from typing import List
from utils import diagnosis_pred, diagnosis_grads, visualise_surface
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
plt.style.use("ggplot")


def bergomi_eval_wrapper(
    X_df: pd.DataFrame,
    true_val: np.array,
    preds: np.array,
    grads: np.array = None,
    hessian_moneyness: np.array = None,
    feat_names: List[str] = ["log-strike", "ttm", "alpha", "rho", "vol-of-vol"],
    lower_bound: np.array = None,
    upper_bound: np.array = None,
    METHOD: str = "standard_ffn",
    plots = ['pricing', 'gradients','grad_error']
):
    f_to_i = lambda x: feat_names.index(x)
    N_FEATS = len(feat_names) 
    temp = pd.concat(
        [
            diagnosis_pred(
                true_val,
                preds,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                method=METHOD,
            ),
            diagnosis_grads(
                hessian_moneyness, grads, f_to_i, "ttm", "log-strike", method=METHOD
            ),
        ],
        axis=1,
    )
    try:
        grad_error = grads[:,f_to_i("log-strike")] - X_df['MC_call_d_log-strike']
        temp['d/dk_l1'] = np.mean(np.abs(grad_error))
        temp['d/dk_l2'] = np.mean((grad_error) ** 2)
        temp['d/dk_max'] = np.max(np.abs(grad_error))
    except:
        pass
    try:
        hessian_error = hessian_moneyness[:,f_to_i("log-strike")] - X_df['MC_call_d2_log-strike']
        temp['d/dk^2_l1'] = np.mean(np.abs(hessian_error))
        temp['d/d^k2_l2'] = np.mean((hessian_error) ** 2)
        temp['d/d^k_max'] = np.max(np.abs(hessian_error))
    except:
        pass

    ### Plot Pricing Errror
    if 'pricing' in plots:
        fig, ax = plt.subplots(figsize=(14, 5), ncols=2)
        sns.scatterplot(
            X_df["log-strike"],
            true_val - preds,
            hue=X_df["ttm"],
            ax=ax[1],
        )
        sns.scatterplot(X_df["log-strike"], preds, hue=X_df["ttm"], ax=ax[0])
        for i in range(2):
            ax[0].set_xlabel("log-moneyness")
        ax[0].set_ylabel("Predicted Price")
        ax[1].set_ylabel("Pricing Error vs MC Price")
        ax[0].set_title(
            f"{METHOD}: Predicted Price against Moneyness\nColour: time-to-maturity"
        )
        ax[1].set_title(
            f"{METHOD}: Pricing Error against Moneyness\nColour: time-to-maturity"
        )

    ### Plot Greeks
    if 'gradients' in plots:
        fig, ax = plt.subplots(ncols=N_FEATS, nrows = 1, figsize=((N_FEATS + 1)  * 7, 6))
        for i, x in enumerate(feat_names):
            sns.scatterplot(
                X_df["log-strike"],
                grads[:, f_to_i(x)],
                hue=X_df["ttm"],
                ax=ax[i],
            )
            ax[i].set_title(
                f"{METHOD} - Gradient wrt {x} vs log-strike\nColour: time-to-maturity"
            )
            ax[i].set_ylabel(f"Sensitivity wrt {x}")
            ax[i].set_xlabel("log-strike")

        sns.scatterplot(
            X_df["log-strike"],
            hessian_moneyness[:, f_to_i("log-strike")],
            ax=ax[-1],
            hue=X_df["ttm"],
        )
        ax[-1].set_title(f"{METHOD} - Hessian wrt {x} vs log-strike\nColour: time-to-maturity")


    if 'grad_error' in plots:
        ### PLOT gradient errors
        fig, ax = plt.subplots(ncols = 2, figsize=(12, 5))
        sns.scatterplot(
            X_df["log-strike"],
            X_df[f'MC_call_d_log-strike'] - grads[:, f_to_i("log-strike")],
            hue=X_df["ttm"],
            ax=ax[0],
        )
        ax[0].set_title(
            f"{METHOD} - Gradient Error wrt log-strike, vs log-strike\nColour: time-to-maturity"
        )
        ax[0].set_ylabel(f"Gradient Error wrt log-strike")
        ax[0].set_xlabel("log-strike")

        ax[1].set_title(
        f"{METHOD}\nError in Hessian (Gamma) wrt log-moneyness\n vs log-strike\nColour: time-to-maturity"
        )
        ax[1].set_ylabel("Gamma")
        sns.scatterplot(
        X_df["log-strike"],
        X_df[f"MC_call_d2_{'log-strike'}"] - hessian_moneyness[:, f_to_i("log-strike")],
        hue=X_df["ttm"], ax = ax[1])

        for i in range(2):
            ax[i].set_xlabel("log-strike")
    return temp


def bergomi_model_inference(all_models, all_model_preds, METHOD, all_model_grads, all_model_hessian, X_df_test, Xs_test, true, feat_names, f_to_i, intrinsic_val, upper_bound, eval_batch_size = 10 ** 5):
    start2 = time.time()
    model = all_models[METHOD]
    all_model_preds[METHOD] = model.predict(Xs_test, batch_size=eval_batch_size).reshape(-1)
    X_tensor = tf.Variable(Xs_test)
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape:
            output = model(X_tensor)
            grads = tape.gradient(output, X_tensor)
        hessian1 = tape2.gradient(grads[:, f_to_i("log-strike")], X_tensor)

    all_model_grads[METHOD] = grads.numpy()
    all_model_hessian[METHOD] = hessian1.numpy()
    inference_time = time.time() - start2

    """
    Evaluate Predictions, Sensitivities
    """
    N_FEATS = len(feat_names)
    temp = bergomi_eval_wrapper(
        X_df_test,
        true,
        all_model_preds[METHOD],
        all_model_grads[METHOD],
        all_model_hessian[METHOD],
        feat_names,
        lower_bound=intrinsic_val,
        upper_bound = upper_bound,
        METHOD=METHOD
    )

    temp["model_parameters"] = all_models[METHOD].count_params()
    temp["inference_time"] = inference_time


    """
    Visualise call surface
    """
    N_POINTS = 128
    SK = np.linspace(-3.0, 3, N_POINTS)
    ts = np.linspace(0, 2.0, N_POINTS)
    X = np.zeros((N_POINTS ** 2, N_FEATS))
    X[:, :2] = np.array(list(product(SK, ts)))
    sample_params = X_df_test.sample(1).iloc[0].to_dict()
    for x in [i for i in feat_names if i not in ["ttm", "log-strike"]]:
        X[:, f_to_i(x)] = sample_params[x] * np.ones(X.shape[0])
    visualise_surface(SK, ts, all_models[METHOD](X).numpy(), title=f"{METHOD} -Call Surface").show()
    return temp
