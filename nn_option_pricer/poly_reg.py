
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
import math
import time
import numpy as np

def train_infer_poly(degree, Xs_train, ys_train, Xs_test, ys_test, feat_names, all_models, all_model_preds, all_model_grads, all_model_hessian, f_to_i, moneyness_var, METHOD="poly_reg", eps=1e-4):
    start = time.time()    
    #spline = SplineTransformer(n_knots=5, degree=10, knots="uniform", extrapolation="linear")
    all_models[METHOD] = Pipeline([('poly', PolynomialFeatures(degree = degree)), 
                               ('lr', LinearRegression(fit_intercept=False))]).fit(Xs_train, ys_train)
    train_time = time.time() - start

    #inference
    start2 = time.time()
    all_model_preds[METHOD] = all_models[METHOD].predict(Xs_test)

    all_model_grads[METHOD] = np.zeros((Xs_test.shape[0], Xs_test.shape[1]))
    all_model_hessian[METHOD] = np.zeros((Xs_test.shape[0], Xs_test.shape[1]))

    for feat in [x for x in feat_names if x != moneyness_var] + [moneyness_var]:
        Xs_bumped = Xs_test.copy()
        Xs_bumped[:, f_to_i(feat)] += eps
        preds2 = all_models[METHOD].predict(Xs_bumped)
        all_model_grads[METHOD][:, f_to_i(feat)] = (preds2 - all_model_preds[METHOD]) / eps

    Xs_bumped2 = Xs_test.copy()
    for feat in [x for x in feat_names if x != moneyness_var] + [moneyness_var]:
        Xs_bumped2[:, f_to_i(moneyness_var)] -= eps
        preds3 = all_models[METHOD].predict(Xs_bumped2)
        all_model_hessian[METHOD][:, f_to_i(feat)] = (
            preds2 - 2 * all_model_preds[METHOD] + preds3
        ) / (eps * eps)
    inference_time = time.time() - start2 
    return train_time, inference_time