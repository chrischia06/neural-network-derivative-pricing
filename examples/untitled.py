import numpy as np

def diagnosis(true, pred:) -> dict:
    return {
        "l1": np.mean(np.abs(true - pred)),
        "l2": np.sqrt(np.mean((true - pred) ** 2)),
        "l_inf": np.max((np.abs(true-pred))),
        "neg_ratio": np.mean((pred < 0))
    }