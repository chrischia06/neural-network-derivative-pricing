import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.callbacks import TQDMProgressBar
"""
Neural Network Hyperparameters
"""
nn_params = {
    "hidden_units": 32,
    "n_layers": 4,
    "dropout_ratio": 0.0,
    "hidden_act": "softplus",
    "output_act": None,
    "batch_norm": True,
    "kernel_init": "lecun_normal",
    "seed": 42
}

### Compile parameters
LR = 1e-3
metric_names = ["MAE", "RMSE"]
METRICS = [
    tf.keras.metrics.MeanAbsoluteError(name="MAE"),
    tf.keras.metrics.RootMeanSquaredError(name="RMSE"),
]
compile_params = {"loss": tf.keras.losses.MeanSquaredError(), 
                    "metrics":METRICS}

### Fit parameters
CALLBACKS = [
    EarlyStopping(patience=10, restore_best_weights=True),
    TQDMProgressBar(show_epoch_progress=False),
   # ReduceLROnPlateau(patience=5),
]
fit_params = {
    "epochs": 30,
    "batch_size": 1024,
    "verbose": 0,
    "validation_split": 0.2,
    "shuffle": False,
    "callbacks": CALLBACKS
    
}