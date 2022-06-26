import tensorflow as tf
from tensorflow.keras.layers import (
    Add,
    Dense,
    Input,
    Concatenate,
    BatchNormalization,
    Multiply,
)
from tensorflow.keras import Model, layers
from typing import List, Union, Callable
import tensorflow_probability as tfp
import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def make_model(
    n_feats: int,
    hidden_units: int,
    n_layers: int,
    dropout_ratio: float,
    hidden_act: Union[str, Callable],
    output_act: Union[str, Callable],
    batch_norm: bool,
    kernel_init: str = "lecun_normal",
    seed: int = 42,
    resnet:bool = False,
    output_dim:int = 1
) -> Model:
    tf.random.set_seed(seed)
    input_layer = Input(n_feats)
    if batch_norm:
        x = BatchNormalization()(input_layer)
        x = Dense(hidden_units, activation=hidden_act, kernel_initializer=kernel_init)(
            x
        )
        x = BatchNormalization()(x)
    else:
        x = Dense(hidden_units, activation=hidden_act, kernel_initializer=kernel_init)(
            input_layer
        )
    for i in range(n_layers - 1):
        if resnet:
            x = Add()([Dense(
                hidden_units,
                activation=hidden_act,
                kernel_initializer=kernel_init,
                kernel_constraint=None,
            )(x), x])
        else:
            x = Dense(
                hidden_units,
                activation=hidden_act,
                kernel_initializer=kernel_init,
                kernel_constraint=None,
            )(x)     
        if batch_norm:
            x = BatchNormalization()(x)
        if dropout_ratio > 0:
            x = Dropout(dropout_ratio)(x)
    output_layer = Dense(output_dim, activation=output_act, kernel_initializer=kernel_init)(x)
    ffn = Model(input_layer, output_layer)
    return ffn


# def ffn_network(n_feats: int, n_layers:int, hidden_layer, output_layer, seed:int = 42) -> tf.keras.Model:
#     input_layer =  Input(n_feats)
#     x = hidden_layer()(input_layer)
#     for i in range(n_layers - 1):
#         x = hidden_layer()(x)
#     final = output_layer()(x)
#     return Model(input_layer, final)


class SumLayer(layers.Layer):
    def __init__(self):
        super(SumLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)


class MeanLayer(layers.Layer):
    def __init__(self):
        super(MeanLayer, self).__init__()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)




def ensemble_model(models: List, N_SEEDS: int) -> tf.keras.Model:
    x = Concatenate()([models[i].output for i in range(N_SEEDS)])
    x = MeanLayer()(x)
    model = Model([models[i].input for i in range(N_SEEDS)], x)
    return model


def rbf(x):
    return tf.math.exp(-0.5 * x**2)


def homogeneity_network(
    n_feats: int,
    hidden_units: int = 100,
    batch_norm: bool = True,
    dropout_ratio:float = 0.0,
    kernel_init: str = "lecun_normal",
    seed: int = 42,
):
    """
    n_feats: No. of inputs, at least moneyness and time-to-maturity are required
    hidden_units: No. of hidden units
    SEED: Hidden Seed

    Constrains the moneyness and time-to-maturity variables to have positive weights, and no constraints for the other paramters
    """
    
    assert n_feats >= 2
    tf.random.set_seed(seed)
    input_layer = Input(n_feats)
    if batch_norm:
        input_layer2 = BatchNormalization()(input_layer)
    else:
        input_layer2 = input_layer
    input_moneyness = Dense(1, name='moneyness')(input_layer2)
    input_ttm = Dense(1, name='ttm')(input_layer2)
    input_ttm.trainable = False
    input_moneyness.trainable = False
    
    x1 = Dense(
        hidden_units,
        activation="sigmoid",
        kernel_constraint="non_neg",
        kernel_initializer=kernel_init,
    )(input_ttm)
    x2 = Dense(
        hidden_units,
        activation="softplus",
        kernel_constraint="non_neg",
        kernel_initializer=kernel_init,
    )(input_moneyness)
    interaction_layers = [x1, x2]
    if n_feats > 2:
        input_other = Dense(n_feats - 2, name = "other_feats")(input_layer2)
        input_other.trainable = False
        x3 = Dense(
            hidden_units,
            activation="softplus",
            kernel_initializer=kernel_init,
        )(input_other)
        interaction_layers += [x3]
    x = Multiply()(interaction_layers)
    if dropout_ratio > 0.0:
        x = Dropout()(x)
    output_layer = Dense(
        1, kernel_constraint="non_neg", use_bias=False, kernel_initializer=kernel_init
    )(x)
    model = Model(input_layer, output_layer)
    return model

def train_nn(model, Xs, ys, fit_params, metric_names):
    start = time.time()
    history = model.fit(Xs, ys, **fit_params)
    train_time = time.time() - start

    fig, ax = plt.subplots(figsize=(len(metric_names) * 6, 5), ncols=len(metric_names))
    for i, metric in enumerate(metric_names):
        ax[i].plot(history.history[metric], label="train")
        ax[i].plot(history.history[f"val_{metric}"], label="val")
        ax[i].legend()
        ax[i].set_title(f"{metric} vs number of Epochs")
        ax[i].set_xlabel("Epochs")
        ax[i].set_ylabel(f"{metric}, log-scale")
        ax[i].set_yscale("log")
    return train_time, history

# def gated_network(n_feats) -> tf.keras.Model:
#     input_layer = Input(n_feats)
#     x = Dense(
#         100,
#         activation="softplus",
#         kernel_constraint="non_neg",
#         kernel_initializer="lecun_normal",
#     )(input_layer)
#     weights = Dense(
#         100,
#         activation="softmax",
#         kernel_constraint="non_neg",
#         kernel_initializer="lecun_normal",
#     )(x)
#     x2 = Dense(
#         100,
#         activation="sigmoid",
#         kernel_constraint="non_neg",
#         kernel_initializer="lecun_normal",
#     )(x)
#     mult = Multiply()([weights, x2])
#     output_layer = SumLayer()(mult)
#     model = Model(input_layer, output_layer)
#     return model



# def homogeneity_network(n_feats):
#     input_moneyness = Input(1)
#     input_ttm = Input(1)

#     x1 = Dense(1, activation='sigmoid', kernel_constraint='non_neg')(input_ttm)
#     x2 = Dense(1, activation='softplus', kernel_constraint='non_neg')(input_moneyness)
#     input_layers = [input_ttm, input_moneyness]
#     interaction_layers = [x1, x2]
#     if n_feats > 2:
#         input_other = Input(n_feats - 2)
#         input_layers += [input_other]
#         x3 = Dense(100, 'softplus')(input_layers[-1])
#         x3 = Dense(1, activation='sigmoid')(x3)
#         x3 = Dense(1, kernel_constraint='non_neg', use_bias=False)(x3)
#         interaction_layers += [x3]

#     x = Multiply()(interaction_layers)
#     output_layer = Dense(1, kernel_constraint='non_neg', use_bias=False)(x)
#     model = Model(input_layers, output_layer)
#     # x3 = Dense(1, activation='softplus', kernel_constraint='non_neg')(input_moneyness)
#     # x4 = Multiply()([x2, x3])

#     # x2 = Dense(1, activation='exponential', name='intrinsic')(input_layer)
#     # x3 = Lambda(lambda x: tf.keras.activations.relu(x - 1))(x2)
#     # x = Dense(32, activation='softplus', kernel_constraint='non_neg')(x)
#     # x = Dense(32, activation='softplus', kernel_constraint='non_neg')(x)
#     # output_layer = Add()([x3, Dense(1, activation='softplus')(x)])

#     # model.layers

#     # model.layers[1].trainable = False
#     # model.layers[1].set_weights([np.array([[1], [0]]), np.array([0.0])])
#     return model


# def create_model(n_feats:int = 1,
#                  activation_func:Union[Callable, str] = 'elu',
#                  kernel_init:Union[Callable,str] = 'glorot_normal',
#                  n_layers:int = 1,
#                  N_UNITS:int = 100,
#                 SEED = 42):
#     tf.random.set_seed(42)
#     input_layer = Input(n_feats, name="input_layer")
#     layers = [Dense(N_UNITS,
#                     activation=activation_func,
#                     kernel_initializer = kernel_init
#                    )(input_layer)]

#     for i in range(1, n_layers):
#         layers += [Dense(N_UNITS,
#                     activation=activation_func,
#                     kernel_initializer = kernel_init
#                    )(layers[-1])]
#     output_layer = Dense(1, name="output_layer")(layers[-1])

#     model = Model(input_layer, output_layer)
#     return model

# model = create_model(n_feats = 10, N_UNITS = 30, n_layers = 5)
# opt = tf.keras.optimizers.Adam(learning_rate = 1e-2)

# def custom_loss_pass(model, x_tensor):
#     def custom_loss(y_true,y_pred):
#         with tf.GradientTape() as t:
#             t.watch(x_tensor)
#             output = model(x_tensor)
#         grad = t.gradient(output, x_tensor)
#         # loss_data = tf.reduce_mean(tf.square(yTrue - yPred), axis=-1)
#         loss = tf.reduce_mean(tf.square(output - y_true[:, :1])) + l * tf.reduce_mean(tf.square(y_true[:, 1:] - grad))
#         return loss
#     return custom_loss


# model.compile(loss=custom_loss_pass(model, tf.convert_to_tensor(S0.numpy(), dtype=tf.float32)), optimizer=opt)
# model.fit(tf.convert_to_tensor(S0.numpy(), dtype=tf.float32),
#           tf.convert_to_tensor(np.hstack([y.numpy(), grads]), dtype=tf.float32), batch_size=1024, epochs=100, shuffle=False)


def delta_loss(grad_true, grad_pred):
    return tf.keras.losses.MeanSquaredError()(grad_true, grad_pred[:, 0])


class DifferentialModel(tf.keras.Model):
    """
    Wrapper to enable differential training
    """
    def set_params(self, lam = 1, grad_loss = delta_loss):
        self.lam = 1
        self.grad_loss = delta_loss
        self.loss_tracker_grad = tf.keras.metrics.Mean(name="grad_loss")
        self.loss_tracker_pred = tf.keras.metrics.Mean(name="pred_loss")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    
    @tf.function
    def train_step(self, data):
        x_var, (y, true_grad) = data
        # https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        with tf.GradientTape() as model_tape:
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(x_var)
                model_pred = self(x_var, training=True)
            gradients = grad_tape.gradient(model_pred, x_var)
            grad_loss = self.grad_loss(true_grad, gradients)
            pred_loss = self.compiled_loss(y, model_pred)
            loss = self.lam * grad_loss + pred_loss
        trainable_vars = self.trainable_variables
        model_grad = model_tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(model_grad, trainable_vars))
        self.compiled_metrics.update_state(y, model_pred)
        self.loss_tracker.update_state(loss)
        self.loss_tracker_grad.update_state(grad_loss)
        self.loss_tracker_pred.update_state(pred_loss)
        metrics_to_ret = {m.name: m.result() for m in self.metrics}
        metrics_to_ret['loss'] = self.loss_tracker.result()
        metrics_to_ret['pred_loss'] = self.loss_tracker_pred.result()
        metrics_to_ret['grad_loss'] = self.loss_tracker_grad.result()
        return metrics_to_ret
    @tf.function
    def test_step(self, data):
        x_var, (y, true_grad) = data
        # https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        with tf.GradientTape() as grad_tape:
            grad_tape.watch(x_var)
            model_pred = self(x_var, training=False)
            gradients = grad_tape.gradient(model_pred, x_var)
            grad_loss = self.grad_loss(true_grad, gradients)
            pred_loss = self.compiled_loss(y, model_pred)
            loss = self.lam * grad_loss + pred_loss
        self.compiled_metrics.update_state(y, model_pred)
        self.loss_tracker.update_state(loss)
        self.loss_tracker_grad.update_state(grad_loss)
        self.loss_tracker_pred.update_state(pred_loss)
        metrics_to_ret = {f"{m.name}": m.result() for m in self.metrics}
        metrics_to_ret['loss'] = self.loss_tracker.result()
        metrics_to_ret['pred_loss'] = self.loss_tracker_pred.result()
        metrics_to_ret['grad_loss'] = self.loss_tracker_grad.result()
        return metrics_to_ret
   

# dataset = tf.data.Dataset.from_tensor_slices((Xs_train, ys_train, true_grads_train))
# opt = Adam(learning_rate=LR)
# model = make_model(**nn_params)

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
#     ax[i].set_title(metric)
    
class PDEModel(tf.keras.Model):
    """
    Wrapper to enable differential training
    """
    def set_params(self, lam = 1, pde_loss = None):
        self.lam = lam
        self.pde_loss = pde_loss
        self.loss_tracker_pde = tf.keras.metrics.Mean(name="pde_loss")
        self.loss_tracker_pred = tf.keras.metrics.Mean(name="pred_loss")
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
    @tf.function
    def train_step(self, data):
        x_var, y = data
        # https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        with tf.GradientTape() as model_tape:
            with tf.GradientTape() as hessian_tape:
                with tf.GradientTape() as grad_tape:
                    grad_tape.watch(x_var)
                    hessian_tape.watch(x_var)
                    model_pred = self(x_var, training = True)
                    gradients = grad_tape.gradient(model_pred, x_var)
                    hessian = hessian_tape.gradient(gradients[:, 0], x_var)
                    pde_loss = tf.math.reduce_mean(
                        tf.math.abs(gradients[:, 1] + x_var[:, 1] * (-hessian[:, 0] + gradients[:, 0]))
                    )
                pred_loss = tf.keras.losses.MeanSquaredError()(model_pred, y)
                loss = pred_loss + self.lam * pde_loss
        trainable_vars = self.trainable_variables
        model_grad = model_tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(model_grad, trainable_vars))
        self.compiled_metrics.update_state(y, model_pred)
        self.loss_tracker.update_state(loss)
        self.loss_tracker_pde.update_state(pde_loss)
        self.loss_tracker_pred.update_state(pred_loss)
        metrics_to_ret = {m.name: m.result() for m in self.metrics}
        metrics_to_ret['loss'] = self.loss_tracker.result()
        metrics_to_ret['pred_loss'] = self.loss_tracker_pred.result()
        metrics_to_ret['pde_loss'] = self.loss_tracker_pde.result()
        return metrics_to_ret
    
    @tf.function
    def test_step(self, data):
        x_var, y = data
        # https://keras.io/guides/customizing_what_happens_in_fit/#going-lowerlevel
        with tf.GradientTape() as hessian_tape:
            with tf.GradientTape() as grad_tape:
                grad_tape.watch(x_var)
                hessian_tape.watch(x_var)
                model_pred = self(x_var, training = True)
                gradients = grad_tape.gradient(model_pred, x_var)
                hessian = hessian_tape.gradient(gradients[:, 0], x_var)
                pde_loss = tf.math.reduce_mean(tf.math.abs(
                    (gradients[:, 1] + x_var[:, 1] * (-hessian[:, 0] + gradients[:, 0])))
                )
            pred_loss = tf.keras.losses.MeanSquaredError()(model_pred, y)
            loss = pred_loss + self.lam * pde_loss
        self.compiled_metrics.update_state(y, model_pred)
        self.loss_tracker.update_state(loss)
        self.loss_tracker_pde.update_state(pde_loss)
        self.loss_tracker_pred.update_state(pred_loss)
        metrics_to_ret = {f"{m.name}": m.result() for m in self.metrics}
        metrics_to_ret['loss'] = self.loss_tracker.result()
        metrics_to_ret['pred_loss'] = self.loss_tracker_pred.result()
        metrics_to_ret['pde_loss'] = self.loss_tracker_pde.result()
        return metrics_to_ret
    
    
# """
# Define Neural Network
# """
# opt = Adam(learning_rate=LR)
# METHOD = "ffn+pde"
# model = make_model(
#     N_FEATS, HIDDEN_UNITS, LAYERS, DROPOUT_RATIO, HIDDEN_ACT, OUTPUT_ACT, BATCH_NORM
# )
# dataset = tf.data.Dataset.from_tensor_slices((Xs_train, ys_train))
# batched_dataset = dataset.batch(BATCH_SIZE)


# @tf.function
# def train(y, x_var):
#     with tf.GradientTape() as model_tape:
#         with tf.GradientTape() as hessian_tape:
#             with tf.GradientTape() as grad_tape:
#                 output = model(x_var)
#             gradients = grad_tape.gradient(output, x_var)
#             hessian = hessian_tape.gradient(gradients[:, 0], x_var)
#             pde_loss = tf.math.reduce_mean(
#                 (gradients[:, 1] + x_var[:, 1] * (-hessian[:, 0] + gradients[:, 0]))
#                 ** 2
#             )
#             pred_loss = tf.keras.losses.MeanSquaredError()(output, y)
#             loss = pde_loss + pred_loss
#             model_grad = model_tape.gradient(loss, model.trainable_variables)
#             opt.apply_gradients(zip(model_grad, model.trainable_variables))
#     return loss, pred_loss, pde_loss


# losses = {"pde": [], "loss": [], "pred": []}
# start = time.time()
# for epoch in tqdm(range(EPOCHS)):
#     temp_pred = []
#     temp_pde = []
#     temp_loss = []
#     for step, (x, y_true) in enumerate(batched_dataset):
#         x_var = tf.Variable(x)
#         loss, pred_loss, pde_loss = train(y_true, x_var)
#         temp_pred += [pred_loss.numpy()]
#         temp_pde += [pde_loss.numpy()]
#         temp_loss += [loss.numpy()]
#     losses["pde"] += [np.mean(temp_pde)]
#     losses["pred"] += [np.mean(temp_pred)]
#     losses["loss"] += [np.mean(temp_loss)]
# train_time = time.time() - start

# all_models[METHOD] = model
# fig, ax = plt.subplots(figsize=(15, 5), ncols=3)
# for i, metric in enumerate(losses.keys()):
#     ax[i].plot(losses[metric])
#     ax[i].set_title(metric)


def gated_network_instantiate(model, f_to_i, feat_map ={"ttm":"ttm", "moneyness":"log(S/K)"}):  
    for x in model.layers:
        if x.name == "ttm":
            weights = x.get_weights()
            weights[0] = np.zeros(weights[0].shape)
            weights[0][f_to_i(feat_map["ttm"])] = 1.0
            x.set_weights(weights)
            x.trainable = False
        if x.name == "moneyness":
            weights = x.get_weights()
            weights[0] = np.zeros(weights[0].shape)
            weights[0][f_to_i(feat_map["moneyness"])] = 1.0
            x.set_weights(weights)
            x.trainable = False
        if x.name == "other_feats":
            weights = x.get_weights()
            weights[0] = np.zeros(weights[0].shape)
            weights[0][2:] = 1.0
            x.set_weights(weights)
            x.trainable = False