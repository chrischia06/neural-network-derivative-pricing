import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Concatenate, 
                                     BatchNormalization, Multiply)
from tensorflow.keras import Model

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Union, Callable
import tensorflow_probability as tfp


def make_model(N_FEATS:int, 
               HIDDEN_UNITS:int, 
               LAYERS:int, 
               DROPOUT_RATIO:float, 
               HIDDEN_ACT: Union[str, Callable], 
               OUTPUT_ACT: Union[str, Callable], 
               BATCH_NORM: bool,
               SEED:int = 42) -> Model:
    tf.random.set_seed(SEED)
    input_layer = Input(N_FEATS)
    if BATCH_NORM:
        x = BatchNormalization()(input_layer)
        x = Dense(HIDDEN_UNITS, activation=HIDDEN_ACT)(x)
    else:
        x = Dense(HIDDEN_UNITS, activation=HIDDEN_ACT)(input_layer)
    for i in range(LAYERS - 1):
        x = Dense(HIDDEN_UNITS, activation=HIDDEN_ACT, 
                  kernel_initializer='glorot_uniform', 
                  kernel_constraint=None)(x)
        if BATCH_NORM:
            x = BatchNormalization()(x)
        if DROPOUT_RATIO > 0:
            x = Dropout(dropout_ratio)(x)
    output_layer = Dense(1, activation=OUTPUT_ACT)(x)
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

def gated_network(N_FEATS) -> tf.keras.Model:
    input_layer =  Input(N_FEATS)
    x = Dense(100, activation='softplus', kernel_constraint='non_neg')(input_layer)
    weights = Dense(100, activation='softmax', kernel_constraint='non_neg')(x)
    x2 = Dense(100, activation='sigmoid', kernel_constraint='non_neg')(x)
    mult = Multiply()([weights, x2])
    output_layer = SumLayer()(mult)
    model = Model(input_layer, output_layer)
    return model



def ensemble_model(models: List, N_SEEDS:int) -> tf.keras.Model:
    x = Concatenate()([models[i].output for i in range(N_SEEDS)])
    x = MeanLayer()(x)
    model = Model([models[i].input for i in range(N_SEEDS)], x)
    return model

def rbf(x):
    return tf.math.exp(-0.5 * x ** 2)

def homogeneity_network(N_FEATS:int, HIDDEN_UNITS:int = 100, seed:int = 42):
    """
    N_FEATS: No. of inputs, at least moneyness and time-to-maturity are required
    HIDDEN_UNITS: No. of hidden units
    SEED: Hidden Seed
    
    Constrains the moneyness and time-to-maturity variables to have positive weights, and no constraints for the other paramters
    """
    assert N_FEATS >= 2
    tf.random.set_seed(seed)
    input_moneyness = Input(1, name='moneyness')
    input_ttm = Input(1, name='ttm') 
    x1 = Dense(HIDDEN_UNITS, activation='sigmoid', kernel_constraint='non_neg')(input_ttm)
    x2 = Dense(HIDDEN_UNITS, activation='softplus', kernel_constraint='non_neg')(input_moneyness)
    interaction_layers = [x1, x2]
    input_layers = [input_moneyness, input_ttm]
    if N_FEATS > 2:
        input_other = Input(N_FEATS - 2)
        input_layers += [input_other]
        x3 = Dense(HIDDEN_UNITS, activation='softplus', name='other_features')(input_other)
        interaction_layers += [x3]
    x = Multiply()(interaction_layers)
    output_layer = Dense(1, kernel_constraint='non_neg', use_bias=False)(x)
    model = Model(input_layers, output_layer)
    return model

# def homogeneity_network(N_FEATS):
#     input_moneyness = Input(1)
#     input_ttm = Input(1)
    
#     x1 = Dense(1, activation='sigmoid', kernel_constraint='non_neg')(input_ttm)
#     x2 = Dense(1, activation='softplus', kernel_constraint='non_neg')(input_moneyness)
#     input_layers = [input_ttm, input_moneyness]
#     interaction_layers = [x1, x2]
#     if N_FEATS > 2:
#         input_other = Input(N_FEATS - 2)
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