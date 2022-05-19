import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Concatenate, 
                                     BatchNormalization, Multiply)
from tensorflow.keras import Model

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping
from typing import List, Union, Callable



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


def homogeneity_network(N_FEATS):
    input_moneyness = Input(1)
    input_ttm = Input(1)
    
    x1 = Dense(1, activation='sigmoid', kernel_constraint='non_neg')(input_ttm)
    x2 = Dense(1, activation=rbf, kernel_constraint='non_neg')(input_moneyness)
    input_layers = [input_ttm, input_moneyness]
    interaction_layers = [x1, x2]
    if N_FEATS > 2:
        input_other = Input(N_FEATS - 2)
        input_layers += [input_other]
        x3 = Dense(100, 'swish')(input_layers[-1])
        x3 = Dense(1, activation='sigmoid')(x3)
        x3 = Dense(1, kernel_constraint='non_neg', use_bias=False)(x3)
        interaction_layers += [x3]

    x = Multiply()(interaction_layers)
    output_layer = Dense(1, kernel_constraint='non_neg', use_bias=False)(x)
    model = Model(input_layers, output_layer)
    # x3 = Dense(1, activation='softplus', kernel_constraint='non_neg')(input_moneyness)
    # x4 = Multiply()([x2, x3])

    # x2 = Dense(1, activation='exponential', name='intrinsic')(input_layer)
    # x3 = Lambda(lambda x: tf.keras.activations.relu(x - 1))(x2)
    # x = Dense(32, activation='softplus', kernel_constraint='non_neg')(x)
    # x = Dense(32, activation='softplus', kernel_constraint='non_neg')(x)
    # output_layer = Add()([x3, Dense(1, activation='softplus')(x)])

    # model.layers

    # model.layers[1].trainable = False
    # model.layers[1].set_weights([np.array([[1], [0]]), np.array([0.0])])
    return model