import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Concatenate, 
                                     BatchNormalization, Multiply)
from tensorflow.keras import Model

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.callbacks import EarlyStopping



def make_model(N_FEATS, HIDDEN_UNITS, LAYERS, DROPOUT_RATIO, HIDDEN_ACT, OUTPUT_ACT, BATCH_NORM):
    tf.random.set_seed(42)
    input_layer = Input(N_FEATS)
    x = BatchNormalization()(input_layer)
    x = Dense(HIDDEN_UNITS)(x)
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

def ffn_network(n_feats: int, n_layers:int, hidden_layer, output_layer, seed:int = 42) -> tf.keras.Model:
    input_layer =  Input(n_feats)
    x = hidden_layer()(input_layer)
    for i in range(n_layers - 1):
        x = hidden_layer()(x)
    final = output_layer()(x)
    return Model(input_layer, final)

def ensemble_model(models: List, N_SEEDS:int) -> tf.keras.Model:
    x = Concatenate()([models[i].output for i in range(N_SEEDS)])
    x = MeanLayer()(x)
    model = Model([models[i].input for i in range(N_SEEDS)], x)
    return model