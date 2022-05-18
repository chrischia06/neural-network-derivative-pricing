import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


from itertools import product
from tqdm.notebook import tqdm
tqdm.pandas()


import seaborn as sns
plt.style.use("ggplot")


from typing import List

import tensorflow as tf
from tensorflow.keras.layers import (Dense, Input, Concatenate, 
                                     BatchNormalization, Multiply)
from tensorflow.keras import Model






from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

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