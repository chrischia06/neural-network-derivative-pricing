import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import grad

from itertools import product
import pandas as pd
from tqdm.notebook import tqdm
tqdm.pandas()


import seaborn as sns
plt.style.use("ggplot")




from utils import diagnosis_pde, diagnosis_pred

import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow_addons.callbacks import TQDMProgressBar

import time
