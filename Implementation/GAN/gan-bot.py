import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, ReLU, LeakyReLU, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.initializers import random_uniform

import matplotlib.pyplot as plt

from Dissertation.Implementation.process_data import read_files, normalise_data, drop_columns

