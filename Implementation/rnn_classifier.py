import sys
import time
import logging
import argparse

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/rnn-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, drop_columns, sort_time
from classifier import label_encode_class


def convert_data(data, n_steps):
  logger.info("2d -> 3d: n_steps = {0}".format(n_steps))
  new_data = np.zeros((data.shape[0] // n_steps, n_steps, data.shape[1]))

  for i in range(len(data) // n_steps):
    new_data[i, :] = data[i * n_steps:i * n_steps + n_steps]

  logger.info("New data shape: {0}".format(new_data.shape))
  return new_data


def generate_train_test(data, test_size=0.3):
  from sklearn.model_selection import train_test_split
  X = data[:, :, 0:78]  # take all features except label.
  y = data[:, :, 78:79]  # last feature = label
  y = y.reshape((y.shape[0], y.shape[1] * y.shape[2]))

  return train_test_split(X, y, test_size=test_size)


def create_model(shape):
  # shape = 10, 78
  model = Sequential()
  model.add(LSTM(512, input_shape=(shape[0],
                                   shape[1]),
                 return_sequences=False))
  model.add(Dense(10, activation='sigmoid'))  # todo change for multiclassification
  model.compile(loss='binary_crossentropy', optimizer='adam')
  return model


def train(data):
  Xy = generate_train_test(data)
  X_train, X_test, y_train, y_test = Xy
  start_time = time.time()

  model = create_model((X_train.shape[1], X_train.shape[2]))
  print("X_train shape: {0}".format(X_train.shape))
  print("Y_train.shape: {0}".format(y_train.shape))
  history = model.fit(X_train, y_train, epochs=2)
  # model.compile()
  print(model.summary())

  logger.info("Random forest classifier took %s seconds" % (time.time() - start_time))

  scores = model.evaluate(X_test, y_test, verbose=1)
  logger.info("Evaluation names: {0}".format(model.metrics_names))
  logger.info("Accuracy: {0}".format(scores[1] * 100))


if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file-location", help="location to files.", default=r"../../../dataset/cleaned")
  parser.add_argument("-o", "--out", help="out folder path", default="out/")
  parser.add_argument("-n", "--n-steps", help="number of steps per one block", default=10)

  args = parser.parse_args()

  original_dataset = read_files([args.file_location], clean_data=False)
  original_dataset = sort_time(original_dataset)
  print(original_dataset.isnull().any())

  label = original_dataset['Label'].to_numpy()[:, np.newaxis]
  OHC_label, _ = label_encode_class(label)

  original_dataset = drop_columns(original_dataset, ['Timestamp', 'Label'])
  original_dataset['Label'] = OHC_label.tolist()

  reshaped_data = convert_data(original_dataset.to_numpy(), args.n_steps)
  train(reshaped_data)
