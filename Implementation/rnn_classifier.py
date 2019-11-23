import sys
import time
import logging
import argparse

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

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
from nn.nn import create_model

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
  num_classes = np.unique(y.ravel())

  return train_test_split(X, y, test_size=test_size), num_classes


def yield_sliding_window_data(data, num_classes, window_size=10):
  X, y = data

  for i in range(1, X.shape[0] + 1):
    if i < window_size:
      if i == 0:  # initial case, everything is zero for context.
        window = np.zeros((1 * window_size, X.shape[1]))
      else:
        # case i = 1, context is index 0.
        # example window size = 5, context for i = 1 -> 00001
        # 00000, 00001, 00012, 00123, 01234
        zero_rows = (1 * window_size) - i
        zeros_window = np.zeros((zero_rows, X.shape[1]))
        data_window = X[0:i, :]

        window = np.vstack((zeros_window, data_window))

    else:
      # 12345, ...
      starting_row = i - window_size
      window = X[starting_row:i, :]

    lstm_window = window[0:window.shape[0] - 1, :]
    window_last_element = window[window.shape[0] - 1: window.shape[0], :]

    fill_shape = np.zeros(num_classes - 1, y.shape[1])
    y_output = np.hstack((fill_shape, y[i,:]))

    yield ([lstm_window, window_last_element], y_output)


def plot_history(history, fp, save):
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))
  ax1.plot(history.history['accuracy'])
  ax1.plot(history.history['val_accuracy'])
  ax1.set_title('accuracy')
  ax1.set_ylabel('accuracy')
  ax1.set_xlabel('epoch')
  ax1.legend(['train', 'val'], loc='upper left')

  ax2.plot(history.history['loss'])
  ax2.plot(history.history['val_loss'])
  ax2.set_title('loss')
  ax2.set_ylabel('loss')
  ax2.set_xlabel('epoch')
  ax2.legend(['train', 'val'], loc='upper left')

  plt.suptitle("Train vs validation accuracy and loss")

  if save:
    plt.savefig("{0}/nn-history.png".format(fp))

  plt.show()


def train(data,
          fp,
          save=False,
          window_size=10,
          loss='categorical_crossentropy',
          activation='relu',
          final_activation='softmax',
          optimiser='adam',
          epochs=10,
          metrics=None):

  (X_train, X_test, y_train, y_test), num_classes = generate_train_test(data)

  model = create_model(shape=(window_size, X_train.shape[2]),
                       activation=activation,
                       final_activation=final_activation,
                       num_classes=num_classes)

  model.compile(loss=loss,
                optimizer=optimiser,
                metrics=metrics)

  logger.info(model.summary())

  start_time = time.time()
  history = model.fit_generator(generator=yield_sliding_window_data(data=(X_train, y_train),
                                                                    num_classes=num_classes),
                                epochs=epochs,
                                validation_data=yield_sliding_window_data(data=(X_test, y_test),
                                                                          num_classes=num_classes))

  plot_history(history, fp, save)

  logger.info("Model took %s seconds to train" % (time.time() - start_time))


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
  train(reshaped_data, args.out, save=False, metrics=['accuracy'], window_size=args.n_steps)
