import os
import sys
import time
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/rnn-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, drop_columns, sort_time
from classifier import save_model, label_encode_class
from nn.nn import create_model, create_mlp


def single_split(X, y, test_size):
  from sklearn.model_selection import train_test_split
  return train_test_split(X, y, test_size=test_size)


def make_dir(path):
  if not os.path.exists(path):
    logger.info("Creating directory at path: {0}".format(path))
    os.makedirs(path)


# 2d rows vs features = 20000, 80
def split_data(data, num_classes):
  x_y_split = data.shape[1] - num_classes

  X = data[:, 0:x_y_split]  # take all features except label.
  y = data[:, x_y_split:]  # last feature = label

  return X, y


def yield_sliding_window_data(data, window_size):
  X, y = data

  for i in range(1, X.shape[0] + 1):
    if i < window_size:
      # case i = 1, context is index 0 therefore not needed
      # example window size = 5, context for i = 1 -> 00001
      # 00001, 00012, 00123, 01234
      zero_rows = (1 * window_size) - i
      zeros_window = np.zeros((zero_rows, X.shape[1]))
      data_window = X[0:i, :]

      window = np.vstack((zeros_window, data_window))

    else:
      # 12345, ...
      starting_row = i - window_size
      window = X[starting_row:i, :]

    lstm_window = window[0:window.shape[0] - 1, :][np.newaxis, :, :]  # [:, :, np.newaxis]
    window_last_element = window[window.shape[0] - 1: window.shape[0], :]

    if lstm_window.shape[0] == 1:
      pass
    yield ([lstm_window, window_last_element], y[i - 1].reshape((1, -1)))


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
    out = "{0}/nn-history.png".format(fp)
    plt.savefig(out)
    logger.info("Neural network history saved at: {0}".format(out))

  plt.show()


def train(data,
          fp,
          num_classes,
          save=False,
          window_size=10,
          loss='categorical_crossentropy',
          activation='relu',
          final_activation='softmax',
          optimiser='adam',
          epochs=10,
          metrics=None,
          batch_size=30,
          test_size=0.3):
  print(loss)

  X, y = split_data(data, num_classes=num_classes)
  print("Shape of X: {0}".format(X.shape))

  print("Shape of y: {0}".format(y.shape))

  # steps_per_epoch = X_train.shape[0] / batch_size
  # validation_steps = X_test.shape[0] / batch_size

  start_time = time.time()

  lstm = False  # TODO edit this variable to make it more reusable but i am lazy.
  if lstm:
    pass
    """
    model = create_model(shape=(window_size, X_train.shape[1]),
                       activation=activation,
                       final_activation=final_activation,
                       num_classes=num_classes)

    model.compile(loss=loss,
                  optimizer=optimiser,
                  metrics=metrics)

    logger.info(model.summary())

    history = model.fit_generator(generator=yield_sliding_window_data(data=(X_train, y_train),
                                                                      window_size=window_size),
                                  epochs=epochs,
                                  validation_data=yield_sliding_window_data(data=(X_test, y_test),
                                                                            window_size=window_size),
                                  steps_per_epoch=steps_per_epoch,
                                  validation_steps=validation_steps,
                                  callbacks=[checkpoint])
                                  """
  else:  # for creating pretrained dense network
    scores = []
    cv = KFold(n_splits=10, random_state=42, shuffle=False)

    i = 0
    fp = r"{0}/{1}".format(fp, i)

    make_dir(fp)
    filepath = fp + r"\weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

    for train_index, test_index in cv.split(X):
      X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
      print(X_train.shape)
      print(y_train.shape)

      model = create_mlp(feature_size=X.shape[1],
                         num_classes=num_classes,
                         activation=activation,
                         final_activation=final_activation)
      model.compile(loss=loss,
                    optimizer=optimiser,
                    metrics=metrics)

      logger.info(model.summary())

      history = model.fit(X_train, y_train,
                          callbacks=[checkpoint],
                          epochs=epochs,
                          validation_data=(X_test, y_test))

      plot_history(history, fp, save)

      scores.append(history.history['val_accuracy'])
      i += 1

    logger.info(np.mean(scores))

  run_time = time.time() - start_time
  logger.info("Model took %s seconds to train" % (run_time))


def one_hot_encode_data(label):
  ohe = OneHotEncoder()
  return ohe.fit_transform(label).toarray()


if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file-location", help="location to files.", default=r"../../../dataset/cleaned")
  parser.add_argument("-o", "--out", help="out folder path", default="out/")
  parser.add_argument("-n", "--n-steps", help="number of steps per one block", default=10, type=int)

  args = parser.parse_args()

  original_dataset = read_files([args.file_location], clean_data=False)
  # original_dataset = sort_time(original_dataset)

  label = original_dataset['Label'].to_numpy()[:, np.newaxis]
  _, mapping = label_encode_class(label)

  encoder_out = '{0}/{1}-encoder-mapping.txt'.format(args.out, "lstm")
  logger.info("Saving label encoder data at location: %s" % encoder_out)
  pd.DataFrame.from_dict(mapping, orient='index').to_csv(encoder_out)

  OHC_Label = one_hot_encode_data(label)

  original_dataset = drop_columns(original_dataset, ['Timestamp', 'Label'])
  num_classes = OHC_Label.shape[1]
  for i in range(num_classes):
   original_dataset[i] = OHC_Label[:, i]
   original_dataset[i] = OHC_Label[:, i]

  train(original_dataset.to_numpy(),
        args.out,
        num_classes=num_classes,
        save=True,
        metrics=['accuracy'],
        window_size=args.n_steps)
