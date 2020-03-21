import os
import sys
import time
import pickle
import logging
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

from keras import optimizers
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
from ensemble import train_ensemble, load_all_models

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


def train(X, y,
          fp,
          num_classes,
          save=False,
          loss='categorical_crossentropy',
          activation='relu',
          final_activation='softmax',
          optimiser='adam',
          epochs=10,
          metrics=None,
          batch_size=256,
          test_size=0.3, y_multiclass=None):

  print("Shape of X: {0}".format(X.shape))
  print("Shape of y: {0}".format(y.shape))

  # steps_per_epoch = X_train.shape[0] / batch_size
  # validation_steps = X_test.shape[0] / batch_size

  start_time = time.time()

  scores = []
  cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)

  i = 0
  for train_index, test_index in cv.split(X, y_multiclass):
    out = r"{0}/{1}".format(fp, i)
    make_dir(out)
    filepath = out + r"/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

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
                        validation_data=(X_test, y_test),
                        batch_size=batch_size)

    print("building classification report")
    y_pred = model.predict(X_test, batch_size=64)
    print(y_pred.shape)
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred.shape)
    print(y_test.shape)
    y_test = np.argmax(y_test, axis=1)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).T
    cf_out = r"{0}/classification-report.txt".format(out)
    report_df.to_csv(cf_out)
    logger.info("Saving classification report at location: {0}".format(cf_out))

    plot_history(history, out, save)

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
  parser.add_argument("-e", "--ensemble", action="store_true")  # train ensemble
  parser.add_argument("-p", "--multiple-model-location")  # multiple pretrained models for ensemble

  args = parser.parse_args()
  make_dir(args.out)

  original_dataset = read_files([args.file_location], clean_data=False)
  original_dataset = original_dataset.sample(frac=1).reset_index(drop=True)  # shuffle dataset

  label = original_dataset['Label'].to_numpy()[:, np.newaxis]
  y_multiclass, mapping = label_encode_class(label)
  print(y_multiclass.shape)
  encoder_out = '{0}/{1}-encoder-mapping.txt'.format(args.out, "lstm")
  logger.info("Saving label encoder data at location: %s" % encoder_out)
  pd.DataFrame.from_dict(mapping, orient='index').to_csv(encoder_out)

  OHC_Label = one_hot_encode_data(label)

  original_dataset = drop_columns(original_dataset, ['Timestamp', 'Label'])
  num_classes = OHC_Label.shape[1]
  for i in range(num_classes):
   original_dataset[i] = OHC_Label[:, i]
   original_dataset[i] = OHC_Label[:, i]

  X, y = split_data(original_dataset.to_numpy(), num_classes=num_classes)

  if args.ensemble:
    model_loc = glob(args.multiple_model_location + r"/*.hdf5")
    assert len(model_loc) >= 2  # atleast two models are required
    pretrained_models = load_all_models(model_loc)
    ensemble_model = train_ensemble(pretrained_models, X, y)

    ensemble_out = ("{0}/{1}.sav").format(args.out, "ensemble")
    pickle.dump(ensemble_model, open(ensemble_out, 'wb'))

    logger.info("Saving ensemble model at location {0}".format(ensemble_model))

  else:
    train(X, y,
          args.out,
          num_classes=num_classes,
          save=True,
          metrics=['accuracy'],
          y_multiclass=y_multiclass,
          optimiser=optimizers.Adam(lr=0.0001))
