import sys
import pickle
import time
import logging

import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/classifier-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, normalise_data


def label_encode_class(data):
  def label_encoder_mapping(le):
    # todo save this label encoder later for prediction.
    return dict(zip(le.classes_, le.transform(le.classes_)))

  assert data.shape[1] == 1  # only one column - output label.

  le = LabelEncoder()
  le.fit(data)

  logger.info("Label encoder mapping {0}".format(label_encoder_mapping(le)))

  return le.transform(data).ravel()


def split_data(data, test_size=0.3, normalise=False):
  from sklearn.model_selection import train_test_split
  categorical_columns = data.select_dtypes(['object'])

  output = ['Label']
  inputs = [label for label in list(data) if label not in output and label not in categorical_columns]

  if normalise:
    logger.info("Data is being normalised.")
    data[inputs] = normalise_data(data[inputs])

  X = data[inputs]
  y = label_encode_class(data[output])

  logger.info("Y/Output variable {0} with shape {1}".format(output, y.shape))
  logger.info("X/Input variables {0} with shape {1}".format(inputs, X.shape))
  logger.info("Train vs Test split: {0}-{1}".format(1 - test_size, test_size))
  return train_test_split(X, y, test_size=test_size)  # 70% training and 30% test



def split_time_series(data):
  no_of_split = int((len(data) - 3) / 3)  # 67-33
  categorical_columns = data.select_dtypes(['object'])

  output = ['Label']
  inputs = [label for label in list(data) if label not in output and label not in categorical_columns]

  if normalise:
    logger.info("Data is being normalised.")
    data[inputs] = normalise_data(data[inputs])

  X = data[inputs]
  y = label_encode_class(data[output])

  time_series_split = TimeSeriesSplit(n_splits=no_of_split)

  for train_index, test_index in time_series_split.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

    # To get the indices
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    yield X_train, X_test, y_train, y_test


def random_forest_classifier(data, save=False, fp=r"out/rf-model.sav"):
  from sklearn.ensemble import RandomForestClassifier

  X_train, X_test, y_train, y_test = split_data(data)

  logger.info("Random forest classifier -- initialised")
  start_time = time.time()
  clf = RandomForestClassifier(n_estimators=100, verbose=2)
  clf.fit(X_train, y_train)

  if save:
    logger.info("Saving RANDOM-FOREST-CLASSIFIER-MODEL at location: %s" % fp)
    pickle.dump(clf, open(fp, 'wb'))

  y_pred = clf.predict(X_test)
  logger.info("Random forest classifier accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
  logger.info("Random forest classifier took %s seconds" % (time.time() - start_time))


def support_vector_machine_classifier(data, fp, save=False, time_series=True,):
  from sklearn import svm
  out = fp + "svm-model.sav"

  if time_series:
    for i, X_train, X_test, y_train, y_test in enumerate(split_time_series(data)):
      logger.info("Support vector machine classifier -- TIME SERIES -- initialised")
      start_time = time.time()
      clf = svm.LinearSVC(verbose=10)
      clf.fit(X_train, y_train)

      if save:
        out = fp + "svm-model-" + i + ".sav"
        logger.info("Saving SUPPORT-VECTOR-MACHINE-CLASSIFIER-MODEL at location: %s" % out)
        pickle.dump(clf, open(out, 'wb'))

      y_pred = clf.predict(X_test)
      logger.info("Support vector machine accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
      logger.info("Support vector machine classifier took %s seconds" % (time.time() - start_time))
  else:
    X_train, X_test, y_train, y_test = split_data(data, normalise=True)

    logger.info("Support vector machine classifier -- initialised")
    start_time = time.time()
    clf = svm.LinearSVC(verbose=2)
    clf.fit(X_train, y_train)

    if save:
      logger.info("Saving SUPPORT-VECTOR-MACHINE-CLASSIFIER-MODEL at location: %s" % out)
      pickle.dump(clf, open(out, 'wb'))

    y_pred = clf.predict(X_test)
    logger.info("Support vector machine accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
    logger.info("Support vector machine classifier took %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  original_dataset, pruned_dataset = read_files(
    [r"../Datasets/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"], prune=True)  # todo remove hardcode

  pd.plotting.register_matplotlib_converters()  # todo convert this to a function
  original_dataset['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d/%m/%Y %H:%M:%S")

  original_dataset = original_dataset.sort_values(['Timestamp'], ascending=[True])

  # random_forest_classifier(pruned_dataset, True)
  support_vector_machine_classifier(pruned_dataset, True)
