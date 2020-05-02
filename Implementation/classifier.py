import os
import json
import csv
import sys
import pickle
import time
import logging
import argparse
import pandas as pd

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold, StratifiedKFold

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/classifier-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, normalise_data, sort_time

def make_dir(path):
  if not os.path.exists(path):
    logger.info("Creating directory at path: {0}".format(path))
    os.makedirs(path)

def label_encode_class(data):
  def label_encoder_mapping(le):
    # todo save this label encoder later for prediction.
    return dict(zip(le.classes_, le.transform(le.classes_)))

  assert data.shape[1] == 1  # only one column - output label.

  le = LabelEncoder()
  le.fit(data)

  logger.info("Label encoder mapping {0}".format(label_encoder_mapping(le)))

  return le.transform(data).ravel(), label_encoder_mapping(le)


def split_data(data, test_size=0.3, normalise=False):
  from sklearn.model_selection import train_test_split
  categorical_columns = data.select_dtypes(['object'])
  excluded_columns = ['Timestamp', 'Dst Port']
  output = ['Label']
  inputs = [label for label in list(data) if label not in output and label not in categorical_columns
            and label not in excluded_columns]

  if normalise:
    logger.info("Data is being normalised.")
    data[inputs] = normalise_data(data[inputs])

  X = data[inputs]
  y, mapping = label_encode_class(data[output])
  logger.info("Y/Output variable {0} with shape {1}".format(output, y.shape))
  logger.info("X/Input variables {0} with shape {1}".format(inputs, X.shape))
  logger.info("Train vs Test split: {0}-{1}".format(1 - test_size, test_size))
  # return train_test_split(X, y, test_size=test_size), mapping  # 70% training and 30% test
  return X.to_numpy(), y, mapping, X.columns

def split_time_series(data, normalise=True):
  no_of_split = 3# int((len(data) - 3) / 3)  # 67-33
  categorical_columns = data.select_dtypes(['object'])
  excluded_columns = ['Timestamp', 'Dst Port']

  output = ['Label']
  inputs = [label for label in list(data) if label not in output and label not in categorical_columns
            and label not in excluded_columns]

  if normalise:
    logger.info("Data is being normalised.")
    logger.info("Inputs: {0}".format(inputs))
    data[inputs] = normalise_data(data[inputs])

  X = data[inputs]
  y, mapping = label_encode_class(data[output])
  time_series_split = TimeSeriesSplit(n_splits=no_of_split)

  for train_index, test_index in time_series_split.split(X):
    # To get the indices
    #logger.info("Train index: {0} - Test index: {1}".format(train_index, test_index))
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    # print("X: {0}".format(X))
    #print("Xtrain: {0}".format(X_train))
    #print("Xtest: {0}".format(X_test))
    y_train, y_test = y[train_index], y[test_index]
    #print("Y: {0}".format(y))
    #print("Y_train: {0}".format(y_train))
    #print("Y_test: {0}".format(y_test))
    logger.info("Observations: {0}".format(len(train_index) + len(test_index)))
    logger.info("Training observations: {0}".format(len(train_index)))
    logger.info("Testing observations: {0}".format(len(test_index)))
    yield X_train, X_test, y_train, y_test, mapping


def generate_classification_report(fp, method, y_test, y_pred, save=False):
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  report_df = pd.DataFrame(report).T

  out = r"{0}/{1}-classification_report.csv".format(fp, method)
  if save:
    report_df.to_csv(out)
    logger.info("Saving classification report at location: {0}".format(out))
  return report


def save_metrics(fp, method_name, mapping, accuracy, time, i=""):
  encoder_out = '{0}/{1}-encoder-mapping-{2}.txt'.format(fp, method_name, i)
  logger.info("Saving label encoder data at location: %s" % encoder_out)
  pd.DataFrame.from_dict(mapping, orient='index').to_csv(encoder_out)

  metrics_out = '{0}/{1}-metrics-{2}.txt'.format(fp, method_name, i)
  metrics_dict = {'accuracy': accuracy,
                  'run-time': time}

  logger.info("Accuracy: {0}".format(accuracy))
  logger.info("Classifier took {0} seconds to train".format(time))
  logger.info("Saving metrics file at location {0}".format(metrics_out))
  pd.DataFrame.from_dict(metrics_dict, orient='index').to_csv(metrics_out)


def save_feature_importances(feature_importances, fp, i=""):
  feature_importances.to_csv(r"{0}/random-forest-feature-importance-{1}.csv".format(fp, i))


def save_model(model, method_name, fp, i=""):
  out = ("{0}/{1}-{2}.sav").format(fp, method_name, i)
  logger.info("Saving {0} model at location: {1}".format(method_name, out))
  pickle.dump(model, open(out, 'wb'))


def calculate_time(start_time):
  return time.time() - start_time

def random_forest_classifier(data, fp, save=False, time_series=False):
  from sklearn.ensemble import RandomForestClassifier
  method_name = "random-forest"

  if time_series:
    for i, Xy in enumerate(split_time_series(data)):
      X_train, X_test, y_train, y_test, mapping = Xy
      print(type(mapping))
      logger.info("Random forest classifier -- TIME SERIES -- initialised")
      start_time = time.time()
      clf = RandomForestClassifier(n_estimators=100, verbose=2)
      clf.fit(X_train, y_train)
      feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns,
                                         columns=['importance']).sort_values('importance', ascending=False)

      y_pred = clf.predict(X_test)
      generate_classification_report(fp, "{0}-{1}".format(method_name, i), y_test, y_pred, save)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      run_time = calculate_time(start_time)
      save_feature_importances(feature_importances, fp, i)

      if save:
        save_metrics(fp, method_name, mapping, accuracy, run_time, i)
        save_model(clf, method_name, fp, i)

  else:
    X, y, mapping, columns = split_data(data)
    # X_train, X_test, y_train, y_test = Xy

    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)
    logger.info("Random forest classifier -- initialised")
    i = 0
    for train_index, test_index in cv.split(X, y):
      X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
      start_time = time.time()
      clf = RandomForestClassifier(n_estimators=100, verbose=2)
      clf.fit(X_train, y_train)

      feature_importances = pd.DataFrame(clf.feature_importances_, index=columns,
                                         columns=['importance']).sort_values('importance', ascending=False)

      y_pred = clf.predict(X_test)
      rf_out = r"{0}/{1}".format(fp, i)
      make_dir(rf_out)
      generate_classification_report(rf_out, method_name, y_test, y_pred, save)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      run_time = calculate_time(start_time)
      save_feature_importances(feature_importances, rf_out)

      if save:
        save_metrics(rf_out, method_name, mapping, accuracy, run_time)
        save_model(clf, method_name, rf_out)
      i += 1


def support_vector_machine_classifier(data, fp, save=False, time_series=False):
  from sklearn import svm
  method_name = "svm"

  if time_series:
    for i, Xy in enumerate(split_time_series(data)):
      X_train, X_test, y_train, y_test, mapping = Xy
      logger.info("Support vector machine classifier -- TIME SERIES -- initialised")
      start_time = time.time()
      clf = svm.LinearSVC(verbose=10)
      clf.fit(X_train, y_train)

      y_pred = clf.predict(X_test)
      generate_classification_report(fp, method_name, y_test, y_pred, save)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      run_time = calculate_time(start_time)

      if save:
        save_metrics(fp, method_name, mapping, accuracy, run_time)
        save_model(clf, method_name, fp)

  else:
    X, y, mapping, _ = split_data(data, normalise=True)
    # X_train, X_test, y_train, y_test = Xy

    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=False)
    i = 0
    for train_index, test_index in cv.split(X, y):
      X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

      logger.info("Support vector machine classifier -- initialised")
      start_time = time.time()
      clf = svm.LinearSVC(verbose=2)
      clf.fit(X_train, y_train)
      svm_out = r"{0}/{1}".format(fp, i)
      make_dir(svm_out)
      y_pred = clf.predict(X_test)
      generate_classification_report(svm_out, method_name, y_test, y_pred, save)
      accuracy = metrics.accuracy_score(y_test, y_pred)
      run_time = calculate_time(start_time)

      if save:
        save_metrics(svm_out, method_name, mapping, accuracy, run_time)
        save_model(clf, method_name, svm_out)
      i += 1


if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file-location", help="location to files.", default="../Datasets/cleaned")
  parser.add_argument("-o", "--out", help="out folder path", default="out/")

  args = parser.parse_args()
  make_dir(args.out)

  original_dataset = read_files([args.file_location], clean_data=False)  # todo remove hardcode

  original_dataset = sort_time(original_dataset)
  support_vector_machine_classifier(original_dataset, fp=args.out, save=True, time_series=False)
  # random_forest_classifier(original_dataset,fp=args.out, save=True, time_series=False)
