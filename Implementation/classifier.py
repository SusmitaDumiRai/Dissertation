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

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/classifier-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, normalise_data

def make_dir(path):
  if not os.path.exists(path):
    logger.info("Creating directory at path: {0}".format(path))
    os.makedirs(path)

def label_encode_class(data,):  # todo pass out.
  def label_encoder_mapping(le):
    # todo save this label encoder later for prediction.
    return dict(zip(le.classes_, le.transform(le.classes_)))

  assert data.shape[1] == 1  # only one column - output label.

  le = LabelEncoder()
  le.fit(data)

  logger.info("Label encoder mapping {0}".format(label_encoder_mapping(le)))

  if save:  # todo fix this for time series, does not know which iteration is which.
    with open('{0}/label-encoder-mapping.txt'.format(out), 'w') as f:
      file.write(json.dumps(label_encoder_mapping(le)))
  return le.transform(data).ravel(), label_encoder_mapping(le)


def split_data(data, test_size=0.3, out, save, normalise=False):
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
  y = label_encode_class(data[output], out, save)
  logger.info("Y/Output variable {0} with shape {1}".format(output, y.shape))
  logger.info("X/Input variables {0} with shape {1}".format(inputs, X.shape))
  logger.info("Train vs Test split: {0}-{1}".format(1 - test_size, test_size))
  return train_test_split(X, y, test_size=test_size)  # 70% training and 30% test



def split_time_series(data, out, save, normalise=True):
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
  y = label_encode_class(data[output], out, save)

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
    yield X_train, X_test, y_train, y_test


def generate_classification_report(fp, method, y_test, y_pred, save=False):
  report = metrics.classification_report(y_test, y_pred, output_dict=True)
  report_df = pd.DataFrame(report).T

  out = r"{0}/{1}-classification_report.csv".format(fp, method)
  if save:
    report_df.to_csv(out)
    logger.info("Saving classification report at location: {0}".format(out))
  return report

def random_forest_classifier(data, fp, save=False, time_series=False):
  from sklearn.ensemble import RandomForestClassifier
  out = r"{0}/random-forest-model.sav".format(fp)

  if time_series:
    for i, Xy in enumerate(split_time_series(data, fp, save)):
      X_train, X_test, y_train, y_test = Xy
      logger.info("Random forest classifier -- TIME SERIES -- initialised")
      start_time = time.time()
      clf = RandomForestClassifier(n_estimators=100, verbose=2)
      clf.fit(X_train, y_train)
      feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns,
                                         columns=['importance']).sort_values('importance', ascending=False)

      logger.info("Feature importance: {0}".format(feature_importances.head(5)))

      if save:
        feature_importances.to_csv(r"{0}/random-forest-feature-importance-{1}.csv".format(fp, i))

        out = ("{0}/random-forest-model-{1}.sav").format(fp, i)
        logger.info("Saving RANDOM-FOREST-CLASSIFIER-MODEL at location: %s" % out)
        pickle.dump(clf, open(out, 'wb'))

      y_pred = clf.predict(X_test)
      report = generate_classification_report(fp, "random-forest", y_test, y_pred, save)
      logger.info("Random forest classifier accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
      logger.info("Random forest classifier classification report: {0}".format(report))
      logger.info("Random forest classifier took %s seconds" % (time.time() - start_time))

  else:
    X_train, X_test, y_train, y_test = split_data(data, fp, save)

    logger.info("Random forest classifier -- initialised")
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=100, verbose=2)
    clf.fit(X_train, y_train)

    feature_importances = pd.DataFrame(clf.feature_importances_, index=X_train.columns,
                                       columns=['importance']).sort_values('importance', ascending=False)
    logger.info("Feature importance: {0}".format(feature_importances.head(5)))
    feature_importances.to_csv(r"{0}/random-forest-feature-importance.csv".format(fp))
    if save:
      logger.info("Saving RANDOM-FOREST-CLASSIFIER-MODEL at location: %s" % out)
      pickle.dump(clf, open(out, 'wb'))

    y_pred = clf.predict(X_test)
    report = generate_classification_report(fp, "random-forest", y_test, y_pred, save)
    logger.info("Random forest classifier accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
    logger.info("Random forest classifier classification report: {0}".format(report))
    logger.info("Random forest classifier took %s seconds" % (time.time() - start_time))


def support_vector_machine_classifier(data, fp, save=False, time_series=False):
  from sklearn import svm
  out = r"{0}/svm-model.sav".format(fp)

  if time_series:
    for i, Xy in enumerate(split_time_series(data, fp, save)):
      X_train, X_test, y_train, y_test = Xy
      logger.info("Support vector machine classifier -- TIME SERIES -- initialised")
      start_time = time.time()
      clf = svm.LinearSVC(verbose=10)
      clf.fit(X_train, y_train)

      if save:
        out = ("{0}/svm-model-{1}.sav").format(fp, i)
        logger.info("Saving SUPPORT-VECTOR-MACHINE-CLASSIFIER-MODEL at location: %s" % out)
        pickle.dump(clf, open(out, 'wb'))

      y_pred = clf.predict(X_test)
      report = generate_classification_report(fp, "svm-{0}".format(i), y_test, y_pred, save)

      logger.info("Support vector machine accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
      logger.info("Support vector machine classification report:{0}".format(report))
      logger.info("Support vector machine classifier took %s seconds" % (time.time() - start_time))
  else:
    X_train, X_test, y_train, y_test = split_data(data, fp, save, normalise=True)

    logger.info("Support vector machine classifier -- initialised")
    start_time = time.time()
    clf = svm.LinearSVC(verbose=2)
    clf.fit(X_train, y_train)

    if save:
      logger.info("Saving SUPPORT-VECTOR-MACHINE-CLASSIFIER-MODEL at location: %s" % out)
      pickle.dump(clf, open(out, 'wb'))

    y_pred = clf.predict(X_test)
    report = generate_classification_report(fp, "svm", y_test, y_pred, save)
    logger.info("Support vector machine accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
    logger.info("Support vector machine confusion matrix: {0}".format(report))
    logger.info("Support vector machine classifier took %s seconds" % (time.time() - start_time))


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
  # original_dataset, pruned_dataset = read_files([r"../Datasets/cleaned/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"], clean_data=False, prune=True)  # todo remove hardcode
  original_dataset = read_files([args.file_location], clean_data=False)  # todo remove hardcode

  pd.plotting.register_matplotlib_converters()  # todo convert this to a function
  original_dataset['Timestamp'] = pd.to_datetime(original_dataset['Timestamp'], format="%d/%m/%Y %H:%M:%S")
  original_dataset = original_dataset.sort_values(['Timestamp'], ascending=[True]).reset_index(drop=True)
  logger.info("Data is being sorted by time")
  support_vector_machine_classifier(original_dataset, fp=args.out, save=True, time_series=True)
  random_forest_classifier(original_dataset,fp=args.out, save=True, time_series=True)
