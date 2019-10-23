import sys
import pickle
import time
import logging

from sklearn import metrics
from sklearn.tree import export_graphviz
from sklearn.preprocessing import LabelEncoder


formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/classifier-log.log",  # todo fix this
                            filemode='a',
                            format=formatter,
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logger = logging.getLogger('urbanGUI')
logger.setLevel(logging.DEBUG)

from Implementation.process_data import read_files, normalise_data

def label_encode_class(data):
    def label_encoder_mapping(le):
        # todo save this label encoder later for prediction.
        return dict(zip(le.classes_, le.transform(le.classes_)))

    assert data.shape[1] == 1  # only one column - output label.

    le = LabelEncoder()
    le.fit(data)

    print("Label encoder mapping {0}".format(label_encoder_mapping(le)))

    return le.transform(data).ravel()


def split_data(data, test_size=0.3, normalise=False):
    from sklearn.model_selection import train_test_split
    categorical_columns = data.select_dtypes(['object'])

    output = ['Label']
    inputs = [label for label in list(data) if label not in output and label not in categorical_columns]

    if normalise:
        logger.info("")
        data[inputs] = normalise_data(data[inputs])

    X = data[inputs]
    y = label_encode_class(data[output])

    return train_test_split(X, y, test_size=test_size)  # 70% training and 30% test


def random_forest_classifier(data, save=False, fp=r"out/rf-model.sav"):
    from sklearn.ensemble import RandomForestClassifier
    X_train, X_test, y_train, y_test = split_data(data)

    logger.info("Random forest classifier -- initialised")
    start_time = time.time()
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    if save:
        logger.info("Saving RANDOM-FOREST-CLASSIFIER-MODEL at location: %s" % fp)
        pickle.dump(clf, open(fp, 'wb'))

    y_pred = clf.predict(X_test)
    logger.info("Random forest classifier accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
    logger.info("Random forest classifier took %s seconds" % (time.time() - start_time))


def support_vector_machine_classifier(data, save=False, fp=r"out/svm-model.sav"):
    from sklearn import svm

    X_train, X_test, y_train, y_test = split_data(data, normalise=True)

    start_time = time.time()
    clf = svm.SVC(kernel='linear', verbose=1)  # Linear Kernel
    clf.fit(X_train, y_train)

    if save:
        logger.info("Saving SUPPORT-VECTOR-MACHINE-CLASSIFIER-MODEL at location: %s" % fp)
        pickle.dump(clf, open(fp, 'wb'))

    y_pred = clf.predict(X_test)
    logger.info("Support vector machine accuracy: %s" % metrics.accuracy_score(y_test, y_pred))
    logger.info("Support vector machine classifier took %s seconds" % (time.time() - start_time))


if __name__ == '__main__':
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(formatter)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    original_dataset, pruned_dataset = read_files(
        [r"C:\Users\908928.TAWE\aws\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"], prune=True)  # todo remove hardcode

    random_forest_classifier(pruned_dataset, True)

