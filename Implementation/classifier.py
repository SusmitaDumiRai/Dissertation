import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

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
        data[inputs] = normalise_data(data[inputs])

    X = data[inputs]
    y = label_encode_class(data[output])

    return train_test_split(X, y, test_size=test_size)  # 70% training and 30% test


def random_forest_classifier(data):
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test = split_data(data)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Random forest classifier accuracy:", metrics.accuracy_score(y_test, y_pred))


def support_vector_machine_classifier(data):
    from sklearn import svm

    X_train, X_test, y_train, y_test = split_data(data, normalise=True)

    clf = svm.SVC(kernel='linear', verbose=1)  # Linear Kernel
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Support vector machine accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    original_dataset, pruned_dataset = read_files(
        [r"C:\Users\908928.TAWE\aws\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"], prune=True)  # todo remove hardcode

    support_vector_machine_classifier(pruned_dataset)