from Implementation.visualiser import read_files
from sklearn.preprocessing import LabelEncoder

def label_encode_class(data):
    def label_encoder_mapping(le):
        # todo save this label encoder later for prediction.
        return dict(zip(le.classes_, le.transform(le.classes_)))

    assert data.shape[1] == 1

    le = LabelEncoder()
    le.fit(data)

    print("Label encoder mapping {0}".format(label_encoder_mapping(le)))

    return le.transform(data).ravel()


def random_forest_classifier(data):
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    categorical_columns = data.select_dtypes(['object'])

    output = ['Label']
    inputs = [label for label in list(data) if label not in output and label not in categorical_columns]

    X = data[inputs]
    y = label_encode_class(data[output])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    original_dataset, pruned_dataset = read_files(
        [r"C:\Users\908928.TAWE\aws\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"], prune=True)  # todo remove hardcode

    random_forest_classifier(pruned_dataset)