from pandas import pd


def random_forest_classifier(data):
    from sklearn import metrics

    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    categorical_columns = data.select_dtypes(['object'])
    # print(list(categorical_columns))

    # data = data.select_dtypes(exclude=['object'])  # for now remove categorical data

    output = ['Label']
    inputs = [label for label in list(data) if label not in output and label not in categorical_columns]

    le = LabelEncoder()
    le.fit(data[output])

    X = data[inputs]
    y = pd.DataFrame(le.transform(data[output]))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))