import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# todo import logging
from glob import glob


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


def visualise_boxplot(data, save=False):
    from sklearn.preprocessing import MinMaxScaler

    values = data.values
    min_max_scaler = MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    scaled_data = pd.DataFrame(values_scaled,
                               columns=list(data))

    scaled_data.boxplot(column=list(scaled_data), figsize=(25, 5), rot=90)

    if save:
        plt.savefig(r'out/boxplot.png')

    plt.show()


def visualise_pie(data, save=False):
    protocols = data.groupby(['Label']).size().reset_index(name='count')

    protocols['count_norm'] = [float(i) / sum(protocols['count'])
                               for i in protocols['count']]

    pr_x = protocols['Label']
    pr_y = protocols['count']
    percent = 100. * pr_y / pr_y.sum()

    patches, texts = plt.pie(pr_y, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.3f} %'.format(i, j) for i, j in zip(pr_x, percent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, protocols['count']),
                                             key=lambda x: x[2],
                                             reverse=True))

    plt.legend(patches, labels, loc='best',
               bbox_to_anchor=(-0.1, 1.), fontsize=10)
    plt.title('Labels')

    if save:
        plt.savefig(r'out/pie.png')

    plt.show()


def read_files(files, shuffle=True):
    chunk_dfs = []
    for i in range(len(files)):
        print("Current file iteration {0} - {1}".format(i, files[i]))
        chunk_list = []
        chunk_size = 10 ** 2  # number of rows to read in one "chunk"

        for j, chunk in enumerate(pd.read_csv(files[i], chunksize=chunk_size, nrows=10,
                                              skiprows=range(1, 5000, 20))):  # todo remove nrows
            print("%s: Chunk process %s" % (i, j))
            chunk = chunk[chunk.Label.str.contains('labels') == False]  # Removes csv rows that have headers repeating
            chunk_list.append(chunk)

        chunk_df = pd.concat(chunk_list)
        chunk_dfs.append(chunk_df)

    return pd.concat(chunk_dfs, sort=True).reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-location", help="location of dataset",
                        default=r"D:\Datasets\aws")  # todo remove default

    args = parser.parse_args()
    dataset_files = glob(args.data_location + r"\*.csv")
    # assert len(dataset_files) == 10  # todo remove assert

    df = read_files(dataset_files)
    categorical_columns = df.select_dtypes(['object'])

    numerical_df = df.drop(categorical_columns, axis=1)

    # visualise_pie(df, True)
    # visualise_boxplot(numerical_df, True)
    print(df.describe().to_csv(r'out/describe.csv'))

    # random_forest_classifier(df)
