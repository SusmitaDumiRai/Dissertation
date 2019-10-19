# for ease of use, remove missing values.
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from glob import glob


def visualise_pie(data):
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
    plt.title('Frequency of protocols')
    plt.show()


def read_files(files, shuffle=True):
    chunk_dfs = []
    for i in range(len(files)):
        print("Current file iteration {0} - {1}".format(i, files[i]))
        chunk_list = []
        chunk_size = 10 ** 2  # number of rows to read in one "chunk"

        for j, chunk in enumerate(pd.read_csv(files[i], chunksize=chunk_size, nrows=3000,
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
    visualise_pie(df)

