import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from glob import glob

def read_files(files, shuffle=True):
    chunk_dfs = []
    for i in range(len(files)):
        print("Current file iteration {0} - {1}".format(i, files[i]))
        chunk_list = []
        chunk_size = 10 ** 2  # number of rows to read in one "chunk"
        for j, chunk in enumerate(pd.read_csv(files[i], chunksize=chunk_size, nrows=100)):  # todo remove nrows
            print("%s: Chunk process %s" % (i, j))
            chunk_list.append(chunk)

        chunk_df = pd.concat(chunk_list)
        print(len(list(chunk_df)))
        chunk_dfs.append(chunk_df)

    x = pd.concat(chunk_dfs, sort=True)
    print(len(list(x)))
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-location", help="location of dataset",
                        default=r"D:\Datasets\aws")  # todo remove default

    args = parser.parse_args()
    dataset_files = glob(args.data_location + r"\*.csv")
    # assert len(dataset_files) == 10  # todo remove assert

    df = read_files(dataset_files)



