import argparse
import pandas as pd

from glob import glob


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


def get_numerical_data(data):
    categorical_columns = df.select_dtypes(['object'])
    return data.drop(categorical_columns, axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-location", help="location of dataset",
                        default=r"D:\Datasets\aws")  # todo remove default

    args = parser.parse_args()
    dataset_files = glob(args.data_location + r"\*.csv")

    df = read_files(dataset_files)
    print(df.info())

