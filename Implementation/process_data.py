import argparse
import pandas as pd

from glob import glob


def read_files(files, shuffle=False, prune=False):
    chunk_dfs = []
    for i in range(len(files)):
        print("Current file iteration {0} - {1}".format(i, files[i]))
        chunk_list = []
        chunk_size = 10 ** 6  # number of rows to read in one "chunk"

        for j, chunk in enumerate(pd.read_csv(files[i], chunksize=chunk_size, nrows=10000)):  # todo remove nrows
            print("%s: Chunk process %s" % (i, j))
            chunk = chunk[chunk.Label.str.contains('labels') == False]  # Removes csv rows that have headers repeating
            chunk_list.append(chunk)

        chunk_df = pd.concat(chunk_list)
        chunk_dfs.append(chunk_df)

    dataframe = pd.concat(chunk_dfs, sort=True).reset_index(drop=True)
    print("All files read, new dataframe created - dataframe shape {0} ".format(dataframe.shape))

    # if shuffle:  # todo test this code.
    #   dataframe = df.sample(frac=1).reset_index(drop=True)
    if prune:
        pruned_dataframe = drop_nan_rows(dataframe)

        return dataframe, pruned_dataframe

    return dataframe


def get_numerical_data(data):
    categorical_columns = data.select_dtypes(['object'])
    return data.drop(categorical_columns, axis=1)


def drop_nan_rows(data):  # some rows contain "Infinity values - removing them for now"
    # todo understand what are these infinite values.
    pruned_data = data
    pruned_data.dropna()
    for col in list(data):
        pruned_data = pruned_data.drop(pruned_data[pruned_data[col] == "Infinity"].index)

    pruned_data = pruned_data.reset_index(drop=True)
    print("NaNs & Infinity removed - dataframe shape {0}".format(pruned_data.shape))
    return pruned_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-location", help="location of dataset",
                        default=r"D:\Datasets\aws")  # todo remove default

    args = parser.parse_args()
    dataset_files = glob(args.data_location + r"\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")

    df = read_files(dataset_files)
    print(df.info())

