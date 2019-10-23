import sys
import argparse
import logging
import pandas as pd

from glob import glob

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/process_data-log.log",  # todo fix this
                            filemode='a',
                            format=formatter,
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

logger = logging.getLogger('urbanGUI')
logger.setLevel(logging.DEBUG)

def read_files(files, shuffle=False, prune=False):
    chunk_dfs = []
    for i in range(len(files)):
        print("Current file iteration {0} - {1}".format(i, files[i]))
        chunk_list = []
        chunk_size = 10 ** 6  # number of rows to read in one "chunk"

        for j, chunk in enumerate(pd.read_csv(files[i], chunksize=chunk_size)):  # todo remove nrows
            print("%s: Chunk process %s" % (i, j))
            chunk = chunk[chunk.Label.str.contains('labels') == False]  # Removes csv rows that have headers repeating
            chunk_list.append(chunk)

        chunk_df = pd.concat(chunk_list)
        chunk_dfs.append(chunk_df)

    dataframe = pd.concat(chunk_dfs, sort=True).reset_index(drop=True)
    logger.info("All files read, new dataframe created - dataframe shape: {0} ".format(dataframe.shape))

    # if shuffle:  # todo test this code.
    #   dataframe = df.sample(frac=1).reset_index(drop=True)
    if prune:
        pruned_dataframe = drop_nan_rows(dataframe)
        return dataframe, pruned_dataframe

    return dataframe


def get_numerical_data(data):
    categorical_columns = data.select_dtypes(['object'])
    return data.drop(categorical_columns, axis=1)


def get_null_dataframe(data):
    null_dfs = []
    null_columns = data.columns[data.isnull().any()]
    nan_data = data[data.isnull().any(axis=1)][null_columns].reset_index()

    for index, row in nan_data.iterrows():
        row_index = row['index']
        null_dfs.append(data.iloc[row_index].to_frame().transpose())

    null_dataframe = pd.concat(null_dfs).reset_index(drop=True)
    logger.info("Null dataframe created - dataframe shape: {0} ".format(null_dataframe.shape))
    return null_dataframe


def drop_nan_rows(data):  # some rows contain "Infinity values - removing them for now"
    # todo understand what are these infinite values.
    pruned_data = data
    pruned_data.dropna()
    for col in list(data):
        pruned_data = pruned_data.drop(pruned_data[pruned_data[col] == "Infinity"].index)

    pruned_data = pruned_data.reset_index(drop=True)
    logger.info("NaNs & Infinity removed - dataframe shape: {0}".format(pruned_data.shape))
    return pruned_data


def normalise_data(data):
    from sklearn import preprocessing
    values = data.values
    min_max_scaler = preprocessing.MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    return pd.DataFrame(values_scaled, columns=list(data))


if __name__ == "__main__":
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(formatter)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-location", help="location of dataset",
                        default=r"D:\Datasets\aws")  # todo remove default

    args = parser.parse_args()
    dataset_files = glob(args.data_location + r"\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv")

    df = read_files(dataset_files)
    print(df.info())

