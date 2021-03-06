import logging
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO FIX LOGGING IN THIS FILE.
formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/visualiser-log.log",
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from process_data import read_files, get_numerical_data, drop_nan_rows, normalise_data, \
  get_null_dataframe

def visualise_boxplot(data, fp, normalise=True, save=False):
  name = fp + "boxplot.png"
  if normalise:
    logger.info("Data being normalised: {0}".format(normalise))
    data = normalise_data(data)
  data.boxplot(column=list(data), figsize=(30, 5), rot=90)
  plt.tight_layout()

  if save:
    plt.savefig(name)
    logger.info("Boxplot visualised at location: {0}".format(name))

  plt.show()


def visualise_pie(data, fp, attribute='Protocol', save=False):
  name = fp + attribute + "_pie.png"
  attr_df = data.groupby([attribute]).size().reset_index(name='count')

  attr_df['count_norm'] = [float(i) / sum(attr_df['count'])
                             for i in attr_df['count']]

  pr_x = attr_df[attribute]
  pr_y = attr_df['count']
  percent = 100. * pr_y / pr_y.sum()

  patches, texts = plt.pie(pr_y, startangle=80, radius=1.2)
  labels = ['{0} - {1:1.3f} %'.format(i, j) for i, j in zip(pr_x, percent)]

  sort_legend = True
  if sort_legend:
    patches, labels, dummy = zip(*sorted(zip(patches, labels, attr_df['count']),
                                         key=lambda x: x[2],
                                         reverse=True))

  plt.legend(patches, labels, loc='best',
             bbox_to_anchor=(0.8, 1.), fontsize=10)
  plt.title(attribute)
  plt.tight_layout()

  if save:
    plt.savefig(name)
    logger.info("Pie chart visualised at location: {0}".format(name))

  plt.show()


def visualise_NaNs(data, fp, save=False):
  name = fp + "nans.png"
  null_sum = data.isnull().sum()
  null_sum.plot.bar(figsize=(20, 5))

  plt.tight_layout()

  if save:
    plt.savefig(name)
    logger.info("NaN graph visualised at location: {0}".format(name))

  plt.show()



def visualise_malware_time_chunks(data, fp, save=False):
  name = fp + "barplot.png"
  pd.plotting.register_matplotlib_converters()  # todo convert this to a function
  data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d/%m/%Y %H:%M:%S")
  data = data.sort_values(['Timestamp', 'Label'], ascending=[True, False])
  # data = data.groupby('Timestamp').apply(lambda x: x.sort_values('Label'))
  # print(data['Timestamp'], data['Label'])
  # data = data.sort_values(by='Timestamp').groupby('Label')
  # print(type(data))
  # print(data['Timestamp'], data['Label'])
  # for index, row in data.iterrows():
  #  print("index: {0}, label: {1}, timestamp:{2}".format(index, row['Label'], row['Timestamp']))

  count_data = data.groupby(['Timestamp', 'Label']).size().reset_index(name='count per second')
  # select_columns_data = count_data[['Timestamp', 'Label', 'count per second']].copy()
  for index, row in count_data.iterrows():
    print("index: {0}, label: {1}, timestamp: {2}, count:{3}".format(index, row['Label'], row['Timestamp'],
                                                                     row['count per second']))
  # count_malware_data = count_data[count_data['Label'] != "Benign"]
  # count_benign_data = count_data[count_data['Label'] == "Benign"]

  # plt.bar(count_malware_data['Timestamp'], count_malware_data['count per second'], c='r')
  # plt.bar(count_benign_data['Timestamp'], count_benign_data['count per second'], c='g')
  # count_data.plot(x='Timestamp', y=['count per second', 'Label'], kind='bar')

  if save:
    plt.savefig(name)
    logger.info("Bargraph visualised at location: {0}".format(name))
  plt.show()
def visualise_timeseries(data, fp, attributes, save=False):
  name = fp + "timeseries.png"

  if attributes is None:
    logger.error("No attributes defined.")
    return None

  pd.plotting.register_matplotlib_converters()
  data['Timestamp'] = pd.to_datetime(data['Timestamp'], format="%d/%m/%Y %H:%M:%S")
  data = data.sort_values(by='Timestamp')

  data.set_index('Timestamp', inplace=True)

  data = data[~data.index.duplicated()]
  data = data.asfreq(freq='30s')

  plt.tight_layout()
  logger.info("Plotting {0} attributes".format(attributes))
  data.plot(y=attributes, subplots=True, figsize=(30, 20))

  if save:
    plt.savefig(name)
    logger.info("Time series graph visualised at location: {0}".format(name))

  plt.show()

if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  parser = argparse.ArgumentParser()

  default_attributes = ["Fwd Pkt Len Std", "Bwd Pkt Len Std"]  # todo fix.
  parser.add_argument("-f", "--filepath", help="filepath to csv", required=True)  # todo turn on recursive.
  parser.add_argument("-n", "--nans", help="visualise nans", action="store_true")
  parser.add_argument("-b", "--boxplot", help="visualise boxplot", action="store_true")
  parser.add_argument("-i", "--pie", help="visualise pie", action="store_true")
  parser.add_argument("-p", "--prune", help="prune nans", action="store_true")  # todo prune nans = remove them
  parser.add_argument("-t", "--timeseries", help="visualise time series",
                      action="store_true")  # todo prune nans = remove them
  parser.add_argument("-o", "--out", help="outfile path to save", default="out/")
  parser.add_argument("-a", "--attributes", help="attribute names, example: a b c 'a b c'",
                      nargs="+", default=default_attributes)
  args = parser.parse_args()

  dataset = read_files([args.filepath], clean_data=False)

  # todo fix this.
  benign_only = False
  malware_only = False

  out = args.out

  visualise_malware_time_chunks(dataset, out, save=True)

  if benign_only:
    out = args.out + "benign_"
    dataset = dataset.loc[dataset['Label'] == 'Benign']
    logger.info("Only {0} data selected".format(dataset['Label'].unique(), dataset.shape))
  elif malware_only:
    out = args.out + "malware_"
    dataset = dataset.loc[dataset['Label'] != 'Benign']
    logger.info("Only {0} data selected - {1}".format(dataset['Label'].unique(), dataset.shape))

  if args.prune:
    pruned_dataset = drop_nan_rows(dataset)

  if args.nans:
    visualise_NaNs(dataset, out, save=True)

  if args.boxplot:
    visualise_boxplot(get_numerical_data(dataset), out, save=True)

  if args.pie:
    visualise_pie(dataset, out, save=True)

  if args.timeseries:
    visualise_timeseries(dataset, out, args.attributes, save=True)
