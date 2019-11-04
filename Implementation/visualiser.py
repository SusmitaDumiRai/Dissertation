import logging
import sys
import argparse

import matplotlib.pyplot as plt

from process_data import read_files, get_numerical_data, drop_nan_rows, normalise_data, \
  get_null_dataframe

# TODO FIX LOGGING IN THIS FILE.
formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/process_data-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger('urbanGUI')
logger.setLevel(logging.DEBUG)


def visualise_boxplot(data, fp, normalise=True, save=False):
  name = fp + "boxplot.png"
  if normalise:
    data = normalise_data(data)
  data.boxplot(column=list(data), figsize=(30, 5), rot=90)
  plt.tight_layout()

  if save:
    plt.savefig(name)
    logger.info("Boxplot visualised at location: {0}".format(name))

  plt.show()


def visualise_pie(data, fp, save=False):
  name = fp + "pie.png"
  protocols = data.groupby(['Label']).size().reset_index(name='count')

  protocols['count_norm'] = [float(i) / sum(protocols['count'])
                             for i in protocols['count']]

  pr_x = protocols['Label']
  pr_y = protocols['count']
  percent = 100. * pr_y / pr_y.sum()

  patches, texts = plt.pie(pr_y, startangle=80, radius=1.2)
  labels = ['{0} - {1:1.3f} %'.format(i, j) for i, j in zip(pr_x, percent)]

  sort_legend = True
  if sort_legend:
    patches, labels, dummy = zip(*sorted(zip(patches, labels, protocols['count']),
                                         key=lambda x: x[2],
                                         reverse=True))

  plt.legend(patches, labels, loc='best',
             bbox_to_anchor=(0.8, 1.), fontsize=10)
  plt.title('Labels')
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


if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  parser = argparse.ArgumentParser()

  parser.add_argument("-f", "--filepath", help="filepath to csv", required=True)  # todo turn on recursive.
  parser.add_argument("-n", "--nans", help="visualise nans", action="store_true")
  parser.add_argument("-b", "--boxplot", help="visualise boxplot", action="store_true")
  parser.add_argument("-i", "--pie", help="visualise pie", action="store_true")
  parser.add_argument("-p", "--prune", help="prune nans", action="store_true")  # todo prune nans = remove them
  parser.add_argument("-o", "--out", help="outfile path to save", default="out/")
  args = parser.parse_args()

  dataset = read_files([args.filepath])  # todo remove hardcode
  print(dataset.head())

  if args.prune:
    pruned_dataset = drop_nan_rows(dataset)

  if args.nans:
    visualise_NaNs(dataset, args.out, save=True)

  if args.boxplot:
    visualise_boxplot(get_numerical_data(dataset), args.out, save=True)

  if args.pie:
    visualise_pie(dataset, args.out, save=True)



