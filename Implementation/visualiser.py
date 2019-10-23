import pandas as pd
import matplotlib.pyplot as plt

from Implementation.process_data import read_files, get_numerical_data, drop_nan_rows, normalise_data, get_null_dataframe


def visualise_boxplot(data, normalise=True, save=False, fp=r"out/boxplot.png"):
    if normalise:
        data = normalise_data(data)
    data.boxplot(column=list(data), figsize=(30, 5), rot=90)
    plt.tight_layout()

    if save:
        plt.savefig(fp)

    plt.show()


def visualise_pie(data, save=False, fp=r"out/pie.png"):
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
        plt.savefig(fp)

    plt.show()


def visualise_NaNs(data, save=False, fp=r'out/nans.png'):
    null_sum = data.isnull().sum()
    null_sum.plot.bar(figsize=(20, 5))

    plt.tight_layout()

    if save:
        plt.savefig(fp)

    plt.show()


def write_csv(data, fp=r"out/null.csv"):
    data.to_csv(fp, index=False)

if __name__ == '__main__':
    dataset = read_files([r"C:\Users\908928.TAWE\aws\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"])  # todo remove hardcode
    null_dataset = get_null_dataframe(dataset)


    # pruned_dataset = drop_nan_rows(dataset)

    # null_columns = pruned_dataset.columns[pruned_dataset.isnull().any()]
    # print(pruned_dataset[pruned_dataset.isnull().any(axis=1)][null_columns].head())

    # visualise_NaNs(null_dataset)
    # visualise_boxplot(get_numerical_data(null_dataset), normalise=False)
    write_csv(null_dataset)
    visualise_pie(null_dataset, save=True, fp=r"out/nan-labels")

