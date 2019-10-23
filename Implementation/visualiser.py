import missingno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Implementation.process_data import read_files, get_numerical_data, drop_nan_rows


def visualise_boxplot(data, save=False):
    from sklearn.preprocessing import MinMaxScaler

    numerical_data = get_numerical_data(data)
    values = numerical_data.values
    min_max_scaler = MinMaxScaler()
    values_scaled = min_max_scaler.fit_transform(values)
    scaled_data = pd.DataFrame(values_scaled,
                               columns=list(numerical_data))

    scaled_data.boxplot(column=list(scaled_data), figsize=(30, 5), rot=90)
    plt.tight_layout()

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
        plt.savefig(r'out/pie.png')

    plt.show()


def visualise_NaNs(data, save=False):
    null_sum = data.isnull().sum()
    null_sum.plot.bar(figsize=(20, 5))

    plt.tight_layout()

    if save:
        plt.savefig(r'out/nans.png')

    plt.show()


def get_infinity_index(data):
    inf_index = []
    cat_data = data.select_dtypes(include='object')

    for col in list(cat_data):
        print(col)
        inf_index.append(list(dataset[dataset[col].str.contains("Infinity") == True].index.values))

    return inf_index

if __name__ == '__main__':
    dataset = read_files([r"C:\Users\908928.TAWE\aws\Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"])  # todo remove hardcode
    visualise_NaNs(dataset)

    pruned_dataset = drop_nan_rows(dataset)
    visualise_NaNs(pruned_dataset)
    visualise_boxplot(pruned_dataset)
    visualise_pie(pruned_dataset)

