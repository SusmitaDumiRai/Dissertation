import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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