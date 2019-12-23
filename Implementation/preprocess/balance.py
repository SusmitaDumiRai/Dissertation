# balances data, either upsample or downsample
import numpy as np
import pandas as pd
from sklearn.utils import resample


def sample_data(data, n_samples):
  return resample(data,
                  replace=True,
                  n_samples=n_samples,
                  random_state=27)


def sample_multiple_labels(data, benign):
  labels = data['Label'].unique()

  number_of_benign_labels = int(data.shape[0] // 2)
  number_of_each_attack_labels = int((data.shape[0] / 2) // (labels.shape[0] - 1)) # remove benign data

  resampled_data = []

  for label in labels:
    if label == benign:
      resampled_data.append(sample_data(data.loc[data['Label'] == label], n_samples=number_of_benign_labels))
    else:
      resampled_data.append(sample_data(data.loc[data['Label'] == label], n_samples=number_of_each_attack_labels))

  return pd.concat(resampled_data)
