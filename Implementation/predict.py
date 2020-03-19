import pickle
import argparse

import numpy as np

from keras.models import load_model
from process_data import read_files, drop_columns

from sklearn.metrics import classification_report



def nn_predict(X, y, model_path):
  model = load_model(model_path)
  model.summary()
  y_pred = model.predict(X, batch_size=64)
  print(y_pred)
  y_pred = np.argmax(y_pred, axis=1)
  u, c = np.unique(y_pred, return_counts=True)
  print(np.asarray((u, c)))

  # print(y_pred)
  count_true = np.count_nonzero(y_pred == y[0])
  print(count_true)

  # print(y_pred.shape)
  # print(y.shape)
  # report = classification_report(y, y_pred)
  # print(report)

def classic_predict(X, y, model_path):
  with open(model_path, 'rb') as f:
    model = pickle.load(f)
  y_pred = model.predict(X)
  # print(y_pred)
  u, c = np.unique(y_pred, return_counts=True)
  print(np.asarray((u, c)))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data-location", required=True)
  parser.add_argument("-m", "--model-location", required=True)
  # parser.add_argument("-l", "--label", required=True)
  # parser.add_argument("-t", "--type", required=True) # filter to RF, SVM, NN (can do mutual exclusion but lazy)
  parser.add_argument("-n", "--neural-network", action="store_true")
  parser.add_argument("-c", "--classic", action="store_true")
  parser.add_argument("-l", "--label-number", type=int, required=True) # todo easier to make them pass lstm encoding 
  args = parser.parse_args()

  data = read_files([args.data_location], clean_data=False)
  print(data.shape)
  # model = load_model(args.model_location)

  y = np.full((data.shape[0],), args.label_number) # create true y of size of data

  # y = data['Label'].to_numpy()

  data = drop_columns(data, ['Label', 'Flow ID', 'Src IP', 'Src Port','Dst IP', 'Dst Port', 'Timestamp'])
  X = data.to_numpy()
  print(X.shape)
  print(y.shape)
  if args.neural_network:
    nn_predict(X, y, args.model_location)
  else:
    classic_predict(X, y, args.model_location)
