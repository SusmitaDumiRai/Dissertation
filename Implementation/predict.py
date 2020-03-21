import pickle
import argparse

import numpy as np

from glob import glob
from keras.models import load_model
from process_data import read_files, drop_columns
from ensemble import stacked_dataset, load_all_models

from sklearn.metrics import classification_report

# predict using ensemble model
def ensemble_predict(pretrained_models, ensemble_model, X):
  # create dataset using ensemble
  X_stacked = stacked_dataset(pretrained_models, X)  # produce pretrained model's prediction

  y_pred = ensemble_model.predict(X_stacked)  # produce final prediction using ensembles
  return y_pred

def nn_predict(X, y, model_paths):
  print(model_paths)
  for model_path in model_paths:
    print(model_path)
    model = load_model(model_path)
    model.summary()
    y_pred = model.predict(X, batch_size=64)
    # print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    u, c = np.unique(y_pred, return_counts=True)
    print(np.asarray((u, c)))

    # print(y_pred)
    count_true = np.count_nonzero(y_pred == y[0])
    # print(count_true)

    # print(y_pred.shape)
    # print(y.shape)
    # report = classification_report(y, y_pred)
    # print(report)


def classic_predict(X, y, model_paths):
  for model_path in model_paths:
    print(model_path)
    with open(model_path, 'rb') as f:
      model = pickle.load(f)
    y_pred = model.predict(X)
    # print(y_pred)
    u, c = np.unique(y_pred, return_counts=True)
    print(np.asarray((u, c)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data-location", required=True)
  parser.add_argument("-m", "--model-location")  # for single prediction
  parser.add_argument("-p", "--multiple-model-location")  # multiple pretrained models for ensemble

  parser.add_argument("-g", "--ensemble-model-location")
  parser.add_argument("-e", "--ensemble", action="store_true")
  parser.add_argument("-n", "--neural-network", action="store_true")
  parser.add_argument("-c", "--classic", action="store_true")
  parser.add_argument("-l", "--label-number", type=int, required=True)  # todo easier to make them pass lstm encoding
  args = parser.parse_args()

  data = read_files([args.data_location], clean_data=False)
  # print(data.shape)

  y = np.full((data.shape[0],), args.label_number)  # create true y of size of data

  try:
    data = drop_columns(data, ['Label', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Timestamp'])
  except:
    print("Could not drop all columns")
  X = data.to_numpy()
  print(X.shape)
  print(y.shape)

  if args.ensemble:  # predict using multiple neural network models
    model_loc = glob(args.multiple_model_location + r"/*.hdf5")
    assert len(model_loc) >= 2  # atleast one model is required
    pretrained_models = load_all_models(model_loc)
    with open(args.ensemble_model_location, 'rb') as f:
      ensemble_model = pickle.load(f)
    # evaluate model on test set
    ensemble_y_pred = ensemble_predict(pretrained_models, ensemble_model, X)
    u, c = np.unique(ensemble_y_pred, return_counts=True)
    print(np.asarray((u, c)))
  elif args.neural_network:  # predict using singular neural network model
    print("neural network")
    model_loc = glob(args.model_location + r"/*.hdf5")
    assert len(model_loc) >= 1
    nn_predict(X, y, model_loc)
  else:  # predict using classic machine learning techniques
    model_loc = glob(args.model_location + r"/*.sav")
    assert len(model_loc) >= 1
    classic_predict(X, y, model_loc)
