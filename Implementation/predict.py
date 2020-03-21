import pickle
import argparse

import numpy as np

from glob import glob
from keras.models import load_model
from process_data import read_files, drop_columns
from ensemble import stacked_dataset, load_all_models


# predict using ensemble model
def singular_ensemble_predict(X, pretrained_model_paths, ensemble_model_path):
  print("Singular ensemble prediction")
  pretrained_models = load_all_models(pretrained_model_paths)
  with open(ensemble_model_path, 'rb') as f:
    ensemble_model = pickle.load(f)

  X_stacked = stacked_dataset(pretrained_models, X)  # produce pretrained model's prediction
  y_pred = ensemble_model.predict(X_stacked)  # produce final prediction using ensembles
  u, c = np.unique(y_pred, return_counts=True)
  print(np.asarray((u, c)))


def integrated_ensemble_predict(X, model_path):
  print("Integrated ensemble prediction")
  model = load_model(model_path)
  X = [X for _ in range(len(model.input))]
  y_pred = model.predict(X, batch_size=64)
  y_pred = np.argmax(y_pred, axis=1)
  u, c = np.unique(y_pred, return_counts=True)
  print(np.asarray((u, c)))


def nn_predict(X, model_paths):
  print("Single neural network prediction")
  for model_path in model_paths:
    print(model_path)
    model = load_model(model_path)
    model.summary()
    y_pred = model.predict(X, batch_size=64)
    # print(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    u, c = np.unique(y_pred, return_counts=True)
    print(np.asarray((u, c)))


def classic_predict(X, model_paths):
  print("Classic prediction")
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

  parser.add_argument("-i", "--integrated-ensemble", action="store_true")
  parser.add_argument("-s", "--singular-ensemble", action="store_true")
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

  if args.singular_ensemble:  # predict using singular ensemble model
    pretrained_model_loc = glob(args.multiple_model_location + r"/*.hdf5")
    assert len(pretrained_model_loc) >= 2  # atleast one model is required
    singular_ensemble_predict(X, pretrained_model_loc, args.ensemble_model_location)
  elif args.integrated_ensemble:  # predict using integrated model
    integrated_ensemble_predict(X, args.ensemble_model_location)
  elif args.neural_network:  # predict using singular neural network model
    print("neural network")
    nn_model_loc = glob(args.model_location + r"/*.hdf5")
    assert len(nn_model_loc) >= 1
    nn_predict(X, nn_model_loc)
  else:  # predict using classic machine learning techniques
    classic_model_loc = glob(args.model_location + r"/*.sav")
    assert len(classic_model_loc) >= 1
    classic_predict(X, classic_model_loc)
