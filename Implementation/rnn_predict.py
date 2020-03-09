import argparse

from numpy import np
from keras.models import load_model
from process_data import read_files

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--data-location", required=True)
  parser.add_argument("-m", "--model-location", required=True)
  # parser.add_argument("-l", "--label", required=True)

  args = parser.parse_args()

  data = read_files([args.data_location], clean_data=False)
  model = load_model(args.model_location)

  Y = data['Label'].to_numpy()
