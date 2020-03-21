import numpy as np

from keras.models import load_model
from sklearn.linear_model import LogisticRegression

def load_all_models(models_path):
  models = []
  for model_path in models_path:
    models.append(load_model(model_path))
    print("loaded model: {0}".format(model_path))

  return models

# make prediction using pretrained models and use that for input for final model.
def stacked_dataset(models, X_input):
  X_stack = None
  for model in models:
    # make prediction
    y_pred = model.predict(X_input, verbose=0)
    # stack predictions into [rows, members, probabilities]
    if X_stack is None:
      X_stack = y_pred
    else:
      X_stack = np.dstack((X_stack, y_pred))

  # flatten predictions to [rows, members x probabilities]
  X_stack = X_stack.reshape((X_stack.shape[0], X_stack.shape[1] * X_stack.shape[2]))
  return X_stack


# fit a logistic regression with inputs from pretrained model's predictions
def train_ensemble(pretrained_models, X_input, y_input):
  # create dataset using ensemble
  X_stacked = stacked_dataset(pretrained_models, X_input)
  # fit standalone model
  ensemble_model = LogisticRegression()
  ensemble_model.fit(X_stacked, y_input)
  return ensemble_model