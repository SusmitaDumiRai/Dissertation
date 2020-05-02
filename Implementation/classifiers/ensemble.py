import numpy as np

from keras.utils import plot_model
from keras.models import load_model, Model
from keras.layers import Dense, concatenate
from sklearn.linear_model import LogisticRegression

# https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/

def load_all_models(models_path):
  return [load_model(model_path) for model_path in models_path]

# ----------------------- singular stacked ensemble ------------------------------
# make prediction using pretrained models and use that for input for final model.
def stacked_dataset(models, X_input):
  X_stack = None
  for i, model in enumerate(models):
    print("predicting model {0}".format(i))
    # make prediction
    y_pred = model.predict(X_input, verbose=0)
    # stack predictions into [rows, members, probabilities]
    if X_stack is None:
      X_stack = y_pred
    else:
      X_stack = np.dstack((X_stack, y_pred))
  print("X_stack shape before reformat: {0}".format(X_stack.shape))
  # flatten predictions to [rows, members x probabilities]
  X_stack = X_stack.reshape((X_stack.shape[0], X_stack.shape[1] * X_stack.shape[2]))
  print("X_stack shape after reformat: {0}".format(X_stack.shape))

  return X_stack

# fit a logistic regression with inputs from pretrained model's predictions
def train_ensemble_singular(pretrained_models, X_input, y_input):
  # create dataset using ensemble
  X_stacked = stacked_dataset(pretrained_models, X_input)
  # fit standalone model
  print("Train ensemble - model input shape: {0}".format(X_stacked.shape))
  y_input = np.argmax(y_input, axis=1)
  ensemble_model = LogisticRegression()
  ensemble_model.fit(X_stacked, y_input)
  return ensemble_model


# ----------------------- integrated stacked ensemble ------------------------------
def integrated_stacked_ensemble(models, out, num_classes):
  # update all layers in all models to not be trainable
  for i, model in enumerate(models):
    print("Freezing model {0}'s weights".format(i))
    for layer in model.layers:
      layer.trainable = False  # freeze weights when retraining
      layer.name = 'ensemble_{0}_{1}'.format((i + 1), layer.name)  # rename to avoid 'unique layer name' issue

  # define multi-headed input
  ensemble_visible = [model.input for model in models]
  # concatenate merge output from each model
  ensemble_outputs = [model.output for model in models]
  merge = concatenate(ensemble_outputs)
  hidden = Dense(256, activation='relu')(merge)
  output = Dense(num_classes, activation='softmax')(hidden)
  model = Model(inputs=ensemble_visible, outputs=output)
  # plot graph of ensemble
  print("Model plotted at location: {0}".format(out))
  plot_model(model, show_shapes=True, to_file=r'{0}/model_graph.png'.format(out))

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  model.summary()
  return model

def train_ensemble_integrated(model, inputX, y, epochs=100):
  # prepare input data
  X = [inputX for _ in range(len(model.input))]
  print("training model")
  model.fit(X, y, epochs=epochs)
  return model
