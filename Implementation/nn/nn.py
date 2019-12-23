from keras.models import Model, Sequential, load_model
from keras.layers import Dense, LSTM, Input, concatenate, Dropout


def create_lstm_model(window_size,
                      feature_size,
                      activation):
  lstm_input = Input(shape=(window_size - 1, feature_size))

  lstm_model = LSTM(64, activation=activation, return_sequences=False)(lstm_input)
  return Model(inputs=lstm_input, outputs=lstm_model)


def create_mlp(feature_size,
               activation):
  label_input = Input(shape=(feature_size,))

  label_input_model = Dense(64, activation=activation)(label_input)
  return Model(inputs=label_input, outputs=label_input_model)


def load_dense_model():
  pretrained_model = load_model(r"C:\Users\kxd\Documents\Thesis_Susi\Dissertation\Implementation\models\dense-256-0.76.hdf5")
  pretrained_model.name = "pretrained_dense_model_1"

  for layer in pretrained_model.layers:
    layer.name = layer.name + str("_pretrained")

  print(pretrained_model.summary())

  # output = Dense(num_classes, name='output', activation=activation)(pretrained_model.layers[-2].output)

  new_model = Model(inputs=pretrained_model.inputs,
                    outputs=pretrained_model.layers[-2].output,
                    name="pretrained_dense_model_2")

  new_model.get_layer("dense_1_pretrained").trainable = False
  return new_model


def create_model(shape,
                 num_classes,
                 activation,
                 final_activation,
                 pre_trained=True):
  assert shape[0] > 1
  lstm_model = create_lstm_model(window_size=shape[0],
                                 feature_size=shape[1],
                                 activation=activation)

  if pre_trained:
    label_input_model = load_dense_model()
  else:
    label_input_model = create_mlp(feature_size=shape[1],
                                   activation=activation)

  combined_model = concatenate([label_input_model.output, lstm_model.output])

  connected_model = Dense(32, activation=activation)(combined_model)
  # connected_model = Dense(256, activation=activation)(connected_model)
  connected_model = Dense(num_classes, activation=final_activation)(connected_model)

  return Model(inputs=[lstm_model.input, label_input_model.input],
                      outputs=connected_model)
