from keras.models import Model
from keras.layers import Dense, LSTM, Input, concatenate


def create_lstm_model(window_size,
                      feature_size,
                      activation):
  lstm_input = Input(shape=(window_size - 1, feature_size))

  lstm_model = LSTM(8, activation=activation, return_sequences=False)(lstm_input)
  return Model(inputs=lstm_input, outputs=lstm_model)


def create_label_input_model(feature_size,
                             activation):
  label_input = Input(shape=(1, feature_size))

  label_input_model = Dense(8, activation=activation)(label_input)
  return Model(inputs=label_input, outputs=label_input_model)


def create_model(shape,
                 num_classes,
                 activation,
                 final_activation):
  assert shape[0] > 1
  lstm_model = create_lstm_model(window_size=shape[0],
                                 feature_size=shape[1],
                                 activation=activation)

  label_input_model = create_label_input_model(feature_size=shape[1],
                                               activation=activation)

  combined_model = concatenate([label_input_model.output, lstm_model.output])

  connected_model = Dense(256, activation=activation)(combined_model)
  connected_model = Dense(num_classes, activation=final_activation)(connected_model)

  return Model(inputs=[lstm_model.input, label_input_model.input],
                      outputs=connected_model)
