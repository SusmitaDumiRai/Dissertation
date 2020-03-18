import numpy as np
import argparse
import logging
import sys
import cv2 as cv
import pandas as pd

from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, ReLU, LeakyReLU, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.initializers import random_uniform

import matplotlib.pyplot as plt

formatter = '%(asctime)s [%(filename)s:%(lineno)s - %(funcName)20s()] %(levelname)s | %(message)s'

logging.basicConfig(filename=r"out/gan-log.log",  # todo fix this
                    filemode='a',
                    format=formatter,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from Implementation.process_data import read_files, drop_columns

z_size = 100  # latent dimension - https://machinelearningmastery.com/how-to-interpolate-and-perform-vector-arithmetic-with-faces-using-a-generative-adversarial-network/
image_size = 77
arr = []

def build_model():
  opt_g = Adam(lr=0.0002, beta_1=0.5)
  opt_d = Adam(lr=0.0002, beta_1=0.5)

  G = Sequential(name="Generator")
  G.add(Dense(256, input_dim=z_size))
  G.add(LeakyReLU(0.2))
  G.add(Dense(512))
  G.add(LeakyReLU(0.2))
  G.add(Dense(1024))
  G.add(LeakyReLU(0.2))
  G.add(Dense(image_size, activation='tanh'))
  G.compile(loss="binary_crossentropy", optimizer=opt_g)

  D = Sequential(name="Discriminator")
  D.add(Dense(1024, input_dim=image_size))
  D.add(LeakyReLU(0.2))
  D.add(Dropout(0.3))
  D.add(Dense(512))
  D.add(LeakyReLU(0.2))
  D.add(Dropout(0.3))
  D.add(Dense(256))
  D.add(LeakyReLU(0.2))
  D.add(Dropout(0.3))
  D.add(Dense(1, activation="sigmoid"))
  D.compile(loss="binary_crossentropy", optimizer=opt_d, metrics=['accuracy'])

  D.trainable = False
  inputs = Input(shape=(z_size,))
  hidden = G(inputs)
  output = D(hidden)

  gan = Model(inputs, output)
  gan.compile(loss='binary_crossentropy', optimizer=opt_g)
  return gan, G, D

def train_main(X_train):
  # X_train = np.expand_dims(X_train, axis=3)  # reshape to N, feature_size

  print(X_train.shape)

  epochs = 200
  batch_size = 16
  gan, G, D = build_model()


  # training
  for e in range(epochs):
    print("epoch (%d / %d)" % (e + 1, epochs))

    valid = np.ones((batch_size, 1))  # all real data = 1
    fake = np.zeros((batch_size, 1))  # all fake data = 0

    # for _ in range(x_train.shape[0] // batch_size):
    # selecting random batch of images
    real_imgs = X_train[np.random.choice(X_train.shape[0], batch_size, replace=False)]
    noise = np.random.randn(batch_size, z_size)  # noise
    gen_imgs = G.predict(noise)  # generate new batch of images

    print(gen_imgs.shape)

    # train discriminator on sample
    D.trainable = True
    d_loss_real = D.train_on_batch(real_imgs, valid)  # returns loss
    d_loss_fake = D.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    noise = np.random.randn(batch_size, z_size)  # generate more noise

    # train generator
    D.trainable = False
    g_loss = gan.train_on_batch(noise, valid)

    # Plot the progress
    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (e, d_loss[0], 100 * d_loss[1], g_loss))

    sample_interval = 100
    # If at save interval => save generated image samples
    if e % sample_interval == 0:
      save_loss(e, d_loss[0], 100 * d_loss[1], g_loss)
      sample_images(e, G)

def save_loss(e, d_loss, d_acc, g_loss):
  with open("out/loss.csv", "a") as myfile:
    myfile.write("{0},{1},{2},{3}\n".format(e, d_loss, d_acc, g_loss))

def sample_images(epoch, generator):
  print(epoch)
  r, c = 3, 5
  noise = np.random.normal(0, 1, (r * c, z_size))
  gen_imgs = generator.predict(noise)

  # Rescale images 0 - 1
  gen_imgs = 0.5 * gen_imgs + 0.5
  arr.append(gen_imgs)
  gen_imgs = gen_imgs.reshape(gen_imgs.shape[0], 1, 77)
  gen_imgs = np.tile(gen_imgs, (77, 1))

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
      axs[i, j].axis('off')
      cnt += 1

  fig.savefig("images/%d.png" % epoch)
  plt.close()


def normalise_data(data):
  from sklearn import preprocessing
  values = data.values
  min_max_scaler = preprocessing.MinMaxScaler()
  values_scaled = min_max_scaler.fit_transform(values)
  return pd.DataFrame(values_scaled, columns=list(data)), min_max_scaler

def unnormalise(scaler, data):
  return scaler.inverse_transform(data)

if __name__ == '__main__':
  handler = logging.StreamHandler(sys.stdout)
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter(formatter)
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  parser = argparse.ArgumentParser()
  parser.add_argument("-f", "--file-location", help="location to files.")
  parser.add_argument("-o", "--out", help="out folder path", default="out/")

  args = parser.parse_args()

  original_dataset = read_files([args.file_location], clean_data=False)
  original_dataset = drop_columns(original_dataset, ['Label'])

  normalised_data, scaler = normalise_data(original_dataset)
  normalised_data = normalised_data.to_numpy()

  #mean_data = normalised_data.mean(axis=0)
  #mean_data = np.tile(mean_data, (77, 1))
  #cv.imshow("test", mean_data)
  #cv.waitKey(0)

  #cv.imwrite(r"out/attack.png", 255*mean_data)

  train_main(normalised_data)
  numpy_arr = np.asarray(arr)
  numpy_arr = numpy_arr.reshape(numpy_arr.shape[0] * numpy_arr.shape[1], numpy_arr.shape[2]) # could transpose instead

  generated_data = unnormalise(scaler, numpy_arr)  # remove first unnecessary column




