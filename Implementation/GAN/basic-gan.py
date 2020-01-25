import numpy as np
from keras.datasets import mnist
from keras.models import Sequential, Model, Input
from keras.layers import Dense, ReLU, LeakyReLU, Dropout
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.initializers import random_uniform

import matplotlib.pyplot as plt

z_size = 10
image_width = 28
image_height = 28
image_size = 28 * 28


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
  G.compile(loss="categorical_crossentropy", optimizer=opt_g)

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


def train_main():
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
  x_train = x_train.reshape(x_train.shape[0], image_size) / 255
  x_train = (x_train * 2) - 1

  epochs = 300
  batch_size = 128
  gan, G, D = build_model()

  z_test = np.random.randn(100, z_size)
  g_cost = []
  d_cost = []

  # training
  for e in range(epochs):
    print("epoch (%d / %d)" % (e + 1, epochs))
    dcost = 0
    gcost = 0

    valid = np.ones((batch_size, 1))  # all real data = 1
    fake = np.zeros((batch_size, 1))  # all fake data = 0

    # for _ in range(x_train.shape[0] // batch_size):
    # selecting random batch of images
    real_imgs = x_train[np.random.choice(x_train.shape[0], batch_size, replace=False)]
    noise = np.random.randn(batch_size, z_size)  # noise
    gen_imgs = G.predict(noise)  # generate new batch of images

    print(gen_imgs.shape)

    # label 0 means fake data, 0.9 means real data
    """
    xd = np.concatenate((real_imgs, gen_imgs), axis=0)
    yd = np.zeros(batch_size*2)
    yd[0:batch_size] = 0.9
    """

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

    sample_interval = 50
    # If at save interval => save generated image samples
    if e % sample_interval == 0:
      sample_images(e, G)


def sample_images(epoch, generator):
  print(epoch)
  r, c = 2, 5
  noise = np.random.normal(0, 1, (r * c, z_size))
  gen_imgs = generator.predict(noise)

  # Rescale images 0 - 1
  gen_imgs = 0.5 * gen_imgs + 0.5
  gen_imgs = gen_imgs.reshape(gen_imgs.shape[0], 28, 28)

  fig, axs = plt.subplots(r, c)
  cnt = 0
  for i in range(r):
    for j in range(c):
      axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
      axs[i, j].axis('off')
      cnt += 1

  fig.savefig("images/%d.png" % epoch)
  plt.close()


train_main()
