'''\
Networks definitions.
Implementations for the CycleGAN model.
Models are implemented as composite layers, too.

Differences from the paper:
  - Weights initialization
  - Their implementation, not the paper, contains an additional convolution
    layer before the last.
  - In their implementation there's no affine transformation, but this seems
    strange.
  - Using a fixed normalization for images in ImagePreprocessing.
'''

import tensorflow as tf
from tensorflow.keras import layers

from layers import *


class Debugging(layers.Layer):
  '''\
  This model is only used during development.

  Testing discriminator as a model (as a classifier).
  Adding input, preprocessing, and output.

  From a batch of input images, computes the vector of probabilities for the
  binary classification task.
  '''

  def build(self, inputs_shape):
    ''' Defines the net '''

    self._layers_stack = [
        lambda x: ImagePreprocessing()(x, out_size=(256,256)),
        Discriminator(),
        ReduceMean(),
      ]


  def call(self, inputs):

    # Sequential
    for layer in self._layers_stack:
      inputs = layer(inputs)
    return inputs


class Discriminator(layers.Layer):
  '''\
  Image discriminator in CycleGAN (PatchGAN).
  Each block is a 2d convolution, instance normalization, and LeakyReLU
  activation. The number of filter double each time, while the image size is
  halved. The final output is a (sigmoid) map of classifications for
  {true, false}. With a 256x256 image input, each pixel of the 16x16 output map
  has 70x70 receptive field.

  From a batch of input images, computes a vector of probabilities for the
  binary classification task.
  '''

  def build(self, inputs_shape):
    ''' Defines the net. '''

    # Parameters
    filters = 64
    Activation = lambda: layers.LeakyReLU(0.2)

    layers_stack = []

    # Input block
    layers_stack += [
        layers.Conv2D(filters=filters, kernel_size=4, strides=2,
          padding='same'),
        Activation(),
      ]

    # Other blocks
    for i in range(3):

      filters *= 2
      layers_stack += [
          layers.Conv2D(filters=filters, kernel_size=4, strides=2,
            padding='same'),
          InstanceNormalization(),
          Activation(),
        ]

    # Output block
    layers_stack += [
        layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same'),
        tf.keras.activations.sigmoid,
      ] # TODO: remove sigmoid here and add in loss. What about metric?

    # Store
    self._layers_stack = layers_stack

  
  def call(self, inputs):

    # Sequential
    for layer in self._layers_stack:
      inputs = layer(inputs)
    return inputs

