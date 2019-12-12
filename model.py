'''\
Networks definitions as keras models.
Implementations for the CycleGAN model.
I'm not using the Keras functional API, but the Model subclassing, because
ops have the wrong scope in TensorBoard, otherwise.

Differences from the paper:
  - Weights initialization
  - Their implementation, not the paper, contains an additional convolution
    layer before the last.
  - In their implementation there's no affine transformation, but this seems
    strange.
  - Using a fixed normalization for images in ImagePreprocessing.
'''
# TODO: check serialization for custom models

import tensorflow as tf
from tensorflow.keras import layers

from layers import *


def debugging_model(input_shape):
  '''\
  Returns a model only used for debugging other models.
  '''

  class ReduceLayer(layers.Layer):
    def call(self, inputs):
      return tf.math.reduce_mean(inputs, axis=(1,2,3))
  
  # Testing discriminator as a model (as a classifier).
  # Adding input, preprocessing, and output

  layers_stack = [
      ImagePreprocessing((256,256), input_shape=input_shape),
      DiscriminatorModel(),
      ReduceLayer(),
    ]

  keras_model = tf.keras.Sequential(layers_stack)
  keras_model.summary()

  return keras_model


class DiscriminatorModel(tf.keras.Model):
  '''\
  Image discriminator in CycleGAN (PatchGAN).
  Each block is a 2d convolution, instance normalization, and LeakyReLU
  activation. The number of filter double each time, while the image size is
  halved. The final output is a (sigmoid) map of classifications for
  {true, false}. With a 256x256 image input, each pixel of the 16x16 output map
  has 70x70 receptive field.
  '''
  
  def __init__(self):

    tf.keras.Model.__init__(self, name='Discriminator')

    # Parameters
    filters = 64
    Activation = lambda: layers.LeakyReLU(0.2)

    layers_stack = []   # Layer stack

    # Input block
    layers_stack += [
        layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same',
          input_shape=(256,256,3)),   # Forcing input image shape
        Activation(),
      ]

    # Other blocks
    for i in range(3):

      filters *= 2
      layers_stack += [
        layers.Conv2D(filters=filters, kernel_size=4, strides=2, padding='same'),
        InstanceNormalization(),
        Activation(),
        ]

    # Output block
    layers_stack += [
      layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same'),
      ] # Sigmoid activation not applied

    # Store
    self._layers_stack = layers_stack


  def call(self, inputs):

    out = inputs
    for layer in self._layers_stack:
      out = layer(out)

    return out


  def get_config(self):
    return dict()

