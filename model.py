'''\
Networks definitions as keras models.
Implementations for the CycleGAN model.

Differences from the paper:
  - Weights initialization
  - Their implementation, not the paper, contains an additional convolution
    layer before the last.
'''

import tensorflow as tf
from tensorflow.keras import layers

from layers import *


def generator():
  '''\
  Image to image transformation in CycleGAN.
  '''
  pass


def discriminator():
  '''\
  Image discriminator in CycleGAN (PatchGAN).
  Each block is a 2d convolution, instance normalization, and LeakyReLU
  activation. The number of filter double each time, while the image size is
  halved. The final output is a (sigmoid) map of classifications for
  {true, false}. With a 256x256 image input, each pixel of the 16x16 output map
  has 70x70 receptive field.

  Returns:
    a keras model.
  '''

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
      Activation(),
      ]

  # Output block
  layers_stack += [
    layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same'),
    # Sigmoid activation not applied
    ]

  # Model
  model = tf.keras.Sequential(layers_stack)
  return model

  # TODO: add instance normalization with trainable variables


def classifier(input_shape, num_classes):
  '''\
  Defines a keras model for image classification.
  Only used for development: classification is not the final project.

  Args:
    input_shape: Input image shape (height, width, channels)
    num_classes: number of labels to classify
  Returns:
    The keras model
  '''

  # Using the simplest API (net copied for tf guide
  model = tf.keras.Sequential([
      ImagePreprocessing(out_size=(256,256,3)),
      layers.Conv2D(16, 3, padding='same', activation='relu',
        input_shape=input_shape),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(512, activation='relu'),
      layers.Dense(num_classes, activation='softmax')
  ])

  return model
