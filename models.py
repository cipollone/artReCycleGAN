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


def define_model(input_shape):
  '''\
  This method is used to select the model to use, so not to modify the main
  training file.

  Returns:
    keras model. It has an additional member 'compile_defaults', for suggested
      compile options.
  '''

  # Define
  model_layer = Debugging()

  # IO behaviour
  inputs = tf.keras.Input(shape=input_shape)
  outputs = model_layer(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

  # Global model options
  keras_model.compile_defaults = model_layer.compile_defaults

  return keras_model


class BaseLayer(layers.Layer):
  '''\
  Base class for layers of this module. These layers can also be used as global
  models.  Subclasses can push layers to self.layers_stack to create a simple
  sequential layer, otherwise they must override call(). self.compile_defaults
  is a dictionary of default argument when one of these layers is used as a
  model.
  '''

  def __init__(self, *args, **kargs):
    layers.Layer.__init__(self, *args, **kargs)
    self.layers_stack = []
    self.compile_defaults = {}


  def call(self, inputs):

    # This must be overridden if layers_stack is not used
    if len(self.layers_stack) == 0:
      raise NotImplementedError(
        'call() must be overridden if self.layers_stack is not used.')

    # Sequential model by default
    for layer in self.layers_stack:
      inputs = layer(inputs)
    return inputs


  def get_config(self):
    ''' Empty dict if fine if subclasses constructors accept no arguments '''
    return {}


class Debugging(BaseLayer):
  '''\
  This model is only used during development.

  Testing discriminator as a model (as a classifier).
  Adding input, preprocessing, and output.

  From a batch of input images, computes the vector of probabilities for the
  binary classification task.
  '''

  def __init__(self):
    BaseLayer.__init__(self)

    self.compile_defaults = {
        'loss': tf.losses.BinaryCrossentropy(),
        'metrics': [tf.keras.metrics.BinaryAccuracy()],
      }


  def build(self, inputs_shape):
    ''' Defines the net '''

    self.layers_stack = [
        lambda x: ImagePreprocessing()(x, out_size=(256,256)),
        Discriminator(),
        ReduceMean(),
      ]


class Discriminator(BaseLayer):
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

    # Input block
    self.layers_stack += [
        layers.Conv2D(filters=filters, kernel_size=4, strides=2,
          padding='same'),
        Activation(),
      ]

    # Other blocks
    for i in range(3):

      filters *= 2
      self.layers_stack += [
          layers.Conv2D(filters=filters, kernel_size=4, strides=2,
            padding='same'),
          InstanceNormalization(),
          Activation(),
        ]

    # Output block
    self.layers_stack += [
        layers.Conv2D(filters=1, kernel_size=4, strides=1, padding='same'),
        tf.keras.activations.sigmoid,
      ] # TODO: remove sigmoid here and add in loss. What about metric?

