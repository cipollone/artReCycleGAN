'''\
Networks definitions.
Implementations for the CycleGAN model.
Models are implemented as composite layers, too.

Differences from the paper:
  - Weights initialization
  - Their implementation, not the paper, contains an additional convolution
    layer before the last.
'''
# note: don't assign new properties to model: autograph interprets these as
# layers. Be careful with name clashes with superclasses, such as _layers.


import tensorflow as tf
from tensorflow.keras import layers

from layers import *


def define_model(input_shape):
  '''\
  This method is used to select the model to use, so not to modify the main
  training file.

  Returns:
    keras model, and a dictionary of default options for compile().
  '''

  # Define
  model_layer = Debugging()

  # IO behaviour
  inputs = tf.keras.Input(shape=input_shape)
  outputs = model_layer(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')

  return keras_model, model_layer.compile_defaults


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
        'loss': tf.losses.BinaryCrossentropy(from_logits=True),
        'metrics': [BinaryAccuracyFromLogits()],
      }


  def build(self, inputs_shape):
    ''' Defines the net '''

    # Def
    self.layers_stack = [

        ImagePreprocessing(out_size=(256,256)),
        Discriminator(),
        ReduceMean(),
      ]

    # Built
    BaseLayer.build(self, inputs_shape)


  def call(self, inputs):

    inputs = BaseLayer.call(self, inputs)
    inputs = tf.identity(inputs, name='logits')

    return inputs


class Discriminator(BaseLayer):
  '''\
  Image discriminator in CycleGAN (PatchGAN).
  Each block is a 2d convolution, instance normalization, and LeakyReLU
  activation. The number of filter double each time, while the image size is
  halved. The final output is a (sigmoid) map of classifications for
  {true, false}. With a 256x256 image input, each pixel of the 16x16 output map
  has 70x70 receptive field.

  From a batch of input images, returns scalar logits for binary
  classification.
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
        lambda x: tf.identity(x, name='logits'),
      ]

    # Built
    BaseLayer.build(self, inputs_shape)

