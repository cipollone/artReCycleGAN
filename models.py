'''\
Networks definitions as keras models.
Implementations for the CycleGAN model.
I'm not using the Keras functional API, but the Model subclassing, because
ops have the wrong scope in TensorBoard, otherwise.
Models defined in this module ends with 'Model'.

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


class DebuggingModel(tf.keras.Model):
  '''\
  This model is only used during development.
  '''

  class ReduceLayer(layers.Layer):

    def __init__(self):
      layers.Layer.__init__(self, name='Reduce')

    def call(self, inputs):
      return tf.math.reduce_mean(inputs, axis=(1,2,3))
  

  def __init__(self, input_shape):
    ''' Defines the net.  '''

    tf.keras.Model.__init__(self, name='Debugging')

    self._input_shape = input_shape

    # Testing discriminator as a model (as a classifier).
    # Adding input, preprocessing, and output
    self._layers_stack = [
        ImagePreprocessing((256,256), input_shape=input_shape),
        DiscriminatorModel(),
        self.ReduceLayer(),
      ]


  def call(self, inputs):
    ''' Forward pass '''

    # Sequential
    out = inputs
    for layer in self._layers_stack:
      out = layer(out)

    return out


  def compile_with_defaults(self, **kargs):
    '''\
    Compiles this keras model, with some suggested defaults.
    
    Args: Given args are forwarded to keras compile. Only keywords arguments
      are supported.
    '''

    # Appropriate loss/metrics for this model
    kargs.setdefault('loss', tf.losses.BinaryCrossentropy())
    kargs.setdefault('metrics', [tf.keras.metrics.BinaryAccuracy()])

    # Compile
    tf.keras.Model.compile(self, **kargs)


  # TODO: not a correct configuration for a model
  def get_config(self):
    return { 'input_shape': self._input_shape }


class DiscriminatorModel(tf.keras.Model):
  '''\
  Image discriminator in CycleGAN (PatchGAN).
  Each block is a 2d convolution, instance normalization, and LeakyReLU
  activation. The number of filter double each time, while the image size is
  halved. The final output is a (sigmoid) map of classifications for
  {true, false}. With a 256x256 image input, each pixel of the 16x16 output map
  has 70x70 receptive field.
  '''
  
  def __init__(self, input_shape=(256,256,3)):

    tf.keras.Model.__init__(self, name='Discriminator')

    # Parameters
    filters = 64
    Activation = lambda: layers.LeakyReLU(0.2)
    self._input_shape = input_shape

    layers_stack = []   # Layer stack

    # Input block
    layers_stack += [
        layers.Conv2D(filters=filters, kernel_size=4, strides=2,
          padding='same', input_shape=input_shape),
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
    out = inputs
    for layer in self._layers_stack:
      out = layer(out)

    return out


  def get_config(self):
    return { 'input_shape': self._input_shape }

