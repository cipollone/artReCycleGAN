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


class BaseLayer(layers.Layer):
  '''\
  Base class for layers of this module. Subclasses can push layers to
  self.layers_stack to create a simple sequential layer, otherwise they must
  override call(). self.compile_defaults is a dictionary of default argument
  when one of these layers is used as a model.
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
    ''' Empty dict is fine if subclasses constructors accept no arguments '''

    config = layers.Layer.get_config(self)
    config.update({})
    return config


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

    self._layers_defs = {}

    self._layers_defs['pre'] = ImagePreprocessing(out_size=(256,256))
    self._layers_defs['net'] = Discriminator()
    self._layers_defs['mean'] = ReduceMean()

    # Built
    BaseLayer.build(self, inputs_shape)


  def call(self, inputs):

    # Model
    inputs = self._layers_defs['pre'](inputs)
    inputs = self._layers_defs['net'](inputs)
    inputs = self._layers_defs['mean'](inputs)

    # Outputs
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

