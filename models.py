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
  model_layer = Generator()

  # IO behaviour
  inputs = tf.keras.Input(shape=input_shape)
  outputs = model_layer(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs, name='Model')

  return keras_model, model_layer.compile_defaults


class Debugging(BaseLayer):
  '''\
  This model is only used during development.
  '''


  def build(self, input_shape):
    ''' Defines the net '''

    # Def
    self.layers_stack = [

        ImagePreprocessing(out_size=(256,256)),
        ResNetBlock(filters=3),
        ResNetBlock(filters=3),
        ReduceMean(),
      ]

    # Super
    BaseLayer.build(self, input_shape)


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

  def build(self, input_shape):
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
    BaseLayer.build(self, input_shape)


class Generator(BaseLayer):
  '''\
  Image generator in CycleGAN. Transforms images from one domain to another.
  The generator is composed of three parts: encoding, transformation,
  deconding, which are respectively based on convolutional, resnet, and
  convolutional transpose blocks. See paper for a more precise description.
  '''

  class Encoding(BaseLayer):
    ''' Encoding phase of the generator '''

    def build(self, input_shape):

      # Def
      self.layers_stack = [

          GeneralConvBlock( filters=64, kernel_size=7, stride=1, pad=3 ),
          GeneralConvBlock( filters=128, kernel_size=3, stride=2 ),
          GeneralConvBlock( filters=256, kernel_size=3, stride=2 ),
        ]

      # Super
      BaseLayer.build(self, input_shape)


  class Transformation(BaseLayer):
    ''' Transformation phase of the generator '''

    resnet_blocks = 9

    def build(self, input_shape):

      # Def
      for i in range(self.resnet_blocks):
        self.layers_stack.append(
            ResNetBlock( filters=256 )
          )

      # Super
      BaseLayer.build(self, input_shape)


  def build(self, input_shape):
    ''' Defines the net '''

    # Def
    self.layers_stack = [

        Generator.Encoding(),
        Generator.Transformation(),
      ]

    # Super
    BaseLayer.build(self, input_shape)

