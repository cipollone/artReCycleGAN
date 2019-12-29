'''\
This module defines the CycleGAN model and provides the training step.
define_model returns a keras model. However, this objects should not be
compiled or fitted() with keras interface. Instead, use cycleGAN_step.
'''

import tensorflow as tf

import nets


def define_model(image_shape):
  '''\
  Creates a CycleGAN model.
  Args:
    image_shape: 3D tensor. Shape of each input image.
  Returns:
    keras model
  '''

  # Define
  model_layer = nets.CycleGAN()

  # Inputs are two batches of images from both datasets
  input_A = tf.keras.Input(shape=image_shape, name='Input_A')
  input_B = tf.keras.Input(shape=image_shape, name='Input_B')
  inputs = (input_A, input_B)

  # Model from IO behaviour
  outputs = model_layer(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs,
      name=model_layer.__class__.__name__)

  return keras_model


def cycleGAN_step(cgan):
  '''\
  One training step for the CycleGAN model.
  Args:
    cgan: cycle gan model
  '''
  pass
