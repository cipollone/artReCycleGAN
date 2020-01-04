'''\
This module defines the CycleGAN model and provides the training step.
define_model returns a keras model. However, this objects should not be
compiled or fitted() with keras interface. Instead, use cycleGAN_step.
All functions in this module depend on the type of model returned by
define_model.
'''

import tensorflow as tf

import nets


def define_model(image_shape):
  '''\
  Creates a CycleGAN model.
  Args:
    image_shape: 3D Shape of each input image.
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


def get_model_metrics(outputs):
  '''\
  Parses the output of the model and returns the associated metrics.
  Args:
    outputs: vector of output of the model. None is allowed to get just the
      output names.
  Returns:
    dict that maps metric name to value
  '''

  if not outputs:
    outputs = list((None for i in range(10)))

  # Parse metrics
  names = [None, None, 'dA_loss', 'dB_loss', 'gAB_loss', 'gBA_loss',]
  metrics = {name: val for name, val in zip(names, outputs) if name}

  return metrics


@tf.function
def compute_model_metrics(keras_model, test_dataset, metrics_mean):
  '''\
  Model valuation on the test set. 
  Averages all metrics on the test set and return them.
  Accumulate metrics for each output.
  Args:
    keras_model: model to valuate
    test_dataset: dataset of inputs for validation
    metrics_mean: dict of tensorflow metrics to update (usually Mean).
      Computed metrics are updated here.
  Returns:
    dict. name: metric value
  '''

  # Evaluate on test set
  for test_batch in test_dataset:

    # Compute
    outputs = keras_model(test_batch)
    metrics = get_model_metrics(outputs)

    # Accumulate
    for name in metrics_mean:
      metrics_mean[name].update_state(metrics[name])


class CycleGAN_trainer:
  '''\
  Trains the CycleGAN model.
  Args:
    cgan_model: CycleGAN keras model to train
    optimizer: a callable that creates an optimizer
  '''

  def __init__(self, cgan_model, optimizer):

    # Store
    self.cgan = cgan_model
    cgan_layer = cgan_model.get_layer('CycleGAN')

    # Also save the parameters
    self.params = {}
    self.params['dA'] = cgan_layer.discriminator_A.trainable_variables
    self.params['dB'] = cgan_layer.discriminator_B.trainable_variables
    self.params['gAB'] = cgan_layer.generator_AB.trainable_variables
    self.params['gBA'] = cgan_layer.generator_BA.trainable_variables

    # Create optimizers
    self.optimizers = {}
    self.optimizers['dA'] = optimizer()
    self.optimizers['dB'] = optimizer()
    self.optimizers['gAB'] = optimizer()
    self.optimizers['gBA'] = optimizer()


  def step(self, input_batch):
    '''\
    One training step for CycleGAN.
    Args:
      input_batch: training batch (pair of batches of images, in this case)
    Returns:
      outputs of the model
    '''
    return _cycleGAN_trainer_step(self.cgan, self.params, self.optimizers,
        input_batch)


@tf.function
def _cycleGAN_trainer_step(cgan, params, optimizers, input_batch):
  ''' This can't be in CycleGAN_trainer due to @tf.function '''

  # Record operations in forward step
  with tf.GradientTape(persistent=True) as tape:
    outputs = cgan(input_batch)
    
  # Parse losses
  losses = get_model_metrics(outputs)

  # Compute gradients
  gradient_dA = tape.gradient(losses['dA_loss'], params['dA'])
  gradient_dB = tape.gradient(losses['dB_loss'], params['dB'])
  gradient_gAB = tape.gradient(losses['gAB_loss'], params['gAB'])
  gradient_gBA = tape.gradient(losses['gBA_loss'], params['gBA'])

  # Step
  optimizers['dA'].apply_gradients(zip(gradient_dA, params['dA']))
  optimizers['dB'].apply_gradients(zip(gradient_dB, params['dB']))
  optimizers['gAB'].apply_gradients(zip(gradient_gAB, params['gAB']))
  optimizers['gBA'].apply_gradients(zip(gradient_gBA, params['gBA']))

  return outputs

