'''\
This module defines the CycleGAN model and provides the training step.
define_model returns a keras model. However, this objects should not be
compiled or fitted() with keras interface. Instead, use cycleGAN_step.
All functions in this module depend on the type of model returned by
define_model.
'''

import tensorflow as tf

import nets


def _set_model():
  ''' This sets the global model used in this layer. Called at the end. '''

  global Model, Trainer, model_metrics

  Model = nets.CycleGAN
  Trainer = CycleGAN_trainer
  #model_metrics = [None, None, 'gBA_loss',]
  model_metrics = [None, None, 'dA_loss', 'dB_loss', 'gAB_loss', 'gBA_loss',]


def define_model(image_shape):
  '''\
  Creates the model.
  Returns:
    keras model, and model layer
  '''

  # Define
  model_layer = Model()

  # Inputs are two batches of images from both datasets
  input_A = tf.keras.Input(shape=image_shape, name='Input_A')
  input_B = tf.keras.Input(shape=image_shape, name='Input_B')
  inputs = (input_A, input_B)

  # Model from IO behaviour
  outputs = model_layer(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs,
      name=model_layer.__class__.__name__)

  return keras_model, model_layer


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
  names = model_metrics
  metrics = {name: val for name, val in zip(names, outputs) if name}

  return metrics


class Tester:
  '''\
  Tests the model.
  Args:
    model: keras model to evaluate
    iterator: dataset iterator of the test set
  '''

  def __init__(self, model, iterator):

    # Store
    self.model = model
    self.iterator = iterator

    # Initialize metrics
    metrics_names = get_model_metrics(None)
    self.metrics_mean = {name: tf.metrics.Mean(name) for name in metrics_names}


  def result(self):
    '''\
    Returns the result of last evaluation steps and clears the accumulators.
    '''

    # Collect
    metrics_val = {name: self.metrics_mean[name].result().numpy() \
        for name in self.metrics_mean}

    # Reset
    for name in self.metrics_mean:
      self.metrics_mean[name].reset_states()

    return metrics_val


  @tf.function
  def step(self):
    ''' One evaluation step '''

    # Compute
    outputs = self.model(next(self.iterator))
    metrics = get_model_metrics(outputs)

    # Accumulate
    for name in self.metrics_mean:
      self.metrics_mean[name].update_state(metrics[name])


class CycleGAN_trainer:
  '''\
  Trains the CycleGAN model.
  Args:
    cgan_model: CycleGAN keras model to train
    optimizer: a callable that creates an optimizer
    iterator: dataset iterator returning a pair of batches of images.
  '''

  def __init__(self, cgan_model, optimizer, iterator):

    # Store
    self.cgan = cgan_model
    cgan_layer = cgan_model.get_layer('CycleGAN')
    self.iterator = iterator

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


  @tf.function
  def step(self):
    ''' One training step for CycleGAN '''

    # Record operations in forward step
    with tf.GradientTape(persistent=True) as tape:
      outputs = self.cgan(next(self.iterator))
      
    # Parse losses
    losses = get_model_metrics(outputs)

    # Compute gradients
    gradient_dA = tape.gradient(losses['dA_loss'], self.params['dA'])
    gradient_dB = tape.gradient(losses['dB_loss'], self.params['dB'])
    gradient_gAB = tape.gradient(losses['gAB_loss'], self.params['gAB'])
    gradient_gBA = tape.gradient(losses['gBA_loss'], self.params['gBA'])

    # Step
    self.optimizers['dA'].apply_gradients(zip(gradient_dA, self.params['dA']))
    self.optimizers['dB'].apply_gradients(zip(gradient_dB, self.params['dB']))
    self.optimizers['gAB'].apply_gradients(zip(gradient_gAB, self.params['gAB']))
    self.optimizers['gBA'].apply_gradients(zip(gradient_gBA, self.params['gBA']))

    return outputs


# Set
_set_model()
