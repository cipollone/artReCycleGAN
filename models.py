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
  names = [None, None, 'dA_loss', 'dB_loss', 'gA_loss', 'gB_loss',]
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


# This should be a class with step function.
def cycleGAN_step(cgan):
  '''\
  One training step for the CycleGAN model.
  Args:
    cgan: cycle gan model
  '''
  pass
