'''\
Custom keras layers that are used in the model.
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


def _make_layer(name, function):
  '''\
  Creates a layer that calls the given tensorflow ops.
  This can be used to create wrappers for tensor operations to include in
  a keras model. This is not suitable for other keras layers.

  Args:
    name: the name of this layer
    function: the function to call. The first argument must be for inputs, 
        other argument must be keyword options.
  '''

  class GenericLayer(layers.Layer):
    ''' Layer class wrapper '''

    count = 0

    def __init__(self, **kargs):

      # Choose a name
      layer_name = kargs.get('name', None)
      if not layer_name:

        layer_name = name
        if GenericLayer.count:
          layer_name = layer_name + '_' + str(GenericLayer.count)
        GenericLayer.count += 1
        kargs['name'] = layer_name

      # Set
      self._function = function
      layers.Layer.__init__(self, **kargs)


    def call(self, inputs, *, training, **kargs):
      return self._function(inputs, **kargs)


  # Rename and return
  GenericLayer.__name__ = name
  return GenericLayer


def layerize(name, scope):
  '''\
  Function decorator.
  Defines a layer with the ops of the decorated function.
  The original function is still available, so it can be used when a 
  layer is not needed.
  NOTE: Assuming the that the first argument of the decorated function is
  for inputs, and other keywords are options. Only suitable for tf ops,
  without variables.

  Args:
    name: destination class name
    scope: dict where to define the class (eg. globals()).
  Returns:
    The same decorated function
  '''

  # Decorator
  def decorator(keras_f):

    # Create a layer
    NewLayer = _make_layer(name=name, function=keras_f)

    # Export layer
    scope[name] = NewLayer

    # Return same function
    return keras_f

  # Decorate
  return decorator


@layerize('InstanceNormalization', globals())
def instance_normalization(inputs):
  '''\
  Instance normalization substitutes Batch normalization in CycleGan paper.
  From their original implementation, I see that the affine transformation
  is not included.

  Args:
    inputs: a batch of 3d tesors (4d input)
  Returns:
    A batch of 3d tensors
  '''

  eps = 1e-4

  # Normalize in image width height dimensions
  mean, var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
  inputs = (inputs - mean) / tf.sqrt(var + eps)

  return inputs


@layerize('ImagePreprocessing', globals())
def image_preprocessing(inputs, out_size):
  '''\
  Initial preprocessing for a batch of images.
  Random crop, flip, normalization.
  Expects as input, when called, a batch of images (4D tensor).
  NOTE: I couldn't use Tf ops for images, due to a probable Tf bug.

  Args:
    inputs: a batch of images (4d input)
    out_shape: (out_height, out_width) Image size after preprocessing.
        out_shape must be smaller than input shape.
  Returns:
    A batch of images
  '''

  # Shapes
  inputs_shape = tf.shape(inputs)
  batch_size = inputs_shape[0]
  in_size = inputs_shape[1:3]
  images = inputs

  # Random crop
  with tf.name_scope('RandomCrop') as scope:
    crop_shift = in_size - out_size
    shift0 = tf.random.uniform([], minval=0, maxval=crop_shift[0],
        dtype=tf.dtypes.int32, name='RandomU1')
    shift1 = tf.random.uniform([], minval=0, maxval=crop_shift[1],
        dtype=tf.dtypes.int32, name='RandomU2')
    images = images[:, shift0:(shift0+out_size[0]),
        shift1:(shift1+out_size[1]), :]  # same crop for all in batch
    images = tf.ensure_shape(images, [None, out_size[0], out_size[1], None])

  # Random flip
  with tf.name_scope('RandomFlip') as scope:
    flips = tf.random.uniform([batch_size], minval=0, maxval=2,
        dtype=tf.dtypes.int32)
    flips = tf.cast(tf.reshape(flips, (batch_size, 1, 1, 1)), # image selection
        dtype=tf.dtypes.float32)
    flipped_images = tf.reverse(images, axis=[2])             # flip all
    images = flipped_images * flips + images * (1-flips)      # select

  # Normalization
  with tf.name_scope('Normalization') as scope:
    images = images/127.5 - 1

  return images


@layerize('ReduceMean', globals())
def reduce_mean(inputs):
  '''\
  Simply collapse all inputs to the average value.
  '''

  return tf.math.reduce_mean(inputs, axis=(1,2,3))


class BinaryAccuracyFromLogits(tf.keras.metrics.Metric):
  '''\
  Just line BinaryAccuracy, but when the output of the model is in logits.
  '''

  def __init__(self, **kwargs):

    # Name
    layer_name = kwargs.get('name', None)
    if not layer_name:
      kwargs['name'] = 'BinaryAccuracy'

    # Internal metric and super
    self._inner_metric = tf.keras.metrics.BinaryAccuracy()
    tf.keras.metrics.Metric.__init__(self, **kwargs)


  def update_state(self, y_true, y_pred, sample_weight=None):

    # Transform
    logits = y_pred
    probabilities = tf.keras.activations.sigmoid(logits)

    # Forward
    self._inner_metric.update_state(y_true, probabilities, sample_weight)


  def result(self):
    return self._inner_metric.result()

