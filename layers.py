'''\
Custom keras layers that are used in the model.
'''

import tensorflow as tf
from tensorflow.keras import layers


class InstanceNormalization():
  pass
  # TODO: What is it, and differences with respect to Layer normalization.


class ImagePreprocessing(layers.Layer):
  '''\
  Initial preprocessing for a batch of images.
  Random crop, flip, instance normalization.
  Expects as input, when called, a batch of images.
  '''

  def __init__(self, out_size, **kwargs):
    '''\
    Create new preprocessing layer.
    Input (height, width) should be greater than (out_height, out_width).

    Args:
      out_size: (out_height, out_width) Image size after preprocessing.
    '''
    layers.Layer.__init__(self, **kwargs)
    self._out_size = out_size

  
  def build(self, input_shape):
    layers.Layer.build(self, input_shape)
    self._channels = input_shape[-1]

 
  def call(self, inputs):

    # Some changes
    batch_shape = tf.shape(inputs)
    images = tf.image.random_crop(inputs,    # Same for all should be fine
        [batch_shape[0], self._out_size[0], self._out_size[1], self._channels])
    images = tf.image.random_flip_left_right(images)

    # Normalization
    images = tf.image.per_image_standardization(images)

    return images


  def get_config(self):
    return {'out_size': self._out_size}
