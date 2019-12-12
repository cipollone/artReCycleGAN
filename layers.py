'''\
Custom keras layers that are used in the model.
'''

import tensorflow as tf
from tensorflow.keras import layers


class InstanceNormalization(layers.Layer):
  '''\
  Instance normalization substitutes Batch normalization in CycleGan paper.
  From their original implementation, I see that the affine transformation
  is not included.
  '''

  eps = 1e-4

  def call(self, inputs):

    # Normalize in image width height dimensions
    mean, var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
    inputs = (inputs - mean) / tf.sqrt(var + self.eps)

    return inputs


class ImagePreprocessing(layers.Layer):
  '''\
  Initial preprocessing for a batch of images.
  Random crop, flip, normalization.
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
    self._channels = input_shape[-1]
    layers.Layer.build(self, input_shape)

 
  def call(self, inputs):

    # Some changes
    batch_shape = tf.shape(inputs)
    images = tf.image.random_crop(inputs,    # Same for all should be fine
        [batch_shape[0], self._out_size[0], self._out_size[1], self._channels])
    images = tf.image.random_flip_left_right(images)

    # Normalization
    images = images/127.5 - 1

    return images


  def get_config(self):
    return {'out_size': self._out_size}
