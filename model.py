'''\
Networks definitions as keras models.
'''

import tensorflow as tf
from tensorflow.keras import layers


class ImagePreprocessing(layers.Layer):
  '''\
  Initial preprocessing for a batch of images.
  Random crop, flip, instance normalization.
  Expects as input, when called, a batch of images.
  '''

  def __init__(self, out_size):
    '''\
    Create new preprocessing layer.
    Input (height, width) should be greater than (out_height, out_width).

    Args:
      out_size: (out_height, out_width) Image size after preprocessing.
    '''
    layers.Layer.__init__(self)
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


def classifier(input_shape, num_classes):
  '''\
  Defines a keras model for image classification.
  Only used for development: classification is not the final project.

  Args:
    input_shape: Input image shape (height, width, channels)
    num_classes: number of labels to classify
  Returns:
    The keras model
  '''

  # Using the simplest API (net copied for tf guide
  model = tf.keras.Sequential([
      ImagePreprocessing(out_size=(256,256,3)),
      tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu',
        input_shape=input_shape),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dense(num_classes, activation='softmax')
  ])

  return model
