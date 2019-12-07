'''\
Networks definitions as keras models.
'''

import tensorflow as tf


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
