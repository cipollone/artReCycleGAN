'''\
Network definition.
'''

import tensorflow as tf
from tensorflow import keras


# TODO: fake model
def model():

  inputs = keras.Input(shape=(256,256,3), name='image')
  out = keras.layers.Dense(10)(inputs)
  return keras.Model(inputs=inputs, outputs=out, name='placeholder')
  
