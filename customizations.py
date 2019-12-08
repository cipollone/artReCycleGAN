'''\
This module defines utilities to personalize the Keras interface with
custom operations.
'''

import os
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


class CountersSaverCallback(tf.keras.callbacks.Callback):
  '''\
  This callback saves and restores counters: step number and epoch.
  '''

  def __init__(self, log_dir):
    '''\
    Args:
      log_dir: directory of logs
    '''
    tf.keras.callbacks.Callback.__init__(self)
    self._log_dir = log_dir
    self._filename = os.path.join(log_dir, 'counters.txt')

    if not os.path.exists(self._filename):
      self.step = 0
      self.epoch = 0
    else:
      with open(self._filename) as f:
        lines = f.readlines()
      epoch_s, step_s = lines[1].split(',', 2)
      self.step = int(step_s)
      self.epoch = int(epoch_s)


  def on_train_batch_end(self, batch, logs=None):
    self.step += 1
 

  def on_epoch_begin(self, epoch, logs=None):
    self.epoch = epoch
    self._update_log()

  def _update_log(self):
    with open(self._filename, 'w') as f:
      f.write('epoch, step\n')
      f.write('{}, {}\n'.format(self.epoch, self.step))


class TensorBoardWithStep(tf.keras.callbacks.TensorBoard):
  '''\
  Same as tf.keras.callbacks.TensorBoard, with the possibility of setting the
  initial step.
  Warining: this object customizes an internal behaviour. Working on version 
  v2.1.0rc0. Just don't use this object, if it doesn't work with future
  versions.
  '''

  def __init__(self, initial_step=0, *args, **kwargs):
    tf.keras.callbacks.TensorBoard.__init__(self, *args, **kwargs)
    self._initial_step = initial_step


  def _init_batch_steps(self):
    '''\
    Copied from tf.keras.callbacks.TensorBoard with few modifications
    (initial step other than 0).
    '''

    if ops.executing_eagerly_outside_functions():
      # Variables are needed for the `step` value of custom tf.summaries
      # to be updated inside a tf.function.
      self._total_batches_seen = {
          self._train_run_name: variables.Variable(
            self._initial_step, dtype='int64'),
          self._validation_run_name: variables.Variable(0, dtype='int64')
      }
    else:
      # Custom tf.summaries are not supported in legacy graph mode.
      self._total_batches_seen = {
          self._train_run_name: self._initial_step,
          self._validation_run_name: 0
      }

