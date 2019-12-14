#!/usr/bin/env python3

'''\
Main script file. Implementation of CycleGAN paper.
Tested with Tensorflow 2.1rc0.
'''

import os
import argparse
import shutil
import json
import tensorflow as tf

import data
import models
from customizations import *


def _prepare_directories(model_name, resume=False):
  '''\
  Prepares empty directories where logs and models are saved.
  'log directory' refers to the path of the current log.
  'logs 'directory' refers to the path of all logs (parent dir).

  Args:
    model_name: identifier for this training job
    resume: if true, directories are not erased.
  Returns:
    model directory, log directory, logs directory
  '''

  # Base paths
  model_path = os.path.join('models', model_name)
  logs_path = os.path.join('logs', model_name)
  dirs = (model_path, logs_path)

  # Erase?
  if not resume:

    # Existing?
    existing = []
    for d in dirs:
      if os.path.exists(d):
        existing.append(d)

    # Confirm
    if existing:
      print('Erasing', *existing, '. Continue (Y/n)? ', end='')
      c = input()
      if not c in ('y', 'Y', ''): quit()

    # New
    for d in existing:
      shutil.rmtree(d)
    for d in dirs:
      os.makedirs(d)

  # Logs alwas use new directories (using increasing numbers)
  i = 0
  while os.path.exists(os.path.join(logs_path, str(i))): i += 1
  log_path = os.path.join(logs_path, str(i))
  os.mkdir(log_path)

  return (model_path, log_path, logs_path)


def train(args):
  '''\
  Training function.

  Args:
    args: namespace of arguments. Run 'artRecycle train --help' for info.
  '''

  # Model name and paths
  model_name = '{}|{}'.format(*args.datasets)
  model_path, log_path, logs_path = _prepare_directories(
      model_name, resume=args.cont)

  model_json = os.path.join(model_path, 'keras.json')
  model_checkpoint = os.path.join(model_path, 'model')

  # Define datasets
  input_shape = (300, 300, 3)
  train_dataset, train_size = data.load('classes', 'train',
      shape=input_shape, batch=args.batch)
  test_dataset, test_size = data.load('classes', 'test',
      shape=input_shape, batch=args.batch)

  # If new training (not resuming)
  if not args.cont:

    # Define keras model
    inputs = tf.keras.Input(shape=input_shape)
    outputs = models.Debugging()(inputs)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Save keras graph
    graph_json = keras_model.to_json()
    graph_json = json.dumps(json.loads(graph_json), indent=2)
    with open(model_json, 'w') as f:
      f.write(graph_json)

    # Create tensorflow graph
    keras_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate),
        loss = tf.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()],
      )

  # If resuming
  else:

    # Reload keras model
    with open(model_json) as f:
      graph_json = f.read()
    keras_model = tf.keras.models.model_from_json(graph_json)

    # TODO: recompile + weights?

  # Training settings
  steps_per_epoch = int(train_size/args.batch) \
      if not args.epoch_steps else args.epoch_steps

  # Step saver
  counter_saver_callback = CountersSaverCallback(logs_path)
  step = counter_saver_callback.step
  epoch = counter_saver_callback.epoch

  # Callbacks
  callbacks = [
      tf.keras.callbacks.ModelCheckpoint(
        filepath = model_checkpoint, monitor = 'loss', save_best_only = True,
        mode = 'max', save_freq = 'epoch', save_weights_only = True
      ),
      TensorBoardWithStep(
        initial_step = step,
        log_dir = log_path,
        write_graph = not args.cont,
        update_freq = args.logs
      ),
      tf.keras.callbacks.TerminateOnNaN(),
      counter_saver_callback
    ]

  # Train
  keras_model.fit(train_dataset,
      epochs = args.epochs,
      steps_per_epoch = steps_per_epoch,
      callbacks = callbacks,
      initial_epoch = epoch
    )

  # Evaluate
  keras_model.evaluate(test_dataset)


def use(args):
  '''\
  Transform images with the net.

  Args:
    args: namespace of arguments. Run 'artRecycle use --help' for info.
  '''

  raise NotImplementedError()


def debug(args):
  '''\
  Debugging function.

  Args:
    args: namespace of arguments. Run --help for info.
  '''
  import matplotlib.pyplot as plt
  import layers

  print('> Debug')

  # Testing the new classification dataset
  dataset, size = data.load('classes', 'test', batch=2)

  # Testing preprocessing layer
  preprocessing = layers.ImagePreprocessing(out_size=(256,256))

  # Show
  print(dataset, size)
  for images, labels in dataset:
    processed_images = preprocessing(images)
    for img, processed_img, label in zip(images, processed_images, labels):
      print('min', tf.math.reduce_min(processed_img), 'max',
          tf.math.reduce_max(processed_img))
      print('label: ',label, flush=True)
      plt.imshow(tf.cast(img, dtype=tf.uint8))
      plt.show()
      input()


def main():
  '''\
  Main function. Called when this file is executed as script.
  '''

  # Default settings
  batch_size = 30
  learning_rate = 0.001
  epochs = 1
  log_frequency = 1

  # Datasets
  datasets_names = {name for name in data.datasets}

  # Parsing arguments
  parser = argparse.ArgumentParser(description=
      'ArtReCycle: my implementation of CycleGAN with Tensorflow 2.0')
  op_parsers = parser.add_subparsers(help='Operation', dest='op')

  # Train op
  train_parser = op_parsers.add_parser('train', help='Train the net')
  train_parser.add_argument('-d', '--datasets', nargs=2, required=True,
      type=str, choices=datasets_names, metavar=('datasetA', 'datasetB'),
      help='Input dataset. Choose from: '+str(datasets_names))
  train_parser.add_argument('-b', '--batch', type=int, default=batch_size,
      help='Number of elements in each batch')
  train_parser.add_argument('-r', '--rate', type=float, default=learning_rate,
      help='Learning rate')
  train_parser.add_argument('-e', '--epochs', type=int, default=epochs,
      help='Number of epochs to train')
  train_parser.add_argument('-s', '--epoch_steps', type=int, default=None,
      help='Force a specific number of steps in each epoch')
  train_parser.add_argument('-l', '--logs', type=int, default=log_frequency,
      help='Save logs after this number of batches')
  train_parser.add_argument('-c', '--continue', action='store_true', dest='cont',
      help='Loads most recent saved model and resumes training.')

  # Use op
  use_parser = op_parsers.add_parser('use',
      help='Apply the transformation to images')

  # Debug op
  debug_parser = op_parsers.add_parser('debug')
  debug_parser.add_argument('args', nargs='*')

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    train(args)
  elif args.op == 'use':
    use(args)
  elif args.op == 'debug':
    debug(args)


if __name__ == '__main__':
  main()
