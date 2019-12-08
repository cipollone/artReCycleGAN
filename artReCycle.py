#!/usr/bin/env python3

'''\
Main script file. Implementation of CycleGAN paper with Tensorflow 2.0.
'''

import os
import argparse
import shutil
import json
import tensorflow as tf

import data
import model
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

  # Define datasets
  input_shape = (256, 256, 3)
  train_dataset, train_size = data.load('classes', 'train',
      shape=input_shape, batch=args.batch)
  test_dataset, test_size = data.load('classes', 'test',
      shape=input_shape, batch=args.batch)

  # If new training (not resuming)
  if not args.cont:

    # Define keras model (classification task for now)
    keras_model = model.classifier(
        input_shape = input_shape,
        num_classes = len(data.datasets)
      )

    # Save keras graph
    graph_json = keras_model.to_json()
    graph_json = json.dumps(json.loads(graph_json), indent=2)
    with open(os.path.join(model_path,'keras.json'), 'w') as f:
      f.write(graph_json)

    # Create tensorflow graph
    keras_model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate),
        loss = tf.losses.CategoricalCrossentropy(),
        metrics = [tf.keras.metrics.CategoricalAccuracy()]
      )

  # If resuming
  else:
    # Load model
    keras_model = tf.keras.models.load_model(model_path)

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
        filepath = model_path, monitor = 'loss', save_best_only = True,
        mode = 'max', save_freq = 'epoch', save_weights_only = False
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
  import code

  print('> Debug')

  # Testing the input pipeline
  dataset, _ = data.load('caravaggio', 'test', batch=4)
  preprocessing = model.ImagePreprocessing(out_size=(256,256,3))

  print(dataset)
  for batch in dataset:
    images = preprocessing(batch)
    print('New batch')
    for img in images:
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
  train_parser.add_argument('--epoch_steps', type=int, default=None,
      help='Number of steps in each epoch')
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
