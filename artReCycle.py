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


def train(args):
  '''\
  Training function.

  Args:
    args: namespace of arguments. Run 'artRecycle train --help' for info.
  '''

  # Prepare folders
  dirs = ('logs', 'models')
  for d in dirs:
    if os.path.exists(d):
      print('Previous models will be erased. Continue (Y/n)? ', end='')
      c = input()
      if c in ('y', 'Y', ''): break
      else: return
  for d in dirs:
    if os.path.exists(d):
      shutil.rmtree(d)
    os.mkdir(d)

  # Define datasets
  input_shape = (256, 256, 3)
  train_dataset, train_size = data.load('classes', 'train',
      shape=input_shape, batch=args.batch)
  test_dataset, test_size = data.load('classes', 'test',
      shape=input_shape, batch=args.batch)

  # Define keras model (classification task for now)
  keras_model = model.classifier(
      input_shape = input_shape,
      num_classes = len(data.datasets))
  model_name = 'classifier'

  # Save keras graph
  model_path = os.path.join('models', model_name)
  graph_json = keras_model.to_json()
  graph_json = json.dumps(json.loads(graph_json), indent=2)
  with open(model_path+'.json', 'w') as f:
    f.write(graph_json)

  # Create tensorflow graph
  keras_model.compile(
      optimizer = tf.keras.optimizers.Adam(learning_rate=args.rate),
      loss = tf.losses.CategoricalCrossentropy(),
      metrics = [tf.keras.metrics.CategoricalAccuracy()]
    )

  # Training settings
  steps_per_epoch = int(train_size/args.batch) \
      if not args.epoch_steps else args.epoch_steps

  # Train. TODO: callbacks
  keras_model.fit(train_dataset,
      epochs = args.epochs,
      steps_per_epoch = steps_per_epoch
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
  dataset, _ = data.load('classes', 'test', batch=2)

  print(dataset)
  for batch in dataset:
    for img, label in zip(batch[0], batch[1]):
      print('label: ',label)
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

  # Datasets
  datasets_names = {name for name in data.datasets}

  # Parsing arguments
  parser = argparse.ArgumentParser(description=
      'ArtReCycle: my implementation of CycleGAN with Tensorflow 2.0')
  op_parsers = parser.add_subparsers(help='Operation', dest='op')

  # Train op
  train_parser = op_parsers.add_parser('train', help='Train the net')
  train_parser.add_argument('-d', '--dataset', nargs=2, required=True,
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
