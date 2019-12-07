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

  # Define keras model
  keras_model = model.model()

  # Create tensorflow graph
  keras_model.compile()

  # Save keras graph
  model_path = os.path.join('models', 'model')
  graph_json = keras_model.to_json()
  graph_json = json.dumps(json.loads(graph_json), indent=2)
  with open(model_path+'.json', 'w') as f:
    f.write(graph_json)

  # Save tensorflow graph
  # TODO: can i use keras.Model.fit




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
  dataset = data.load('classes', 'test', batch=2)

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

  # Datasets
  datasets_names = {name for name in data.datasets}

  # Parsing arguments
  parser = argparse.ArgumentParser(description=
      'ArtReCycle: my implementation of CycleGAN with Tensorflow 2.0')
  op_parsers = parser.add_subparsers(help='Operation', dest='op')

  # Train op
  train_parser = op_parsers.add_parser('train', help='Train the net')
  train_parser.add_argument('-d', '--dataset', nargs=2, required=True,
      choices=datasets_names, metavar=('datasetA', 'datasetB'),
      help='Input dataset. Choose from: '+str(datasets_names))

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
