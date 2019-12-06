#!/usr/bin/env python3

'''\
Main script file. Implementation of CycleGAN paper with Tensorflow 2.0.
'''

import argparse
import tensorflow as tf

import data

# Debug
from matplotlib import pyplot as plt


def train(args):
  '''\
  Training function.

  Args:
    args: namespace of arguments. Run --help for info.
  '''

  raise NotImplementedError()


def use(args):
  '''\
  Transform images with the net.

  Args:
    args: namespace of arguments. Run --help for info.
  '''

  raise NotImplementedError()


def debug(args):
  '''\
  Debugging function.

  Args:
    args: namespace of arguments. Run --help for info.
  '''
  print('> Debug')

  # Testing the input pipeline
  dataset = data.load(*args.args, batch=2)
  for batch in dataset:
    for img in batch:
      plt.imshow(tf.cast(img, dtype=tf.uint8))
      plt.show()
      input()



def main():
  '''\
  Main function. Called when this file is executed as script.
  '''


  ## Parsing arguments
  parser = argparse.ArgumentParser(description=
      'ArtReCycle: my implementation of CycleGAN with Tensorflow 2.0')
  op_parsers = parser.add_subparsers(help='Operation', dest='op')

  # Train op
  train_parser = op_parsers.add_parser('train', help='Train the net')

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
