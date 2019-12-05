#!/usr/bin/env python3

'''\
Main script file. Implementation of CycleGAN paper with Tensorflow 2.0.
'''

import argparse



def train():
  '''\
  Training function.
  '''

  raise NotImplementedError()


def use():
  '''\
  Transform images with the net.
  '''

  raise NotImplementedError()


def debug():
  '''\
  Debugging function.
  '''
  print('> Debug')


def main():
  '''\
  Main function. Called when this file is executed as script.
  '''

  # Datasets. Identifiers and paths
  # TODO: create consistent train/test split in txt files
  datasets = {
      'sisley': '../datasets/art/Alfred Sisley',
      'guillaumin': '../datasets/art/Armand Guillaumin',
      'caravaggio': '../datasets/art/Caravaggio',
      'monet': '../datasets/art/Claude Monet',
      'manet': '../datasets/art/Edouard Manet',
      'caillebotte': '../datasets/art/Gustave Caillebotte',
      'signac': '../datasets/art/Paul Signac',
      'vangogh': '../datasets/art/Vincent van Gogh',
      'foto': '../datasets/foto',
      'summer': '../datasets/summer',
      'winter': '../datasets/winter',
  }

  ## Parsing arguments
  parser = argparse.ArgumentParser(description=
      'ArtReCycle: my implementation of CycleGAN with Tensorflow 2.0')
  parser.add_argument('op', choices=['train','use','debug'],
      help='What to do with the net. Most options only affect training.')

  datasets_help = 'One of the available 
  #parser.add_argument('-f', '--from', required=True, 

  args = parser.parse_args()

  # Go
  if args.op == 'train':
    train()
  elif args.op == 'use':
    use()
  elif args.op == 'debug':
    debug()


if __name__ == '__main__':
  main()
