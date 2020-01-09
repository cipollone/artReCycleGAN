'''\
Data module: loading Datasets.

This project uses the following convention for datases:
Each dataset is a set of jpeg pictures in a directory. In the same directory,
an info file, usually called 'dataset.txt', contains the list of images to
load from that folder (some images might be excluded). This file must contain
an empty line: this is interpreted as the separator for the training/testing
split.
'''

import os
import tensorflow as tf


# Supported datasets. 'name': 'info file'
#   These contain a list of images to consider in their directories.
#   A blank line is used to separate train images from test
datasets = {
    'sisley': '../datasets/art/Alfred Sisley/dataset.txt',
    'guillaumin': '../datasets/art/Armand Guillaumin/dataset.txt',
    'caravaggio': '../datasets/art/Caravaggio/dataset.txt',
    'monet': '../datasets/art/Claude Monet/dataset.txt',
    'manet': '../datasets/art/Edouard Manet/dataset.txt',
    'caillebotte': '../datasets/art/Gustave Caillebotte/dataset.txt',
    'signac': '../datasets/art/Paul Signac/dataset.txt',
    'vangogh': '../datasets/art/Vincent van Gogh/dataset.txt',
    'foto': '../datasets/foto/dataset.txt',
    'summer': '../datasets/summer/dataset.txt',
    'winter': '../datasets/winter/dataset.txt',
}


def _classification_dataset(split):
  '''\
  This dataset is only used during development as a simple classification task.
  Classifying monet's (0) over van Gogh's (1).

  Args:
    split: 'train' or 'test'

  Returns:
    Dataset of images (image, label), total number of images
  '''

  # Used datasets
  names = ('monet', 'vangogh')

  # Load all
  dataset = None
  size = 0
  for name, label in zip(names, (0,1)):

    # Images and labels
    images, n = _dataset_files(name, split)
    labels = tf.data.Dataset.from_tensors(label).repeat()
    dataset_set = tf.data.Dataset.zip((images, labels))

    dataset = dataset.concatenate(dataset_set) if dataset else dataset_set
    size += n

  return dataset, size


def _dataset_files(name, split):
  '''\
  Returns all filenames that compose the requested dataset.

  Args:
    name: a dataset name
    split: 'train', 'test' or 'all'

  Returns:
    Dataset of all filenames, total number of images.
  '''

  # Checks
  if not name in datasets or not split in ('train', 'test', 'all'):
    raise ValueError('Illegal dataset specification')

  # Dataset info
  info_file = datasets[name]
  dataset_dir = os.path.dirname(info_file)
  with open(info_file) as dataset_info:
    files = dataset_info.read().splitlines()

  # Split
  for split_i in range(len(files)):
    if len(files[split_i]) == 0:
      break
  if split == 'train':
    files = files[:split_i]
  elif split == 'test':
    files = files[split_i+1:]

  # Paths
  files = [os.path.join(dataset_dir, f) for f in files if f]

  # Dataset
  return tf.data.Dataset.from_tensor_slices(files), len(files)


def decode_image(path, out_shape):
  '''\
  Decodes a single image from path and resize it to the given dimension.

  Args:
    path: image file path
    out_shape: desired shape of each image
  Returns:
    An image Tensor
  '''

  # Read
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=out_shape[2])

  # Square crop
  shape = tf.shape(img)
  square_size = tf.reduce_min(shape[:2])
  img = tf.image.random_crop(img, [square_size, square_size, shape[2]])

  # Resize
  img = tf.image.resize(img, out_shape[:2])

  return img


def load(name, split, shape=(300, 300, 3), batch=None, shuffle=True):
  '''\
  Returns a Dataset. The dataset is already transformed to create the input
  pipeline.

  Args:
    name: a dataset name. ('classes' is a combined dataset for classification)
    split: 'train', 'test' or 'all'
    shape: desired shape of each image
    batch: how many samples to return. If None, the entire dataset is returned.
    shuffle: set to false if shuffling is not necessary.

  Returns:
    Tf Dataset, dataset size
  '''

  # Is this a classification task? Just for development
  classification = (name == 'classes')

  def load_image(path):
    return decode_image(path, shape)

  def load_labelled_image(path, label):
    return decode_image(path, shape), label

  # Dataset of paths
  images, size = _dataset_files(name, split) \
      if not classification else _classification_dataset(split)

  # Select batch
  if not batch or batch < 1 or batch > size :
    batch = size

  # Input pipeline
  if shuffle: images = images.shuffle(min(size, 10000))
  images = images.repeat()
  images = images.map(load_image \
      if not classification else load_labelled_image,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  images = images.batch(batch)
  images = images.prefetch(1)

  return images, size


def load_pair(name_A, name_B, split, **kwargs):
  ''' Load a pair of datasets '''

  dataset_A, size_A = load(name_A, split, **kwargs)
  dataset_B, size_B = load(name_B, split, **kwargs)

  # Dataset size is average of the two
  size = (size_A + size_B) / 2

  return tf.data.Dataset.zip((dataset_A, dataset_B)), size


def load_few(name, split, shape, n):
  '''\
  It may be useful to print just few images for visualization.
  These images shouldn't change. This function returns few samples from the
  required dataset.
  Args:
    name: a dataset name
    split: 'train', 'test' or 'all'
    shape: desired shape of each image
    n: number of images
  Returns:
    Images as a 4D tensor
  '''

  # Decode paths
  paths, size = _dataset_files(name, split)
  paths = [path.numpy() for path in paths]

  if size < n:
    raise RuntimeError(name + ' (' + split + ') does not contain ' + str(n) +
        ' images.')

  # Select n among all
  offset = int((size-n)/2)
  paths = paths[offset:offset+n]

  # Load all
  images = [decode_image(path, shape) for path in paths]
  return tf.convert_to_tensor(images)
