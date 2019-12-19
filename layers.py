'''\
Custom keras layers that are used in the model.
'''

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses
import inspect


class BaseLayer(layers.Layer):
  '''\
  Base class for all layers and model parts.
  This is mainly used to create namespaces in TensorBoard graphs.
  This must be subclassed, not instantiated directly. Subclasses can push
  layers to self.layers_stack to create a simple sequential layer, otherwise
  they must override call(). self.compile_defaults is a dictionary of default
  argument when one of these layers is used as a model.
  '''

  # Number all layers. Map classes to count
  layers_count = {}

  def __init__(self, **kwargs):

    # Only from subclasses
    this_class = self.__class__
    if this_class is BaseLayer:
      raise NotImplementedError('BaseLayer is an abstract class')

    # Add this layer to count
    if not this_class in BaseLayer.layers_count:
      BaseLayer.layers_count[this_class] = 0
    else:
      BaseLayer.layers_count[this_class] += 1

    # Choose a name
    if not 'name' in kwargs:
      name = this_class.__name__
      if BaseLayer.layers_count[this_class] > 0:
        name += '_' + str(BaseLayer.layers_count[this_class])
      kwargs['name'] = name

    # Empty layers and compile options
    self.layers_stack = []
    self.compile_defaults = {}

    # Super
    layers.Layer.__init__(self, **kwargs)


  def call(self, inputs):
    ''' Sequential layer by default. '''

    # This must be overridden if layers_stack is not used
    if not self.layers_stack:
      raise NotImplementedError(
        'call() must be overridden if self.layers_stack is not used.')

    # Sequential model by default
    for layer in self.layers_stack:
      inputs = layer(inputs)
    return inputs


  def get_config(self):
    ''' Empty dict is fine if subclasses constructors accept no arguments '''

    return layers.Layer.get_config(self)


def _make_layer(name, function):
  '''\
  Creates a keras layer that calls the given function.
  The first argument of the function must be for the inputs, and other options
  may follow. If created Layer class is initialized with arguments for
  'function', they will be used at each call.

  Args:
    name: the name of this layer
    function: the function to call.
  Returns:
    a keras layer that calls function
  '''

  def __init__(self, **kwargs):
    ''' Layer constructor '''

    # Store the inner function
    self._function = function

    # Set the argument defaults
    signature = inspect.signature(function)
    arg_names = list(signature.parameters)
    function_kwargs = { arg: kwargs[arg] \
        for arg in kwargs if arg in arg_names }
    layer_kwargs = { arg: kwargs[arg] \
        for arg in kwargs if not arg in arg_names}
    self._function_bound_args = signature.bind_partial(**function_kwargs)

    # Super
    BaseLayer.__init__(self, **layer_kwargs)


  def call(self, inputs, **kwargs):
    ''' Layer call method '''

    defaults = self._function_bound_args.arguments
    kwargs.pop('training', None)
    return self._function(inputs, **defaults, **kwargs)

  # Define layer
  LayerClass = type(name, (BaseLayer,), { '__init__': __init__, 'call': call })
  return LayerClass


def layerize(name, scope):
  '''\
  Function decorator. Keras wrapper for Tf ops.
  Defines a layer with the ops of the decorated function.
  The original function is still available, so it can be used when a 
  layer is not needed. Only suitable for tf ops,
  without persistent variables. See _make_layer for further help.
  Layers are the preferred way to create namespaces.

  Args:
    name: destination class name
    scope: namespace (dict) where to define the class (eg. globals()).
  Returns:
    The same decorated function
  '''

  # Decorator
  def decorator(keras_f):

    # Create a layer
    NewLayer = _make_layer(name=name, function=keras_f)

    # Export layer
    scope[name] = NewLayer

    # Return same function
    return keras_f

  # Decorate
  return decorator


class InstanceNormalization(BaseLayer):
  '''\
  Instance normalization substitutes Batch normalization in CycleGan paper.
  From their original implementation, I see that the affine transformation
  is not included, but I leave an option here.
  '''

  def __init__(self, affine=False, **kwargs):
    '''\
    If affine is true, each channel is transformed with a scalar affine
    transformation (adding scale and offset parameters).
    '''

    BaseLayer.__init__(self, **kwargs)
    self.affine = affine


  def build(self, input_shape):

    if self.affine:
      scalars_shape = [1, 1, 1, input_shape[3]]
      self.scale = self.add_weight(shape = scalars_shape, name='scale')
      self.offset = self.add_weight(shape = scalars_shape, name='offset')

    BaseLayer.build(self, input_shape)


  def call(self, inputs):
    '''\
    Args:
      inputs: a batch of 3d tesors (4d input)
    Returns:
      A batch of 3d tensors
    '''

    eps = 1e-4

    # Normalize in image width height dimensions
    mean, var = tf.nn.moments(inputs, axes=[1,2], keepdims=True)
    inputs = (inputs - mean) / tf.sqrt(var + eps)

    # Affine transformation
    if self.affine:
      inputs = self.scale * inputs + self.offset

    return inputs


  def get_config(self):

    config = BaseLayer.get_config(self)
    config.update({ 'affine': self.affine })
    return config


class GeneralConvBlock(BaseLayer):
  '''\
  Generic convolutional block that consists of:
  - optional padding
  - 2d convolution (no padding)
  - Instance normalization
  - ReLU activation
  '''

  def __init__(self, filters, kernel_size, stride=1, pad=0, **kwargs):
    '''\
    Options for this block.
    Args:
      filters: number of filters (output channels)
      kernel_size: lenght of the square kernel
      stride: convolution stride
      pad: if given, a reflection padding of 'pad' values is added before
        convolution.
    '''

    # Save options
    self.block_options = {
        'filters': filters,
        'kernel_size': kernel_size,
        'stride': stride,
        'pad': pad,
      }
    
    # Super
    BaseLayer.__init__(self, **kwargs)


  def build(self, input_shape):
    ''' Instantiations '''

    # Vars
    stack = []
    filters, kernel_size, stride, pad = [ self.block_options[opt] for opt in \
        ('filters', 'kernel_size', 'stride', 'pad') ]

    # Padding
    if pad:
      stack.append( PadReflection(pad=pad) )

    # Convolution
    stack.append( layers.Conv2D( filters=filters, kernel_size=kernel_size,
        strides=stride, padding='valid') )

    # Normalization
    stack.append( InstanceNormalization() )

    # Activation
    stack.append( tf.keras.layers.ReLU() )

    # Store
    self.layers_stack = stack

    # Super
    BaseLayer.build(self, input_shape)


  def get_config(self):

    config = BaseLayer.get_config(self)
    config.update( self.block_options )
    return config


@layerize('ImagePreprocessing', globals())
def image_preprocessing(inputs, out_size):
  '''\
  Initial preprocessing for a batch of images.
  Random crop, flip, normalization.
  Expects as input, when called, a batch of images (4D tensor).
  NOTE: I couldn't use built-in Tf ops for images, due to a probable Tf bug.

  Args:
    inputs: a batch of images (4d input)
    out_shape: (out_height, out_width) Image size after preprocessing.
        out_shape must be smaller than input shape.
  Returns:
    A batch of images
  '''

  # Shapes
  input_shape = tf.shape(inputs)
  batch_size = input_shape[0]
  in_size = input_shape[1:3]
  images = inputs

  # Random crop
  with tf.name_scope('RandomCrop') as scope:
    crop_shift = in_size - out_size
    shift0 = tf.random.uniform([], minval=0, maxval=crop_shift[0],
        dtype=tf.dtypes.int32, name='RandomU1')
    shift1 = tf.random.uniform([], minval=0, maxval=crop_shift[1],
        dtype=tf.dtypes.int32, name='RandomU2')
    images = images[:, shift0:(shift0+out_size[0]),
        shift1:(shift1+out_size[1]), :]  # same crop for all in batch
    images = tf.ensure_shape(images, [None, out_size[0], out_size[1], None])

  # Random flip
  with tf.name_scope('RandomFlip') as scope:
    flips = tf.random.uniform([batch_size], minval=0, maxval=2,
        dtype=tf.dtypes.int32)
    flips = tf.cast(tf.reshape(flips, (batch_size, 1, 1, 1)), # image selection
        dtype=tf.dtypes.float32)
    flipped_images = tf.reverse(images, axis=[2])             # flip all
    images = flipped_images * flips + images * (1-flips)      # select

  # Normalization
  with tf.name_scope('Normalization') as scope:
    images = images/127.5 - 1

  return images


@layerize('ReduceMean', globals())
def reduce_mean(inputs):
  '''\
  Simply collapse all inputs to the average value.
  '''

  return tf.math.reduce_mean(inputs, axis=(1,2,3))


@layerize('PadReflection', globals())
def pad_reflection(inputs, pad):
  '''\
  Apply a pad, with reflection strategy, to all input images.
  
  Args:
    inputs: a batch of images 4D tensor
    pad: number of values to add around each image
  Returns:
    padded input batch
  '''

  inputs = tf.pad(inputs,
      paddings = [[0,0], [pad,pad], [pad,pad], [0,0]],
      mode = 'reflect')
  return inputs


class BinaryAccuracyFromLogits(tf.keras.metrics.Metric):
  '''\
  Just line BinaryAccuracy, but when the output of the model is in logits.
  '''

  def __init__(self, **kwargs):

    # Name
    layer_name = kwargs.get('name', None)
    if not layer_name:
      kwargs['name'] = 'BinaryAccuracy'

    # Internal metric and super
    self._inner_metric = tf.keras.metrics.BinaryAccuracy()
    tf.keras.metrics.Metric.__init__(self, **kwargs)


  def update_state(self, y_true, y_pred, sample_weight=None):

    # Transform
    logits = y_pred
    probabilities = tf.keras.activations.sigmoid(logits)

    # Forward
    self._inner_metric.update_state(y_true, probabilities, sample_weight)


  def result(self):
    return self._inner_metric.result()

