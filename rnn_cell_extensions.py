# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Module for constructing RNN Cells.

## Extensions to tensorflow's RNN class

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs

import collections
import math
import tensorflow as tf

class ResidualWrapper(RNNCell):
  """Operator adding residual connections to a given cell."""

  def __init__(self, cell):
    """Create a cell with added residual connection.

    Args:
      cell: an RNNCell. The input is added to the output.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add a residual connection."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Add the residual connection
    output = tf.add(output, inputs)

    return output, new_state

class LinearSpaceEncoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, input_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    # Do a simple linear layer
    # FIXME pass an initializer as well
    #with vs.variable_scope( scope or "linear_space_encoder_rnn"):
    self._cell = cell
    print( 'Encoder: input_size = {0}'.format(input_size) )
    print( 'Encoder: state_size = {0}'.format(self._cell.state_size) )
    #self.input_size = input_size

    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      outsize = self._cell.state_size[0]

      # LSTMStateTuple if LSTM
      if isinstance( outsize, LSTMStateTuple ):
        outsize = outsize.h

    else:
      # Fine if not multi-rnn
      outsize = self._cell.state_size

    self.w_in = tf.get_variable("proj_w_in",
        [input_size, outsize],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    self.b_in = tf.get_variable("proj_b_in", [outsize],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Use a linear layer and pass the output to the cell."""

    #print( "Inputs: {0}".format( inputs ) )

    # Apply the multiplication to everything
    inputs = tf.matmul(inputs, self.w_in) + self.b_in

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    return output, new_state

class LinearSpaceDecoderWrapper(RNNCell):
  """Operator adding a linear encoder to an RNN cell"""

  def __init__(self, cell, output_size):
    """Create a cell with with a linear encoder in space.

    Args:
      cell: an RNNCell. The input is passed through a linear layer.

    Raises:
      TypeError: if cell is not an RNNCell.
    """
    if not isinstance(cell, RNNCell):
      raise TypeError("The parameter cell is not a RNNCell.")

    self._cell = cell

    print( 'output_size = {0}'.format(output_size) )
    print( ' state_size = {0}'.format(self._cell.state_size) )

    #with vs.variable_scope( scope or "linear_space_decoder_rnn"):
    # FIXME state size is a tuple when using a multi-rnn


    # Tuple if multi-rnn
    if isinstance(self._cell.state_size,tuple):

      # Fine if GRU...
      insize = self._cell.state_size[-1]

      # LSTMStateTuple if LSTM
      if isinstance( insize, LSTMStateTuple ):
        insize = insize.h

    else:
      # Fine if not multi-rnn
      insize = self._cell.state_size

    self.w_out = tf.get_variable("proj_w_out",
        #[self._cell.state_size[-1] if isinstance(self._cell.state_size,tuple) else self._cell.state_size , output_size],
        [insize, output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
    self.b_out = tf.get_variable("proj_b_out", [output_size],
        dtype=tf.float32,
        initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

    self.linear_output_size = output_size


  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self.linear_output_size

  def __call__(self, inputs, state, scope=None):
    """Use a linear layer and pass the output to the cell."""

    # Run the rnn as usual
    output, new_state = self._cell(inputs, state, scope)

    # Apply the multiplication to everything
    output = tf.matmul(output, self.w_out) + self.b_out

    return output, new_state
