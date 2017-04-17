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

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

import random

import numpy as np
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import rnn_cell_extensions # my extensions of the tf repos
import data_utils

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder.
  """

  def __init__(self,
               architecture,
               source_seq_len,
               target_seq_len,
               rnn_size, # hidden recurrent layer size
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               summaries_dir,
               loss_to_use,
               residual_rnn,
               use_space_encoder,
               number_of_actions,
               one_hot=True,
               residual_velocities=False,
               loss_velocities_weight=1.0, # Weight to give to velocities
               forward_only=False,
               dtype=tf.float32):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """

    self.HUMAN_SIZE = 54
    self.input_size = self.HUMAN_SIZE + number_of_actions if one_hot else self.HUMAN_SIZE

    print( "One hot is ", one_hot )
    print( "Input size is %d" % self.input_size )

    # Summary writers for train and test runs
    self.train_writer = tf.summary.FileWriter(os.path.join( summaries_dir, 'train'))
    self.test_writer  = tf.summary.FileWriter(os.path.join( summaries_dir, 'test'))

    self.source_seq_len = source_seq_len
    self.target_seq_len = target_seq_len
    self.rnn_size = rnn_size
    self.batch_size = batch_size
    self.learning_rate = tf.Variable( float(learning_rate), trainable=False, dtype=dtype )
    self.learning_rate_decay_op = self.learning_rate.assign( self.learning_rate * learning_rate_decay_factor )
    self.global_step = tf.Variable(0, trainable=False)

    # === Create the RNN that will keep the state ===
    print('rnn_size = {0}'.format( rnn_size ))
    single_cell = tf.contrib.rnn.GRUCell( self.rnn_size )

    # Might have residual connection
    if residual_rnn:
      single_cell = rnn_cell_extensions.ResidualWrapper( single_cell )

    # Might be a stack of many layers
    if num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell( [single_cell] * num_layers )
    else:
      cell = single_cell

    # === Transform the inputs ===
    with tf.name_scope("inputs"):

      enc_in = tf.placeholder(dtype, shape=[None, source_seq_len-1, self.input_size], name="enc_in")
      dec_in = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_in")
      dec_out = tf.placeholder(dtype, shape=[None, target_seq_len, self.input_size], name="dec_out")

      self.encoder_inputs = enc_in
      self.decoder_inputs = dec_in
      self.decoder_outputs = dec_out

      enc_in = tf.transpose(enc_in, [1, 0, 2])
      dec_in = tf.transpose(dec_in, [1, 0, 2])
      dec_out = tf.transpose(dec_out, [1, 0, 2])

      enc_in = tf.reshape(enc_in, [-1, self.input_size])
      dec_in = tf.reshape(dec_in, [-1, self.input_size])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])

      enc_in = tf.split(enc_in, source_seq_len-1, axis=0)
      dec_in = tf.split(dec_in, target_seq_len, axis=0)
      dec_out = tf.split(dec_out, target_seq_len, axis=0)

    # === Add space encoders and decoders ===
    if use_space_encoder:
      cell = rnn_cell_extensions.LinearSpaceEncoderWrapper( cell, self.input_size )

    cell = rnn_cell_extensions.LinearSpaceDecoderWrapper( cell, self.input_size )

    # Finally, wrap everything in a residual layer if we want to model velocities
    if residual_velocities:
      cell = rnn_cell_extensions.ResidualWrapper( cell )

    # Store the outputs here
    outputs  = []

    # Define the loss function
    lf = None

    if loss_to_use == "supervised":
      pass
    elif loss_to_use == "self_fed":
      def lf(prev, i): # function for self-fed loss
        return prev
    else:
      raise(ValueError, "unknown loss: %s" % loss_to_use)

    print( "Loop function is ", lf )

    # Build the RNN
    if architecture == "basic":
      # Basic RNN does not have a loop function in its API, so Copying here.
      with vs.variable_scope("basic_rnn_seq2seq"):
        _, enc_state = tf.nn.rnn(cell, enc_in, dtype=tf.float32) # Encoder
        outputs, self.states = tf.nn.seq2seq.rnn_decoder( dec_in, enc_state, cell, loop_function=lf ) # Decoder

    elif architecture == "tied":
      # outputs, self.states = tf.nn.seq2seq.tied_rnn_seq2seq( enc_in, dec_in, cell, loop_function=lf )
      outputs, self.states = tf.contrib.legacy_seq2seq.tied_rnn_seq2seq( enc_in, dec_in, cell, loop_function=lf )

    elif architecture == "attention":
      with vs.variable_scope("attention_rnn_seq2seq"):

        # Simple encoder
        enc_outputs, enc_state = tf.nn.rnn(cell, enc_in, dtype=tf.float32)

        # Get the top states
        top_states = [array_ops.reshape(e, [-1, 1, cell.output_size])
            for e in enc_outputs]
        attention_states = array_ops.concat(1, top_states)

        # Pass to the attention decoder
        outputs, self.states = tf.nn.seq2seq.attention_decoder(dec_in, enc_state,
                      attention_states, cell, loop_function=lf)

    else:
        raise(ValueError, "Uknown architecture: %s" % architecture )

    self.outputs = outputs

    with tf.name_scope("loss_positions"): # Loss in positions

      loss_positions = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))

    with tf.name_scope("loss_velocities"): # Loss in velocities

      # Revert the output
      outputs = tf.concat( outputs, axis=0 )
      outputs = tf.reshape( outputs, [target_seq_len, -1, self.input_size] )
      outputs = tf.transpose( outputs, [1, 0, 2] )
      outputs = tf.subtract(outputs[:,1:,], outputs[:,:-1,])

      # Rever the input
      dec_out = tf.concat( dec_out, axis=0 )
      dec_out = tf.reshape( dec_out, [target_seq_len, -1, self.input_size] )
      dec_out = tf.transpose( dec_out, [1, 0, 2] )
      dec_out = tf.subtract(dec_out[:,1:,], dec_out[:,:-1,])

      loss_velocities   = tf.reduce_mean(tf.square(tf.subtract(dec_out, outputs)))

    self.loss         = ((1.0 - loss_velocities_weight) * loss_positions) + (loss_velocities_weight * loss_velocities)
    self.loss_summary = tf.summary.scalar('loss/loss', self.loss)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    opt = tf.train.GradientDescentOptimizer( self.learning_rate )
    #opt = tf.train.AdamOptimizer( self.learning_rate )

    # Update all the trainable parameters
    gradients = tf.gradients( self.loss, params )

    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)

    # Keep track of the learning rate
    self.learning_rate_summary = tf.summary.scalar('learning_rate/learning_rate', self.learning_rate)

    # actions = ["directions", "greeting", "phoning",
    #             "posing", "purchases", "sitting", "sittingdown",
    #             "takingphoto", "waiting", "walkingdog", "walkingtogether"]

    # === variables for loss in Euler Angles -- for each action
    with tf.name_scope( "euler_error_walking" ):
      self.walking_err80   = tf.placeholder( tf.float32, name="walking_ashesh_seeds_0080" )
      self.walking_err160  = tf.placeholder( tf.float32, name="walking_ashesh_seeds_0160" )
      self.walking_err320  = tf.placeholder( tf.float32, name="walking_ashesh_seeds_0320" )
      self.walking_err400  = tf.placeholder( tf.float32, name="walking_ashesh_seeds_0400" )
      self.walking_err560  = tf.placeholder( tf.float32, name="walking_ashesh_seeds_0560" )
      self.walking_err1000 = tf.placeholder( tf.float32, name="walking_ashesh_seeds_1000" )

      self.walking_err80_summary   = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_0080', self.walking_err80 )
      self.walking_err160_summary  = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_0160', self.walking_err160 )
      self.walking_err320_summary  = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_0320', self.walking_err320 )
      self.walking_err400_summary  = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_0400', self.walking_err400 )
      self.walking_err560_summary  = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_0560', self.walking_err560 )
      self.walking_err1000_summary = tf.summary.scalar( 'euler_error_walking/ashesh_seeds_1000', self.walking_err1000 )
    with tf.name_scope( "euler_error_eating" ):
      self.eating_err80   = tf.placeholder( tf.float32, name="eating_ashesh_seeds_0080" )
      self.eating_err160  = tf.placeholder( tf.float32, name="eating_ashesh_seeds_0160" )
      self.eating_err320  = tf.placeholder( tf.float32, name="eating_ashesh_seeds_0320" )
      self.eating_err400  = tf.placeholder( tf.float32, name="eating_ashesh_seeds_0400" )
      self.eating_err560  = tf.placeholder( tf.float32, name="eating_ashesh_seeds_0560" )
      self.eating_err1000 = tf.placeholder( tf.float32, name="eating_ashesh_seeds_1000" )

      self.eating_err80_summary   = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_0080', self.eating_err80 )
      self.eating_err160_summary  = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_0160', self.eating_err160 )
      self.eating_err320_summary  = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_0320', self.eating_err320 )
      self.eating_err400_summary  = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_0400', self.eating_err400 )
      self.eating_err560_summary  = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_0560', self.eating_err560 )
      self.eating_err1000_summary = tf.summary.scalar( 'euler_error_eating/ashesh_seeds_1000', self.eating_err1000 )
    with tf.name_scope( "euler_error_smoking" ):
      self.smoking_err80   = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_0080" )
      self.smoking_err160  = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_0160" )
      self.smoking_err320  = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_0320" )
      self.smoking_err400  = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_0400" )
      self.smoking_err560  = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_0560" )
      self.smoking_err1000 = tf.placeholder( tf.float32, name="smoking_ashesh_seeds_1000" )

      self.smoking_err80_summary   = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_0080', self.smoking_err80 )
      self.smoking_err160_summary  = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_0160', self.smoking_err160 )
      self.smoking_err320_summary  = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_0320', self.smoking_err320 )
      self.smoking_err400_summary  = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_0400', self.smoking_err400 )
      self.smoking_err560_summary  = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_0560', self.smoking_err560 )
      self.smoking_err1000_summary = tf.summary.scalar( 'euler_error_smoking/ashesh_seeds_1000', self.smoking_err1000 )
    with tf.name_scope( "euler_error_discussion" ):
      self.discussion_err80   = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_0080" )
      self.discussion_err160  = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_0160" )
      self.discussion_err320  = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_0320" )
      self.discussion_err400  = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_0400" )
      self.discussion_err560  = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_0560" )
      self.discussion_err1000 = tf.placeholder( tf.float32, name="discussion_ashesh_seeds_1000" )

      self.discussion_err80_summary   = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_0080', self.discussion_err80 )
      self.discussion_err160_summary  = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_0160', self.discussion_err160 )
      self.discussion_err320_summary  = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_0320', self.discussion_err320 )
      self.discussion_err400_summary  = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_0400', self.discussion_err400 )
      self.discussion_err560_summary  = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_0560', self.discussion_err560 )
      self.discussion_err1000_summary = tf.summary.scalar( 'euler_error_discussion/ashesh_seeds_1000', self.discussion_err1000 )
    with tf.name_scope( "euler_error_directions" ):
      self.directions_err80   = tf.placeholder( tf.float32, name="directions_ashesh_seeds_0080" )
      self.directions_err160  = tf.placeholder( tf.float32, name="directions_ashesh_seeds_0160" )
      self.directions_err320  = tf.placeholder( tf.float32, name="directions_ashesh_seeds_0320" )
      self.directions_err400  = tf.placeholder( tf.float32, name="directions_ashesh_seeds_0400" )
      self.directions_err560  = tf.placeholder( tf.float32, name="directions_ashesh_seeds_0560" )
      self.directions_err1000 = tf.placeholder( tf.float32, name="directions_ashesh_seeds_1000" )

      self.directions_err80_summary   = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_0080', self.directions_err80 )
      self.directions_err160_summary  = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_0160', self.directions_err160 )
      self.directions_err320_summary  = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_0320', self.directions_err320 )
      self.directions_err400_summary  = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_0400', self.directions_err400 )
      self.directions_err560_summary  = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_0560', self.directions_err560 )
      self.directions_err1000_summary = tf.summary.scalar( 'euler_error_directions/ashesh_seeds_1000', self.directions_err1000 )
    with tf.name_scope( "euler_error_greeting" ):
      self.greeting_err80   = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_0080" )
      self.greeting_err160  = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_0160" )
      self.greeting_err320  = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_0320" )
      self.greeting_err400  = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_0400" )
      self.greeting_err560  = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_0560" )
      self.greeting_err1000 = tf.placeholder( tf.float32, name="greeting_ashesh_seeds_1000" )

      self.greeting_err80_summary   = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_0080', self.greeting_err80 )
      self.greeting_err160_summary  = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_0160', self.greeting_err160 )
      self.greeting_err320_summary  = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_0320', self.greeting_err320 )
      self.greeting_err400_summary  = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_0400', self.greeting_err400 )
      self.greeting_err560_summary  = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_0560', self.greeting_err560 )
      self.greeting_err1000_summary = tf.summary.scalar( 'euler_error_greeting/ashesh_seeds_1000', self.greeting_err1000 )
    with tf.name_scope( "euler_error_phoning" ):
      self.phoning_err80   = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_0080" )
      self.phoning_err160  = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_0160" )
      self.phoning_err320  = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_0320" )
      self.phoning_err400  = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_0400" )
      self.phoning_err560  = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_0560" )
      self.phoning_err1000 = tf.placeholder( tf.float32, name="phoning_ashesh_seeds_1000" )

      self.phoning_err80_summary   = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_0080', self.phoning_err80 )
      self.phoning_err160_summary  = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_0160', self.phoning_err160 )
      self.phoning_err320_summary  = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_0320', self.phoning_err320 )
      self.phoning_err400_summary  = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_0400', self.phoning_err400 )
      self.phoning_err560_summary  = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_0560', self.phoning_err560 )
      self.phoning_err1000_summary = tf.summary.scalar( 'euler_error_phoning/ashesh_seeds_1000', self.phoning_err1000 )
    with tf.name_scope( "euler_error_posing" ):
      self.posing_err80   = tf.placeholder( tf.float32, name="posing_ashesh_seeds_0080" )
      self.posing_err160  = tf.placeholder( tf.float32, name="posing_ashesh_seeds_0160" )
      self.posing_err320  = tf.placeholder( tf.float32, name="posing_ashesh_seeds_0320" )
      self.posing_err400  = tf.placeholder( tf.float32, name="posing_ashesh_seeds_0400" )
      self.posing_err560  = tf.placeholder( tf.float32, name="posing_ashesh_seeds_0560" )
      self.posing_err1000 = tf.placeholder( tf.float32, name="posing_ashesh_seeds_1000" )

      self.posing_err80_summary   = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_0080', self.posing_err80 )
      self.posing_err160_summary  = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_0160', self.posing_err160 )
      self.posing_err320_summary  = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_0320', self.posing_err320 )
      self.posing_err400_summary  = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_0400', self.posing_err400 )
      self.posing_err560_summary  = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_0560', self.posing_err560 )
      self.posing_err1000_summary = tf.summary.scalar( 'euler_error_posing/ashesh_seeds_1000', self.posing_err1000 )
    with tf.name_scope( "euler_error_purchases" ):
      self.purchases_err80   = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_0080" )
      self.purchases_err160  = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_0160" )
      self.purchases_err320  = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_0320" )
      self.purchases_err400  = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_0400" )
      self.purchases_err560  = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_0560" )
      self.purchases_err1000 = tf.placeholder( tf.float32, name="purchases_ashesh_seeds_1000" )

      self.purchases_err80_summary   = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_0080', self.purchases_err80 )
      self.purchases_err160_summary  = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_0160', self.purchases_err160 )
      self.purchases_err320_summary  = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_0320', self.purchases_err320 )
      self.purchases_err400_summary  = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_0400', self.purchases_err400 )
      self.purchases_err560_summary  = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_0560', self.purchases_err560 )
      self.purchases_err1000_summary = tf.summary.scalar( 'euler_error_purchases/ashesh_seeds_1000', self.purchases_err1000 )
    with tf.name_scope( "euler_error_sitting" ):
      self.sitting_err80   = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_0080" )
      self.sitting_err160  = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_0160" )
      self.sitting_err320  = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_0320" )
      self.sitting_err400  = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_0400" )
      self.sitting_err560  = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_0560" )
      self.sitting_err1000 = tf.placeholder( tf.float32, name="sitting_ashesh_seeds_1000" )

      self.sitting_err80_summary   = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_0080', self.sitting_err80 )
      self.sitting_err160_summary  = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_0160', self.sitting_err160 )
      self.sitting_err320_summary  = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_0320', self.sitting_err320 )
      self.sitting_err400_summary  = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_0400', self.sitting_err400 )
      self.sitting_err560_summary  = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_0560', self.sitting_err560 )
      self.sitting_err1000_summary = tf.summary.scalar( 'euler_error_sitting/ashesh_seeds_1000', self.sitting_err1000 )
    with tf.name_scope( "euler_error_sittingdown" ):
      self.sittingdown_err80   = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_0080" )
      self.sittingdown_err160  = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_0160" )
      self.sittingdown_err320  = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_0320" )
      self.sittingdown_err400  = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_0400" )
      self.sittingdown_err560  = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_0560" )
      self.sittingdown_err1000 = tf.placeholder( tf.float32, name="sittingdown_ashesh_seeds_1000" )

      self.sittingdown_err80_summary   = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_0080', self.sittingdown_err80 )
      self.sittingdown_err160_summary  = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_0160', self.sittingdown_err160 )
      self.sittingdown_err320_summary  = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_0320', self.sittingdown_err320 )
      self.sittingdown_err400_summary  = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_0400', self.sittingdown_err400 )
      self.sittingdown_err560_summary  = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_0560', self.sittingdown_err560 )
      self.sittingdown_err1000_summary = tf.summary.scalar( 'euler_error_sittingdown/ashesh_seeds_1000', self.sittingdown_err1000 )
    with tf.name_scope( "euler_error_takingphoto" ):
      self.takingphoto_err80   = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_0080" )
      self.takingphoto_err160  = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_0160" )
      self.takingphoto_err320  = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_0320" )
      self.takingphoto_err400  = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_0400" )
      self.takingphoto_err560  = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_0560" )
      self.takingphoto_err1000 = tf.placeholder( tf.float32, name="takingphoto_ashesh_seeds_1000" )

      self.takingphoto_err80_summary   = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_0080', self.takingphoto_err80 )
      self.takingphoto_err160_summary  = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_0160', self.takingphoto_err160 )
      self.takingphoto_err320_summary  = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_0320', self.takingphoto_err320 )
      self.takingphoto_err400_summary  = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_0400', self.takingphoto_err400 )
      self.takingphoto_err560_summary  = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_0560', self.takingphoto_err560 )
      self.takingphoto_err1000_summary = tf.summary.scalar( 'euler_error_takingphoto/ashesh_seeds_1000', self.takingphoto_err1000 )
    with tf.name_scope( "euler_error_waiting" ):
      self.waiting_err80   = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_0080" )
      self.waiting_err160  = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_0160" )
      self.waiting_err320  = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_0320" )
      self.waiting_err400  = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_0400" )
      self.waiting_err560  = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_0560" )
      self.waiting_err1000 = tf.placeholder( tf.float32, name="waiting_ashesh_seeds_1000" )

      self.waiting_err80_summary   = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_0080', self.waiting_err80 )
      self.waiting_err160_summary  = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_0160', self.waiting_err160 )
      self.waiting_err320_summary  = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_0320', self.waiting_err320 )
      self.waiting_err400_summary  = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_0400', self.waiting_err400 )
      self.waiting_err560_summary  = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_0560', self.waiting_err560 )
      self.waiting_err1000_summary = tf.summary.scalar( 'euler_error_waiting/ashesh_seeds_1000', self.waiting_err1000 )
    with tf.name_scope( "euler_error_walkingdog" ):
      self.walkingdog_err80   = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_0080" )
      self.walkingdog_err160  = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_0160" )
      self.walkingdog_err320  = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_0320" )
      self.walkingdog_err400  = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_0400" )
      self.walkingdog_err560  = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_0560" )
      self.walkingdog_err1000 = tf.placeholder( tf.float32, name="walkingdog_ashesh_seeds_1000" )

      self.walkingdog_err80_summary   = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_0080', self.walkingdog_err80 )
      self.walkingdog_err160_summary  = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_0160', self.walkingdog_err160 )
      self.walkingdog_err320_summary  = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_0320', self.walkingdog_err320 )
      self.walkingdog_err400_summary  = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_0400', self.walkingdog_err400 )
      self.walkingdog_err560_summary  = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_0560', self.walkingdog_err560 )
      self.walkingdog_err1000_summary = tf.summary.scalar( 'euler_error_walkingdog/ashesh_seeds_1000', self.walkingdog_err1000 )
    with tf.name_scope( "euler_error_walkingtogether" ):
      self.walkingtogether_err80   = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_0080" )
      self.walkingtogether_err160  = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_0160" )
      self.walkingtogether_err320  = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_0320" )
      self.walkingtogether_err400  = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_0400" )
      self.walkingtogether_err560  = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_0560" )
      self.walkingtogether_err1000 = tf.placeholder( tf.float32, name="walkingtogether_ashesh_seeds_1000" )

      self.walkingtogether_err80_summary   = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_0080', self.walkingtogether_err80 )
      self.walkingtogether_err160_summary  = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_0160', self.walkingtogether_err160 )
      self.walkingtogether_err320_summary  = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_0320', self.walkingtogether_err320 )
      self.walkingtogether_err400_summary  = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_0400', self.walkingtogether_err400 )
      self.walkingtogether_err560_summary  = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_0560', self.walkingtogether_err560 )
      self.walkingtogether_err1000_summary = tf.summary.scalar( 'euler_error_walkingtogether/ashesh_seeds_1000', self.walkingtogether_err1000 )

    self.saver = tf.train.Saver( tf.all_variables(), max_to_keep=10 )

  def linear_space_encoder( self, inputs, dtype, scope=None ):
    """
    Creates all the operations that we want to apply to the input before passing
    it to the stack of rnns.

    Args:
      inputs: the ith-entry in space, to be transformed before being passed to the rnns.

    Returns:
      outputs: the transformed input after space encoding.
    """

    with vs.variable_scope( "linear_space_encoder" ):
      scope = scope or "space_encoder"

      w_in = tf.get_variable("proj_w_in", [self.input_size, self.rnn_size],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      b_in = tf.get_variable("proj_b_in", [self.rnn_size],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      # Apply the multiplication to everything
      outputs = [tf.matmul(input_, w_in) + b_in for input_ in inputs]

    return outputs

  def linear_space_decoder( self, inputs, dtype, scope=None ):
    """
    A space decoder, for after-rnn processing.
    """

    with vs.variable_scope( "linear_space_decoder" ):
      scope = scope or "space_decoder"

      w_out = tf.get_variable("proj_w_out", [self.rnn_size, self.HUMAN_SIZE],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))
      b_out = tf.get_variable("proj_b_out", [self.HUMAN_SIZE],
          dtype=dtype,
          initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04))

      # Apply the linear transform to the inputs
      outputs = [tf.matmul(input_, w_out) + b_out for input_ in inputs]

    return outputs


  def step(self, session, encoder_inputs, decoder_inputs, decoder_outputs,
             forward_only, ashesh_seeds=False ):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy vectors to feed as encoder inputs.
      decoder_inputs: list of numpy vectors to feed as decoder inputs.
      decoder_outputs: list of numpy vectors that are the expected decoder outputs.
      forward_only: whether to do the backward step or only forward.
      ashesh_seeds: True if you want to evaluate using the sequences of Ashesh

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      mean squared error, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    input_feed = {self.encoder_inputs: encoder_inputs,
                  self.decoder_inputs: decoder_inputs,
                  self.decoder_outputs: decoder_outputs}

    # Output feed: depends on whether we do a backward step or not.
    if not ashesh_seeds:
      if not forward_only:

        # Training step
        output_feed = [self.updates,       # Update Op that does SGD.
                     self.gradient_norms,  # Gradient norm.
                     self.loss,
                     self.loss_summary,
                     self.learning_rate_summary]

        outputs = session.run( output_feed, input_feed )
        return outputs[1], outputs[2], outputs[3], outputs[4]  # Gradient norm, loss, summaries

      else:
        # Validation step, not on Ashesh's seeds
        output_feed = [self.loss, # Loss for this batch.
                     self.loss_summary]

        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]  # No gradient norm
    else:
      # Validation on Ashesh's seeds
      output_feed = [self.loss, # Loss for this batch.
                    self.outputs,
                    self.loss_summary]

      outputs = session.run(output_feed, input_feed)

      return outputs[0], outputs[1], outputs[2]  # No gradient norm, loss, outputs.



  def get_batch( self, data, actions ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args:
      data: a list of sequences of size n-by-d to fit the model to.
      actions: a list of the actions we are using

    Returns:
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    # Select entries at random
    all_keys        = data.keys()
    chosen_keys = np.random.choice( len(all_keys), self.batch_size )

    # How many frames in total do we need?
    total_frames = self.source_seq_len + self.target_seq_len

    encoder_inputs  = np.zeros((self.batch_size, self.source_seq_len-1, self.input_size), dtype=float)
    decoder_inputs  = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)
    #decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.HUMAN_SIZE), dtype=float)
    decoder_outputs = np.zeros((self.batch_size, self.target_seq_len, self.input_size), dtype=float)

    for i in xrange( self.batch_size ):

      the_key = all_keys[ chosen_keys[i] ]

      # Get the number of frames
      n, _ = data[ the_key ].shape

      # Sample somewherein the middle
      idx = np.random.randint( 16, n-total_frames )

      # Select the data around the sampled points
      data_sel = data[ the_key ][idx:idx+total_frames ,:]

      # Add the data
      encoder_inputs[i,:,0:self.input_size]  = data_sel[0:self.source_seq_len-1, :]
      decoder_inputs[i,:,0:self.input_size]  = data_sel[self.source_seq_len-1:self.source_seq_len+self.target_seq_len-1, :]
      decoder_outputs[i,:,0:self.input_size] = data_sel[self.source_seq_len:, 0:self.input_size]

    return encoder_inputs, decoder_inputs, decoder_outputs

  def build_seeds(self, action, frames):
    seeds = []
    for i in range(8):
      # action, subsequence, idx
      seeds.append( (action, (i%2)+1, frames[i]) )

    return seeds

  def get_batch_ashesh(self, data, action ):
    """Get a random batch of data from the specified bucket, prepare for step.

    Args:
      data: a list of 2 sequences of size d-by-n to fit the model to.
      action: the action to load data from.

    Returns:
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """

    actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

    if not action in actions:
      raise ValueError("Unrecognized action {0}".format(action))

    frames = {}
    frames['walking']    = [1087, 955,  1145, 332,  660,  304,  201,  54]
    frames['smoking']    = [1426, 1398, 1087, 1180, 1329, 955,  1145, 332]
    frames['eating']     = [1426, 374,  1087, 156,  1329, 955,  1146, 332]
    frames['discussion'] = [1426, 2063, 1398, 1087, 1180, 1145, 332,  1438]

    # The ones we were missing
    frames['directions'] = [2299, 30, 1106, 625, 1829, 83, 1801, 170]
    frames['greeting'] = [422, 964, 138, 622, 234, 105, 216, 1344]
    frames['phoning'] = [844, 319, 324, 514, 685, 204, 1349, 241]
    frames['posing'] = [630, 23, 369, 722, 868, 595, 440, 174]
    frames['purchases'] = [1087, 109, 386, 210, 700, 498, 1009, 1129]
    frames['sitting'] = [150, 890, 508, 925, 517, 1334, 1571, 1354]
    frames['sittingdown'] = [2009, 1115, 81, 1428, 1493, 1412, 111, 594]
    frames['takingphoto'] = [222, 480, 496, 166, 1431, 556, 550, 681]
    frames['waiting'] = [451, 1383, 1476, 1727, 1393, 482, 1465, 1900]
    frames['walkingdog'] = [600, 486, 534, 278, 79, 341, 773, 541]
    frames['walkingtogether'] = [473, 875, 203, 1296, 1065, 489, 528, 1247]

    # Create the actual seed object
    seeds = self.build_seeds( action, frames[action] )

    batch_size = 8 # we always evaluate 8 seeds
    subject    = 5 # we always evaluate on subject 5
    source_seq_len = self.source_seq_len
    target_seq_len = self.target_seq_len

    encoder_inputs  = np.zeros( (batch_size, source_seq_len-1, self.input_size), dtype=float )
    decoder_inputs  = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )
    #decoder_outputs = np.zeros( (batch_size, target_seq_len, self.HUMAN_SIZE), dtype=float )
    decoder_outputs = np.zeros( (batch_size, target_seq_len, self.input_size), dtype=float )

    # How many frames in total do we need?
    total_frames = source_seq_len + target_seq_len

    # Trying to reproduce Ashesh's sequence cherry-picking as done in
    # https://github.com/libicocco/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in xrange( batch_size ):

      _, subsequence, idx = seeds[i]
      idx = idx + 50

      data_sel = data[ (subject, action, subsequence, 'even') ]
      #print( data_sel.shape )

      data_sel = data_sel[(idx-source_seq_len):(idx+target_seq_len) ,:]
      #print( data_sel.shape )

      encoder_inputs[i, :, :]  = data_sel[0:source_seq_len-1, :]
      decoder_inputs[i, :, :]  = data_sel[source_seq_len-1:(source_seq_len+target_seq_len-1), :]
      decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]


    return encoder_inputs, decoder_inputs, decoder_outputs
