
# Super-simple baselines for short term human motion prediction

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import translate
import data_utils
import seq2seq_model

class Object(object):
    pass

def running_average( ashesh_data, actions, k ):
  """
  Compute the error if we take the average of the last k frames.

  Args:
    ashesh_data

    k: number of frames to average

  Returns:
    errs: a dictionary where, for each action, we have a 100-long list with the
          error at each point in time.
  """

  # if k == 1:
    # return last_frame_constant( ashesh_data, actions )

  # Get how many batches we have
  enc_in, dec_in, dec_out = ashesh_data[ actions[0] ]

  n_sequences = len( enc_in )
  seq_length_out = dec_out[0].shape[0]

  errs = dict()

  for action in actions:

    # Make space for the error
    errs[ action ] = np.zeros( (n_sequences, seq_length_out) )

    # Get the lists for this action
    enc_in, dec_in, dec_out = ashesh_data[action]

    for i in np.arange( n_sequences ):

      n, d = dec_out[i].shape

      # The last frame
      last_frame = dec_in[i][0, :]
      last_frame[0:6] = 0

      # Get the last k-1 frames
      if k > 1:
        last_k = enc_in[i][(-k+1):, :]
        # last_k = enc_in[i][-k:, :]

        assert( last_k.shape[0] == (k-1) )

        # Merge and average them
        avg = np.mean( np.vstack( (last_k, last_frame) ), 0 )
      else:
        avg = last_frame

      dec_out[i][:, 0:6] = 0
      idx_to_use = np.where( np.std( dec_out[i], 0 ) > 1e-4 )[0]

      ee = np.power( dec_out[i][:,idx_to_use] - avg[idx_to_use], 2 )
      ee = np.sum( ee, 1 )
      ee = np.sqrt( ee )
      errs[ action ][i, :] = ee

    errs[action] = np.mean( errs[action], 0 )

  return errs

def denormalize_and_convert_to_euler( data, data_mean, data_std, dim_to_ignore, actions, one_hot ):
  """
  Args:
    data
    data_mean
    ...

  Returns:
    all_denormed: a list with nbatch entries. Each entry is an n-by-d matrix
                  that corresponds to a denormalized sequence in Euler angles
  """

  all_denormed = []

  for i in np.arange( data.shape[0] ):
    denormed = data_utils.unNormalizeData(data[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

    for j in np.arange( denormed.shape[0] ):
      for k in np.arange(3,97,3):
        denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

    all_denormed.append( denormed )

  return all_denormed


def main():

  actions = ["discussion", "eating", "smoking", "walking"]

  # actions = ["directions", "discussion", "eating", "greeting", "phoning",
  #             "posing", "purchases", "sitting", "sittingdown", "smoking",
  #             "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

  one_hot = False
  FLAGS = Object()
  FLAGS.data_dir = "./data/h3.6m/dataset"
  FLAGS.architecture = "tied"
  FLAGS.seq_length_in = 50
  FLAGS.seq_length_out = 100
  FLAGS.num_layers = 1
  FLAGS.size = 128
  FLAGS.max_gradient_norm = 5
  FLAGS.batch_size = 8
  FLAGS.learning_rate = 0.005
  FLAGS.learning_rate_decay_factor = 1
  summaries_dir = "./log/"
  FLAGS.loss_to_use = "self_fed"
  FLAGS.residual_rnn = False
  FLAGS.space_encoder = False
  FLAGS.omit_one_hot = True,
  FLAGS.use_lstm = False,
  FLAGS.residual_velocities = False,
  forward_only = False,
  dtype = tf.float32

  # WE do not need a GPU for this
  with tf.Session(config=tf.ConfigProto( device_count = {"GPU": 0} )) as sess:

    model = seq2seq_model.Seq2SeqModel(
        FLAGS.architecture,
        FLAGS.seq_length_in,
        FLAGS.seq_length_out,
        FLAGS.size, # hidden layer size
        FLAGS.num_layers,
        FLAGS.max_gradient_norm,
        FLAGS.batch_size,
        FLAGS.learning_rate,
        FLAGS.learning_rate_decay_factor,
        summaries_dir,
        FLAGS.loss_to_use,
        FLAGS.residual_rnn,
        FLAGS.space_encoder,
        len( actions ),
        not FLAGS.omit_one_hot,
        FLAGS.use_lstm,
        FLAGS.residual_velocities,
        forward_only=forward_only,
        dtype=dtype)

    # Load the data
    _, test_set, data_mean, data_std, dim_to_ignore, dim_to_use =  translate.read_all_data(
      actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

    # Get all the data for Ashesh's seeds, and convert it to euler angles
    ashesh_data = {}

    np.random.seed(4321)
    for action in actions:
      enc_in, dec_in, dec_out = model.get_batch_ashesh( test_set, action )

      enc_in  = denormalize_and_convert_to_euler(
        enc_in, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )
      dec_in  = denormalize_and_convert_to_euler(
        dec_in, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )
      dec_out = denormalize_and_convert_to_euler(
        dec_out, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

      ashesh_data[action] = (enc_in, dec_in, dec_out)

    # Compute baseline errors
    errs_constant_frame = running_average( ashesh_data, actions, 1 )
    running_average_2   = running_average( ashesh_data, actions, 2 )
    running_average_4   = running_average( ashesh_data, actions, 4 )

    print()
    print("=== Zero-velocity (running avg. 1) ===")
    print("{0: <16} | {1:4d} | {2:4d} | {3:4d} | {4:4d}".format("milliseconds", 80, 160, 380, 400))
    for action in actions:
      print("{0: <16} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f}".format( action,
            errs_constant_frame[action][1], errs_constant_frame[action][3],
            errs_constant_frame[action][7], errs_constant_frame[action][9] ))

    print()
    print("=== Runnning avg. 2 ===")
    print("{0: <16} | {1:4d} | {2:4d} | {3:4d} | {4:4d}".format("milliseconds", 80, 160, 380, 400))
    for action in actions:
      print("{0: <16} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f}".format( action,
            running_average_2[action][1], running_average_2[action][3],
            running_average_2[action][7], running_average_2[action][9] ))

    print()
    print("=== Runnning avg. 4 ===")
    print("{0: <16} | {1:4d} | {2:4d} | {3:4d} | {4:4d}".format("milliseconds", 80, 160, 380, 400))
    for action in actions:
      print("{0: <16} | {1:.2f} | {2:.2f} | {3:.2f} | {4:.2f}".format( action,
            running_average_4[action][1], running_average_4[action][3],
            running_average_4[action][7], running_average_4[action][9] ))


if __name__ == "__main__":
  main()
