
# A bunch of simple baselines that reviewers probably want to see for short term prediction

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

def last_frame_constant( ashesh_data, actions ):
  """
  Compute the error if we simply take the last frame as a constant.

  Args:
    ashesh_data

  Returns:
    errs: a dictionary where, for each action, we have a 100-long list with the
          error at each point in time.
  """

  # Get how many batches we have
  enc_in, dec_in, dec_out = ashesh_data[ actions[0] ]

  n_sequences = len( enc_in )
  seq_length_out = dec_out[0].shape[0]

  print( n_sequences )
  print( seq_length_out )

  errs = dict()

  for action in actions:

    # Make space for the error
    errs[ action ] = np.zeros( (n_sequences, seq_length_out) )

    # Get the lists for this action
    enc_in, dec_in, dec_out = ashesh_data[action]

    for i in np.arange( n_sequences ):

      n, d = dec_out[i].shape

      first_frame = dec_in[i][0, :]
      first_frame[0:6] = 0

      dec_out[i][:, 0:6] = 0

      idx_to_use = np.where( np.std( dec_out[i], 0 ) > 1e-4 )[0]

      ee = np.power( dec_out[i][:,idx_to_use] - first_frame[idx_to_use], 2 )
      ee = np.sum( ee, 1 )
      ee = np.sqrt( ee )
      errs[ action ][i, :] = ee

    errs[action] = np.mean( errs[action], 0 )
    print( action )
    # print( errs[action] )
    # print( ",".join(map(str,errs[action].tolist())) )
    # for idx in [1,3,7,9]: # 80, 160, 380, 400
    #   print( "e@{0}: {1:.2f}".format( (idx+1)*40, errs[action][idx] ) )

    #for idx in [1,3,7,9]: # 80, 160, 380, 400
    print( "{0:.2f} & {1:.2f} & {2:.2f} & {3:.2f} &".format( errs[action][1], errs[action][3], errs[action][7], errs[action][9] ) )

  return errs

def last_vel_constant( ashesh_data, actions ):
  """
  Compute the error if we simply take the last velocity as constant.

  Args:
    ashesh_data

  Returns:
    errs: a dictionary where, for each action, we have a 100-long list with the
          error at each point in time.
  """

  # Get how many batches we have
  enc_in, dec_in, dec_out = ashesh_data[ actions[0] ]

  n_sequences = len( enc_in )
  seq_length_out = dec_out[0].shape[0]

  print( n_sequences )
  print( seq_length_out )

  errs = dict()

  for action in actions:

    # Make space for the error
    errs[ action ] = np.zeros( (n_sequences, seq_length_out) )

    # Get the lists for this action
    enc_in, dec_in, dec_out = ashesh_data[action]

    for i in np.arange( n_sequences ):

      n, d = dec_out[i].shape

      first_frame_decoder = dec_in[i][0, :]
      first_frame_decoder[0:6] = 0

      last_frame_encoder = enc_in[i][-1, :]
      last_frame_encoder[0:6] = 0

      # Compute the velocity
      velocity = first_frame_decoder - last_frame_encoder

      # Keep the prediction here
      prediction = np.zeros_like( dec_out[i] )
      prediction[0, :] = first_frame_decoder + velocity

      # Just keep adding the velocity
      for j in np.arange( 1, n ):
        prediction[i, :] = prediction[i-1,:] + velocity

      # Compute the loss
      dec_out[i][:, 0:6] = 0
      idx_to_use = np.where( np.std( dec_out[i], 0 ) > 1e-4 )[0]

      ee = np.power( dec_out[i][:,idx_to_use] - prediction[:, idx_to_use], 2 )
      ee = np.sum( ee, 1 )
      ee = np.sqrt( ee )
      errs[ action ][i, :] = ee

    errs[action] = np.mean( errs[action], 0 )
    print( action )
    print( errs[action] )
    print( ",".join(map(str,errs[action].tolist())) )

  return errs

def running_average( ashesh_data, actions, k ):
  """
  Compute the error if we simply take the last frame as a constant.

  Args:
    ashesh_data

    k: number of frames to average

  Returns:
    errs: a dictionary where, for each action, we have a 100-long list with the
          error at each point in time.
  """

  if k == 1:
    return last_frame_constant( ashesh_data, actions )

  # Get how many batches we have
  enc_in, dec_in, dec_out = ashesh_data[ actions[0] ]

  n_sequences = len( enc_in )
  seq_length_out = dec_out[0].shape[0]

  print( n_sequences )
  print( seq_length_out )

  errs = dict()

  for action in actions:

    # Make space for the error
    errs[ action ] = np.zeros( (n_sequences, seq_length_out) )

    # Get the lists for this action
    enc_in, dec_in, dec_out = ashesh_data[action]

    for i in np.arange( n_sequences ):

      n, d = dec_out[i].shape

      first_frame = dec_in[i][0, :]
      first_frame[0:6] = 0

      # Get the last k frames
      last_k = enc_in[i][(-k+1):, :]
      assert( last_k.shape[0] == (k-1) )

      # Merge them an average them
      avg = np.mean( np.vstack( (last_k, first_frame) ), 0 )

      dec_out[i][:, 0:6] = 0
      idx_to_use = np.where( np.std( dec_out[i], 0 ) > 1e-4 )[0]

      ee = np.power( dec_out[i][:,idx_to_use] - avg[idx_to_use], 2 )
      ee = np.sum( ee, 1 )
      ee = np.sqrt( ee )
      errs[ action ][i, :] = ee

    errs[action] = np.mean( errs[action], 0 )
    print( action )
    print( errs[action] )
    print( ",".join(map(str,errs[action].tolist())) )

  return errs

def main():

  # actions = ["walking", "eating", "smoking", "discussion"]
  # actions = ["directions"]
  # actions = ["eating"]
  actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
  one_hot = False

  FLAGS = Object()
  FLAGS.data_dir = "../../rnn/data/ashesh"
  FLAGS.architecture = "tied"
  FLAGS.seq_length_in = 50
  FLAGS.seq_length_out = 100
  FLAGS.num_layers = 1
  FLAGS.size = 128
  FLAGS.max_gradient_norm = 5
  FLAGS.batch_size = 8
  FLAGS.learning_rate = 0.005
  FLAGS.learning_rate_decay_factor = 1
  summaries_dir = "/home/julieta/Desktop/scratch/log/"
  FLAGS.loss_to_use = "self_fed"
  FLAGS.residual_rnn = False
  FLAGS.space_encoder = False
  # len( actions ),
  FLAGS.omit_one_hot = True,
  FLAGS.use_lstm = False,
  FLAGS.residual_velocities = False,
  forward_only = False,
  dtype = tf.float32

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:

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

    # Error with the last frame constant
    errs_constant_frame = last_frame_constant( ashesh_data, actions )
    # errs_constant_vel = last_vel_constant( ashesh_data, actions )
    # running_average_2 = running_average( ashesh_data, actions, 2 )
    # running_average_4 = running_average( ashesh_data, actions, 4 )

  print("done")


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

if __name__ == "__main__":
  main()
