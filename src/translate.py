
"""Simple code for training an RNN for motion prediction.

See the paper at http://arxiv.org/abs/1409.3215 for more information
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import h5py
import socket

import cPickle

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model

# Learning
tf.app.flags.DEFINE_float("learning_rate", .005, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 1, "Learning rate is multiplied by this much. 1 means no decay.")
tf.app.flags.DEFINE_integer("learning_rate_step", 1000, "Every this many steps, do decay.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("iterations", 20000, "Iterations to train for.")
# Architecture
tf.app.flags.DEFINE_string("architecture", "tied", "Seq2seq architecture to use: [basic, tied].")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("seq_length_in", 50, "Length of sequences to feed into the encoder")
#tf.app.flags.DEFINE_integer("seq_length_out", 25, "Length of sequences that the decoder has to predict")
tf.app.flags.DEFINE_integer("seq_length_out", 10, "Length of sequences that the decoder has to predict")
tf.app.flags.DEFINE_boolean("omit_one_hot", False, "Whether to remove one-hot encoding from the data")
tf.app.flags.DEFINE_boolean("residual_velocities", False, "Add a residual connection that effectively models velocities")
tf.app.flags.DEFINE_float("loss_velocities_weight", 0.0, "Weight to give to residual velocities")
# Directories
tf.app.flags.DEFINE_string("data_dir", "./data/h3.6m/dataset", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./log/", "Training directory.")

tf.app.flags.DEFINE_string("action","all", "The action to train on. all means all the actions, all_periodic means walking, eating and smoking")
tf.app.flags.DEFINE_string("loss_to_use","self_fed", "The type of loss to use, supervised or self_fed")
tf.app.flags.DEFINE_boolean("space_encoder", False, "Whether to use an encoder in space")


tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("test_every", 100, "How often to compute error on the test set.")
tf.app.flags.DEFINE_integer("save_every", 1000, "How often to compute error on the test set.")
tf.app.flags.DEFINE_boolean("sample", False, "Set to True for sampling.")
tf.app.flags.DEFINE_boolean("use_cpu", False, "Whether to use the CPU")
tf.app.flags.DEFINE_integer("try_to_load", 0, "Try to load a previous checkpoint.")

FLAGS = tf.app.flags.FLAGS

train_dir = os.path.join( FLAGS.train_dir, FLAGS.action,
  'out_{0}'.format(FLAGS.seq_length_out),
  FLAGS.architecture,
  FLAGS.loss_to_use,
  'omit_one_hot' if FLAGS.omit_one_hot else 'one_hot',
  'depth_{0}'.format(FLAGS.num_layers),
  'size_{0}'.format(FLAGS.size),
  'lr_{0}'.format(FLAGS.learning_rate),
  'residual_vel' if FLAGS.residual_velocities else 'not_residual_vel',
  'space_encoder' if FLAGS.space_encoder else 'not_space_encoder')

summaries_dir = os.path.join( train_dir, "log" ) # Directory for TB summaries

def create_model(session, actions, forward_only, sampling=False):
  """Create translation model and initialize or load parameters in session."""

  model = seq2seq_model.Seq2SeqModel(
      FLAGS.architecture,
      FLAGS.seq_length_in if not sampling else 50,
      FLAGS.seq_length_out if not sampling else 100,
      FLAGS.size, # hidden layer size
      FLAGS.num_layers,
      FLAGS.max_gradient_norm,
      FLAGS.batch_size,
      FLAGS.learning_rate,
      FLAGS.learning_rate_decay_factor,
      summaries_dir,
      FLAGS.loss_to_use if not sampling else "self_fed",
      FLAGS.space_encoder,
      len( actions ),
      not FLAGS.omit_one_hot,
      FLAGS.residual_velocities,
      FLAGS.loss_velocities_weight,
      forward_only=forward_only,
      dtype=tf.float32)

  if FLAGS.try_to_load <= 0:
    print("Creating model with fresh parameters.")
    session.run(tf.global_variables_initializer())
    return model

  ckpt_path = os.path.join( train_dir, "checkpoint-{0}".format(FLAGS.try_to_load) )
  print( ckpt_path )
  if os.path.exists( ckpt_path ):
    print("Reading model parameters from %s" % ckpt_path)
    model.saver.restore(session, ckpt_path )
  else:
    print("Could not find checkpoint. Aborting.")
    raise( ValueError, "Checkpoint does not seem to exist" )

  return model


def train():
  """Train a seq2seq model on human motion"""

  actions = define_actions( FLAGS.action )

  number_of_actions = len( actions )

  train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
    actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:

    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    forward_only = False
    model = create_model( sess, actions, forward_only )
    model.train_writer.add_graph( sess.graph )
    print( "Model created" )

    # === Read and denormalize the gt with Ashesh's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    ashesh_gts_euler = get_ashesh_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot )

    #=== This is the training loop ===
    step_time, loss, val_loss = 0.0, 0.0, 0.0
    current_step = 0 if FLAGS.try_to_load <= 0 else FLAGS.try_to_load + 1
    previous_losses = []

    step_time, loss = 0, 0

    for _ in xrange( FLAGS.iterations ):

      start_time = time.time()

      # === Training step ===
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( train_set, not FLAGS.omit_one_hot )
      _, step_loss, loss_summary, lr_summary = model.step( sess, encoder_inputs, decoder_inputs, decoder_outputs, False )
      model.train_writer.add_summary( loss_summary, current_step )
      model.train_writer.add_summary( lr_summary, current_step )

      if current_step % 10 == 0:
        print("step {0:04d}; step_loss: {1:.4f}".format(current_step, step_loss ))

      step_time += (time.time() - start_time) / FLAGS.test_every
      loss += step_loss / FLAGS.test_every
      current_step += 1

      # === step decay ===
      if current_step % FLAGS.learning_rate_step == 0:
        sess.run(model.learning_rate_decay_op)

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.test_every == 0:

        # === Validation with randomly chosen seeds ===
        forward_only = True

        encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch( test_set, not FLAGS.omit_one_hot )
        step_loss, loss_summary = model.step(sess,
            encoder_inputs, decoder_inputs, decoder_outputs, forward_only)
        val_loss = step_loss # Loss book-keeping

        model.test_writer.add_summary(loss_summary, current_step)

        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 1000]:
          print(" {0:5d} |".format(ms), end="")
        print()

        # === Validation with Ashesh's seeds ===
        for action in actions:

          if action not in ['walking', 'eating', 'smoking', 'discussion']:
            continue

          encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_ashesh( test_set, action )
          ashesh_loss, ashesh_poses, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                   decoder_outputs, True, True)

          # denormalizes too
          ashesh_pred_expmap = data_utils.revert_output_format( ashesh_poses,
            data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

          R0 = np.eye(3)
          T0 = np.zeros(3)

          # Save the errors here
          mean_errors = np.zeros( (len(ashesh_pred_expmap), ashesh_pred_expmap[0].shape[0]) )

          for i in np.arange(8):
            # TODO see if we need this reverse coordinate thing
            #ashesh_pred_expmap, _, _ = data_utils.revert_coordinate_space( ashesh_pred_expmap[0], R0, T0 )
            eulerchannels_pred = ashesh_pred_expmap[i]

            for j in np.arange( eulerchannels_pred.shape[0] ):
              for k in np.arange(3,97,3):
                eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
                  data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

            eulerchannels_pred[:,0:6] = 0
            idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

            euc_error = np.power( ashesh_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
            euc_error = np.sum(euc_error, 1)
            euc_error = np.sqrt( euc_error )
            mean_errors[i,:] = euc_error


          mean_mean_errors = np.mean( mean_errors, 0 )
          # print( action, mean_mean_errors )

          print("{0: <16} |".format(action), end="")
          for ms in [1,3,7,9,13,24]:
            if FLAGS.seq_length_out >= ms+1:
              print(" {0:.3f} |".format( mean_mean_errors[ms] ), end="")
            else:
              print("   n/a |", end="")
          print()

          #print( action, mean_mean_errors[[1,3,7,13,24]] )
          # Simply set the errors to log in TB

          if action == "walking":
            summaries = sess.run(
              [model.walking_err80_summary,
               model.walking_err160_summary,
               model.walking_err320_summary,
               model.walking_err400_summary,
               model.walking_err560_summary,
               model.walking_err1000_summary],
              {model.walking_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.walking_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.walking_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.walking_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.walking_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.walking_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "eating":
            summaries = sess.run(
              [model.eating_err80_summary,
               model.eating_err160_summary,
               model.eating_err320_summary,
               model.eating_err400_summary,
               model.eating_err560_summary,
               model.eating_err1000_summary],
              {model.eating_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.eating_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.eating_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.eating_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.eating_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.eating_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "smoking":
            summaries = sess.run(
              [model.smoking_err80_summary,
               model.smoking_err160_summary,
               model.smoking_err320_summary,
               model.smoking_err400_summary,
               model.smoking_err560_summary,
               model.smoking_err1000_summary],
              {model.smoking_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.smoking_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.smoking_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.smoking_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.smoking_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.smoking_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "discussion":
            summaries = sess.run(
              [model.discussion_err80_summary,
               model.discussion_err160_summary,
               model.discussion_err320_summary,
               model.discussion_err400_summary,
               model.discussion_err560_summary,
               model.discussion_err1000_summary],
              {model.discussion_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.discussion_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.discussion_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.discussion_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.discussion_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.discussion_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "directions":
            summaries = sess.run(
              [model.directions_err80_summary,
               model.directions_err160_summary,
               model.directions_err320_summary,
               model.directions_err400_summary,
               model.directions_err560_summary,
               model.directions_err1000_summary],
              {model.directions_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.directions_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.directions_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.directions_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.directions_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.directions_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "greeting":
            summaries = sess.run(
              [model.greeting_err80_summary,
               model.greeting_err160_summary,
               model.greeting_err320_summary,
               model.greeting_err400_summary,
               model.greeting_err560_summary,
               model.greeting_err1000_summary],
              {model.greeting_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.greeting_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.greeting_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.greeting_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.greeting_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.greeting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "phoning":
            summaries = sess.run(
              [model.phoning_err80_summary,
               model.phoning_err160_summary,
               model.phoning_err320_summary,
               model.phoning_err400_summary,
               model.phoning_err560_summary,
               model.phoning_err1000_summary],
              {model.phoning_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.phoning_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.phoning_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.phoning_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.phoning_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.phoning_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "posing":
            summaries = sess.run(
              [model.posing_err80_summary,
               model.posing_err160_summary,
               model.posing_err320_summary,
               model.posing_err400_summary,
               model.posing_err560_summary,
               model.posing_err1000_summary],
              {model.posing_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.posing_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.posing_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.posing_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.posing_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.posing_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "purchases":
            summaries = sess.run(
              [model.purchases_err80_summary,
               model.purchases_err160_summary,
               model.purchases_err320_summary,
               model.purchases_err400_summary,
               model.purchases_err560_summary,
               model.purchases_err1000_summary],
              {model.purchases_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.purchases_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.purchases_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.purchases_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.purchases_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.purchases_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "sitting":
            summaries = sess.run(
              [model.sitting_err80_summary,
               model.sitting_err160_summary,
               model.sitting_err320_summary,
               model.sitting_err400_summary,
               model.sitting_err560_summary,
               model.sitting_err1000_summary],
              {model.sitting_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.sitting_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.sitting_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.sitting_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.sitting_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.sitting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "sittingdown":
            summaries = sess.run(
              [model.sittingdown_err80_summary,
               model.sittingdown_err160_summary,
               model.sittingdown_err320_summary,
               model.sittingdown_err400_summary,
               model.sittingdown_err560_summary,
               model.sittingdown_err1000_summary],
              {model.sittingdown_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.sittingdown_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.sittingdown_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.sittingdown_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.sittingdown_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.sittingdown_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "takingphoto":
            summaries = sess.run(
              [model.takingphoto_err80_summary,
               model.takingphoto_err160_summary,
               model.takingphoto_err320_summary,
               model.takingphoto_err400_summary,
               model.takingphoto_err560_summary,
               model.takingphoto_err1000_summary],
              {model.takingphoto_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.takingphoto_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.takingphoto_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.takingphoto_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.takingphoto_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.takingphoto_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "waiting":
            summaries = sess.run(
              [model.waiting_err80_summary,
               model.waiting_err160_summary,
               model.waiting_err320_summary,
               model.waiting_err400_summary,
               model.waiting_err560_summary,
               model.waiting_err1000_summary],
              {model.waiting_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.waiting_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.waiting_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.waiting_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.waiting_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.waiting_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "walkingdog":
            summaries = sess.run(
              [model.walkingdog_err80_summary,
               model.walkingdog_err160_summary,
               model.walkingdog_err320_summary,
               model.walkingdog_err400_summary,
               model.walkingdog_err560_summary,
               model.walkingdog_err1000_summary],
              {model.walkingdog_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.walkingdog_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.walkingdog_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.walkingdog_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.walkingdog_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.walkingdog_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})
          elif action == "walkingtogether":
            summaries = sess.run(
              [model.walkingtogether_err80_summary,
               model.walkingtogether_err160_summary,
               model.walkingtogether_err320_summary,
               model.walkingtogether_err400_summary,
               model.walkingtogether_err560_summary,
               model.walkingtogether_err1000_summary],
              {model.walkingtogether_err80: mean_mean_errors[1] if FLAGS.seq_length_out >= 2 else None,
               model.walkingtogether_err160: mean_mean_errors[3] if FLAGS.seq_length_out >= 4 else None,
               model.walkingtogether_err320: mean_mean_errors[7] if FLAGS.seq_length_out >= 8 else None,
               model.walkingtogether_err400: mean_mean_errors[9] if FLAGS.seq_length_out >= 10 else None,
               model.walkingtogether_err560: mean_mean_errors[13] if FLAGS.seq_length_out >= 14 else None,
               model.walkingtogether_err1000: mean_mean_errors[24] if FLAGS.seq_length_out >= 25 else None})

          for i in np.arange(len( summaries )):
            model.test_writer.add_summary(summaries[i], current_step)


        print()
        print("==========================\n"
              "Global step:         %d\n"
              "Learning rate:       %.4f\n"
              "Step-time (ms):      %.4f\n"
              "Train loss avg:      %.4f\n"
              "--------------------------\n"
              "Val loss:            %.4f\n"
              "Ashesh loss:         %.4f\n"
              "==========================" % (model.global_step.eval(),
              model.learning_rate.eval(), step_time*1000, loss,
              val_loss, ashesh_loss))
        print()

        # Decrease learning rate if no improvement was seen over last 3 times.
        # if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
        #   sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)

        # Save the model
        if current_step % FLAGS.save_every == 0:
          print( "Saving the model..." ); start_time = time.time()
          model.saver.save(sess, os.path.join(train_dir, 'checkpoint'), global_step=current_step )
          print( "done in {0:.2f} ms".format( (time.time() - start_time)*1000) )

        # Reset global time and loss
        step_time, loss = 0, 0

        sys.stdout.flush()


def get_ashesh_gts( actions, model, test_set, data_mean, data_std, dim_to_ignore, one_hot ):
  """
  Get the ground truths for asheh's seeds

  Args:
    actions: a list of actions to get ground truths for

  Returns:
    ashesh_gts_euler: a dictionary where the keys are actions, and the values
      are the ground_truth, denormalized expected outputs of asheh's seeds.
  """

  ashesh_gts_euler = {}

  for action in actions:

    #if action not in ["walking", "eating", "smoking", "discussion"]:
    #  continue

    ashesh_gt_euler = []
    _, _, ashesh_expmap = model.get_batch_ashesh( test_set, action )

    for i in np.arange( ashesh_expmap.shape[0] ):
      denormed = data_utils.unNormalizeData(ashesh_expmap[i,:,:], data_mean, data_std, dim_to_ignore, actions, one_hot )

      for j in np.arange( denormed.shape[0] ):
        for k in np.arange(3,97,3):
          denormed[j,k:k+3] = data_utils.rotmat2euler( data_utils.expmap2rotmat( denormed[j,k:k+3] ))

      ashesh_gt_euler.append( denormed );

    # Put back in the dictionary
    ashesh_gts_euler[action] = ashesh_gt_euler

  return ashesh_gts_euler


def sample():
  """Sample predictions for ashesh's seeds"""

  if FLAGS.try_to_load <= 0:
    raise( ValueError, "Must give an iteration to read parameters from")

  actions = define_actions( FLAGS.action )

  # Limit TF to take a fraction of the GPU memory
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0} if FLAGS.use_cpu else {"GPU": 1}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:
    # === Create the model ===
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
    forward_only = False
    sampling     = True
    model = create_model(sess, actions, forward_only, sampling)
    print("Model created")

    # Load all the data
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = read_all_data(
      actions, FLAGS.seq_length_in, FLAGS.seq_length_out, FLAGS.data_dir, not FLAGS.omit_one_hot )

    # === Read and denormalize the gt with Ashesh's seeds, as we'll need them
    # many times for evaluation in Euler Angles ===
    ashesh_gts_euler = get_ashesh_gts( actions, model, test_set, data_mean,
                              data_std, dim_to_ignore, not FLAGS.omit_one_hot )

    try:
      os.remove( os.path.join(train_dir, 'samples_full.h5') )
    except OSError:
      pass

    for action in actions:

      if action not in ["walking", "eating", "smoking", "discussion"]:
        continue

      # Make prediction with Ashesh' seeds
      encoder_inputs, decoder_inputs, decoder_outputs = model.get_batch_ashesh( test_set, action )
      ashesh_loss, ashesh_poses, _ = model.step(sess, encoder_inputs, decoder_inputs, decoder_outputs, 1.0, 0.0, True, True)

      # denormalizes too
      ashesh_pred_expmap = data_utils.revert_output_format( ashesh_poses, data_mean, data_std, dim_to_ignore, actions, not FLAGS.omit_one_hot )

      # Save the sample
      with h5py.File( os.path.join(train_dir, 'samples_full.h5'), 'a' ) as hf:
        for i in np.arange(8):
          eulerchannels_pred = ashesh_pred_expmap[i]
          node_name = 'seeds_expmap/{1}_{0}'.format(i, action)
          hf.create_dataset( node_name, data=eulerchannels_pred )

      # Compute and save the errors here
      mean_errors = np.zeros( (len(ashesh_pred_expmap), ashesh_pred_expmap[0].shape[0]) )

      for i in np.arange(8):
        # TODO see if we need this reverse coordinate thing
        #ashesh_pred_expmap, _, _ = data_utils.revert_coordinate_space( ashesh_pred_expmap[0], R0, T0 )
        eulerchannels_pred = ashesh_pred_expmap[i]

        for j in np.arange( eulerchannels_pred.shape[0] ):
          for k in np.arange(3,97,3):
            eulerchannels_pred[j,k:k+3] = data_utils.rotmat2euler(
              data_utils.expmap2rotmat( eulerchannels_pred[j,k:k+3] ))

        eulerchannels_pred[:,0:6] = 0
        idx_to_use = np.where( np.std( eulerchannels_pred, 0 ) > 1e-4 )[0]

        euc_error = np.power( ashesh_gts_euler[action][i][:,idx_to_use] - eulerchannels_pred[:,idx_to_use], 2)
        euc_error = np.sum(euc_error, 1)
        euc_error = np.sqrt( euc_error )
        mean_errors[i,:] = euc_error

      mean_mean_errors = np.mean( mean_errors, 0 )
      print( action )
      print( ','.join(map(str, mean_mean_errors.tolist() )) )


      with h5py.File( os.path.join(train_dir, 'samples_full.h5'), 'a' ) as hf:
        node_name = 'mean_{0}_error'.format( action )
        hf.create_dataset( node_name, data=mean_mean_errors )

  return

def define_actions( action ):

  if action in ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]:
    actions = [action]
  elif action == "all":
    actions = ["directions", "discussion", "eating", "greeting", "phoning",
                "posing", "purchases", "sitting", "sittingdown", "smoking",
                "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
  elif action == "all_ashesh":
    actions = ["walking", "eating", "smoking", "discussion"]
  elif action == "all_periodic":
    actions = ["walking", "eating", "smoking"]
  else:
    raise( ValueError, "Unrecognized action: %d" % action )

  return actions


def read_all_data( actions, seq_length_in, seq_length_out, data_dir, one_hot ):
  """Load data for training and normalizes it

  Input
    actions: a list of actions that we are dealing with
    seq_length_in: Number of frames to use in the burn-in sequence
    seq_length_out: Number of frames to use in the output sequence
    data_dir: Where to load the data from
    one_hot: whether to use one-hot encoding

  Output
    train_set:
    test_set:
    data_mean:
    data_std:
    dim_to_ignore:
    dim_to_use:
  """

  # === Read training data ===
  print ("Reading training data (seq_len_in: {0}, seq_len_out {1}).".format(
           seq_length_in, seq_length_out))
  train_set, complete_train = data_utils.load_data( data_dir, [1, 6, 7, 8, 9, 11], actions, one_hot )
  test_set, complete_test   = data_utils.load_data( data_dir, [5], actions, one_hot )
  # Compute normalization statistics.
  data_mean, data_std, dim_to_ignore, dim_to_use, _ = data_utils.normalization_stats(complete_train)
  # Normalize the data (substract mean, divide by stdev).
  train_set = data_utils.normalize_data( train_set, data_mean, data_std, dim_to_use, actions, one_hot )
  test_set  = data_utils.normalize_data( test_set,  data_mean, data_std, dim_to_use, actions, one_hot )
  print("done reading data.")

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def main(_):
  if FLAGS.sample:
    sample()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()
