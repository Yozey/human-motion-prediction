
import tensorflow as tf
import numpy as np


class TestModel():
  # Limit TF to take a fraction of the GPU memory

  def __init__(self):

    self.batch_size = 8
    self.seq_length = 12
    self.input_size = 54

    # Create a simple model
    with tf.name_scope("inputs"):

      dec_out = tf.placeholder(tf.float32, shape=[None, self.seq_length, self.input_size], name="dec_out")
      self.decoder_outputs = dec_out

      # Do all the splitting
      dec_out = tf.transpose(dec_out, [1, 0, 2])
      dec_out = tf.reshape(dec_out, [-1, self.input_size])
      dec_out = tf.split(0, self.seq_length, dec_out)

      # Concatenate back
      #dec_out = tf.add( dec_out, 1)

      dec_out = tf.concat( 0, dec_out )
      dec_out = tf.reshape( dec_out, [self.seq_length, -1, self.input_size] )
      dec_out = tf.transpose( dec_out, [1, 0, 2] )

      dec_out = tf.sub(dec_out[:,1:,], dec_out[:,:-1,])


      #filter = tf.Variable(np.array([-1.0, 1.0]).reshape(1, 2, 1 ), dtype=tf.float32)
      #dec_out = tf.nn.conv2d( dec_out, filter, strides=[1,1,1], padding="VALID")
      # dec_out = past

      self.reshaped = dec_out

      self.sol = tf.reduce_mean( dec_out )

  def run(self, sess):

    # Run it
    output_feed = [ self.sol, self.reshaped ]

    print( self.batch_size, self.seq_length, self.input_size )

    inp = np.random.rand( self.batch_size, self.seq_length, self.input_size).astype( np.float32 )
    input_feed  = {self.decoder_outputs: inp}

    outs = sess.run( output_feed, input_feed )

    print( outs[1] )
    print( outs[1].shape )
    print( outs[1] == (inp[:,1:,] - inp[:,:-1,]) )


def main(_):

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
  device_count = {"GPU": 0}

  with tf.Session(config=tf.ConfigProto( gpu_options=gpu_options, device_count = device_count )) as sess:
    model = TestModel()
    model.run( sess )


if __name__ == "__main__":
  # I'm assuming this calls main(_)?
  tf.app.run()
