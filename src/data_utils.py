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

"""Functions that help with data processing for human3.6m"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy

def rotmat2euler( R ):
  """Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1
  """
  if R[0,2] == 1 or R[0,2] == -1:
    # special case
    E3   = 0 # set arbitrarily
    dlta = np.arctan2( R[0,1], R[0,2] );

    if R[0,2] == -1:
      E2 = np.pi/2;
      E1 = E3 + dlta;
    else:
      E2 = -np.pi/2;
      E1 = -E3 + dlta;

  else:
    E2 = -np.arcsin( R[0,2] )
    E1 = np.arctan2( R[1,2]/np.cos(E2), R[2,2]/np.cos(E2) )
    E3 = np.arctan2( R[0,1]/np.cos(E2), R[0,0]/np.cos(E2) )

  eul = np.array([E1, E2, E3]);

  return eul


def quat2expmap(q):
  """Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1
  """
  if (np.abs(np.linalg.norm(q)-1)>1e-3):
    raise(ValueError, "quat2expmap: input quaternion is not norm 1")

  sinhalftheta = np.linalg.norm(q[1:])
  coshalftheta = q[0]

  r0    = np.divide( q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
  theta = 2 * np.arctan2( sinhalftheta, coshalftheta )
  theta = np.mod( theta + 2*np.pi, 2*np.pi )

  if theta > np.pi:
    theta =  2 * np.pi - theta
    r0    = -r0

  r = r0 * theta

  return r

def rotmat2quat(R):
  """Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4
  """
  d = R - R.T;

  r = np.zeros(3)
  r[0] = -d[1,2]
  r[1] =  d[0,2]
  r[2] = -d[0,1]
  sintheta = np.linalg.norm(r) / 2;
  r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps );

  costheta = (np.trace(R)-1) / 2;

  theta = np.arctan2( sintheta, costheta );

  q      = np.zeros(4)
  q[0]   = np.cos(theta/2)
  q[1:] = r0*np.sin(theta/2)
  return q

def rotmat2expmap(R):
  return quat2expmap( rotmat2quat(R) );

def expmap2rotmat(r):
  """Matlab port to python for evaluation purposes
  I think this is called Rodrigues' formula
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
  """
  theta = np.linalg.norm( r )
  r0  = np.divide( r, theta + np.finfo(np.float32).eps )
  #r0  = np.divide( r, theta + np.finfo(np.float64).eps )
  r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3,3)
  r0x = r0x - r0x.T
  R = np.eye(3,3) + np.sin(theta)*r0x + (1-np.cos(theta))*(r0x).dot(r0x);

  return R

def revert_coordinate_space( channels_self, R0, T0 ):
  """ Matlab port to python for evaluation purposes
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m#L6
  """
  channels_reconstruct = channels_self
  R_prev = R0
  T_prev = T0
  #rootRotInd = 4:6;
  rootRotInd = np.arange(3,6)

  #for ii = 1:size(channels_self,1)
  for ii in np.arange( channels_self.shape[0] ):

    R_diff = expmap2rotmat( channels_self[ii, rootRotInd] )

    R = R_diff.dot( R_prev )
    channels_reconstruct[ii,rootRotInd] = rotmat2expmap(R)
    T = T_prev + ( np.linalg.inv(R_prev).dot((channels_self[ii,0:3]).T) ).T
    channels_reconstruct[ii,0:3] = T
    T_prev = T
    R_prev = R

  return channels_reconstruct, R_prev, T_prev

def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot ):
  """Borrowed from Ashesh. Reads a csv and returns a float matrix.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12
  """
  T = normalizedData.shape[0]
  D = data_mean.shape[0]

  origData = np.zeros((T, D), dtype=np.float32)
  dimensions_to_use = []
  for i in range(D):
    if i in dimensions_to_ignore:
      continue
    dimensions_to_use.append(i)
  dimensions_to_use = np.array(dimensions_to_use)

  if one_hot:
    origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
  else:
    origData[:, dimensions_to_use] = normalizedData

  # if not len(dimensions_to_use) == normalizedData.shape[1]:
  #   raise(ValueError, "The lenght of the dimensions to use does not match "
  #                     " the lenght of the unnormalized data ({0} vs {1})".format(
  #                     len(dimensions_to_use), normalizedData.shape[1] ))

  # TODO this might be very inefficient? idk
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  origData = np.multiply(origData, stdMat) + meanMat
  return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """Convert the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization.

    Args:
      poses: The output from TF. A list with (seq_length) entries, each with a
      (batch_size, dim) output

    Returns:
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each batch
                 is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename):
    """Borrowed from Ashesh. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    return np.array(returnArray)


def load_data(path_to_dataset, subjects, actions, one_hot):
  """Borrowed from Ashesh. This is how he reads his own .txt data.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L270

  Inputs
    path_to_dataset: string. directory where the data resides
    subjects: list of numbers. The subjects to load
    actions: list of string. The actions to load
    one_hot: Whether to add a one-hot encoding to the data

  Output
    trainData:
    completeData:
  """
  nactions = len( actions )

  trainData = {}
  completeData = []
  for subj in subjects:
    for action_idx in np.arange(len(actions)):

      action = actions[ action_idx ]

      for subact in [1, 2]:  # subactions

        print("Reading S: {0}, action: {1}, subaction: {2}".format(subj, action, subact))

        filename = '{0}/S{1}/{2}_{3}.txt'.format( path_to_dataset, subj, action, subact)
        action_sequence = readCSVasFloat(filename)

        n, d = action_sequence.shape
        # odd_list = range(1, n, 2)
        even_list = range(0, n, 2)

        #trainData[(subj, action, subact)] = action_sequence
        #trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]
        #trainData[(subj, action, subact, 'odd')] = action_sequence[odd_list, :]

        if one_hot:
          # Add a one-hot encoding at the end of the representation
          the_sequence = np.zeros( (len(even_list), d + nactions), dtype=float )
          the_sequence[ :, 0:d ] = action_sequence[even_list, :]
          the_sequence[ :, d+action_idx ] = 1
          trainData[(subj, action, subact, 'even')] = the_sequence
        else:
          trainData[(subj, action, subact, 'even')] = action_sequence[even_list, :]


        if len(completeData) == 0:
          completeData = copy.deepcopy(action_sequence)
        else:
          completeData = np.append(completeData, action_sequence, axis=0)

  return trainData, completeData


def normalize_data( data, data_mean, data_std, dim_to_use, actions, one_hot ):

  data_out = {}
  nactions = len(actions)

  if not one_hot:
    # No one-hot encoding... no need to do anything special
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ]

  else:
    # FIXME hard-coding 99 dimensions for un-normalized human poses
    for key in data.keys():
      data_out[ key ] = np.divide( (data[key][:, 0:99] - data_mean), data_std )
      data_out[ key ] = data_out[ key ][ :, dim_to_use ]
      data_out[ key ] = np.hstack( (data_out[key], data[key][:,-nactions:]) )

  return data_out


def normalization_stats(completeData):
  """"Also borrowed for Ashesh. Computes mean, stdev and dimensions to ignore.
  https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33
  """
  data_mean = np.mean(completeData, axis=0)
  data_std  =  np.std(completeData, axis=0)

  dimensions_to_ignore = []
  dimensions_to_use    = []

  # if not full_skeleton:
  #   dimensions_to_ignore = [0,1,2,3,4,5]
  dimensions_to_ignore.extend( list(np.where(data_std < 1e-4)[0]) )
  dimensions_to_use.extend( list(np.where(data_std >= 1e-4)[0]) )

  data_std[dimensions_to_ignore] = 1.0
  #print(dimensions_to_ignore)
  new_idx = []
  count = 0
  for i in range(completeData.shape[1]):
    if i in dimensions_to_ignore:
      new_idx.append(-1)
    else:
      new_idx.append(count)
      count += 1

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use, np.array(new_idx)
