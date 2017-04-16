import subprocess, os
bpath = "/is/ps2/jmartinez2/Desktop/scratch/gru/{0}/no_space_encoder/{1}/videos"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/conditioning_experiments/all/batch_16/in_25/out_25/0.005/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/conditioning_experiments/all/iterations_100000/batch_16/in_50/out_25/0.005/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/residual_experiments/all/gru/depth_6/size_512/lr_0.005/residual_vel/not_residual_rnn/not_space_encoder/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/residual_experiments/all/out_10/tied/gru/depth_1/size_1024/lr_0.01/residual_vel/not_residual_rnn/not_space_encoder/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/loss_velocities/all/out_25/loss_velocities1.0/tied/gru/depth_1/size_1024/lr_0.2/residual_vel/not_residual_rnn/not_space_encoder/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/cvpr_experiments_3/all/out_10/tied/supervised/one_hot/gru/depth_1/size_1024/lr_0.01/not_add_noise/residual_vel/not_residual_rnn/not_space_encoder/videos_{0}"
bpath = "/is/ps2/jmartinez2/Desktop/scratch/cvpr_experiments_3/all/out_10/tied/supervised/one_hot/gru/depth_1/size_1024/lr_0.01/add_noise/residual_vel/not_residual_rnn/not_space_encoder/videos_{0}"
#for action in ["walking", "eating", "smoking", "discussion"]:
#  for decay in [1.0, 0.99]:

for action in ["walking", "eating", "smoking", "discussion"]:

  this_path = bpath.format( action )

  find_cmd = 'find {0}/ -name \'*.avi\' -printf \"file \'%p\'\\n\"'.format(this_path)

  #import pdb; pdb.set_trace()
  output, error = subprocess.Popen( find_cmd, universal_newlines=True, shell=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()

  with open('/tmp/mylist.txt', 'w') as f: #tempfile.NamedTemporaryFile(mode='w') as f:
      f.write(output)
  cmd = "ffmpeg -y -safe 0 -f concat -i {0} -c copy {1}/output.avi".format(f.name, this_path )
  print( cmd )
  #os.system( cmd )
  output,error = subprocess.Popen( cmd, universal_newlines=True, shell=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
  print output
  print error
