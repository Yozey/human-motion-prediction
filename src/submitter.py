
"""Sends jobs with all combinations of certain parameters"""

import os
import itertools

fname   = "jobseq2seq.sub"

def submit_jobs( fields, values ):
  """
  Function that submits jobs with certain parameters

  Args.
    fields. List of strings with parameters to pass.
    values. Tuple of lists. Each list has values for each field.

  Returns
    Nothing. The jobs will be sent to sched.

  """

  # === Argument examples ===
  # fields = ["learning_rate", "seq_length_out", "loss_velocities"]
  #
  # learning_rate   = [0.2, 0.1, .05, 0.01]
  # seq_length_out  = [10, 25]
  # loss_velocities = [1.0, 0.5]
  # values = ( learning_rate, seq_length_out, loss_velocities )

  configs = []
  for element in itertools.product( *values ):
    configs.append( element )

  for config in configs:

    # print config
    line = "arguments = translate.py "

    for i in range( len(config) ):
      if isinstance( config[i], bool ):
        if config[i]:
          line = line + "--{0} ".format( fields[i] )
        else:
          pass # do not add the flag

      else:
        line = line + "--{0} {1} ".format( fields[i], config[i] )


    line = line + "\n"
    print( line )

    # Open the submission file
    with open(fname, "r") as f:
      data = f.readlines()

    # Change the arguments line
    data[1] = line

    # Write back
    with open(fname, "w") as f:
      f.writelines( data )

    # Send to the cluster
    bashCommand = "condor_submit {0}".format( fname )
    print( bashCommand )
    os.system( bashCommand )

def lstm3lr_no_noise_experiment():

  actions = ["walking", "eating", "smoking", "discussion"]

  # (1) action-specific, lsmt-3lr, no noise, supervised
  fields = []
  values = ()

  fields.append("action")
  values = values + (actions,)

  fields.append("loss_to_use")
  values = values + (["supervised", "self_fed"],)

  fields.append("omit_one_hot")
  values = values + ([True],)

  fields.append("num_layers")
  values = values + ([3],)

  fields.append("use_lstm")
  values = values + ([True],)

  fields.append("space_encoder")
  values = values + ([True],)

  fields.append("learning_rate")
  #values = values + ([0.005, 0.01],)
  values = values + ([0.05, 0.1],)

  fields.append("train_dir")
  values = values + (["/is/ps2/jmartinez2/Desktop/scratch/cvpr_experiments_1/"],)

  submit_jobs(fields, values)

def lstm3lr_residual_self_fed_experiment():

  actions = ["walking", "eating", "smoking", "discussion"]

  # (1) action-specific, lsmt-3lr, no noise, supervised
  fields = []
  values = ()

  fields.append("action")
  values = values + (actions,)

  fields.append("residual_velocities")
  values = values + ([True],)

  fields.append("loss_to_use")
  values = values + (["self_fed"],)

  fields.append("omit_one_hot")
  values = values + ([True],)

  fields.append("num_layers")
  values = values + ([3],)

  fields.append("use_lstm")
  values = values + ([True],)

  fields.append("space_encoder")
  values = values + ([True],)

  fields.append("learning_rate")
  values = values + ([0.005, 0.01],)

  fields.append("train_dir")
  values = values + (["/is/ps2/jmartinez2/Desktop/scratch/cvpr_experiments_2/"],)

  submit_jobs(fields, values)

def rebuttal_experimentst():

  actions = ["directions", "discussion", "eating", "greeting", "phoning",
              "posing", "purchases", "sitting", "sittingdown", "smoking",
              "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]

  # (1) action-specific, lsmt-3lr, no noise, supervised
  fields = []
  values = ()

  fields.append("action")
  values = values + (actions,)

  fields.append("residual_velocities")
  values = values + ([True],)

  fields.append("loss_to_use")
  values = values + (["self_fed"],)

  fields.append("omit_one_hot")
  values = values + ([True],)

  fields.append("num_layers")
  values = values + ([1],)

  fields.append("learning_rate")
  values = values + ([0.05],)

  # fields.append("train_dir")
  # values = values + (["/is/ps2/jmartinez2/Desktop/scratch/cvpr17_rebuttal_experiments/"],)

  submit_jobs(fields, values)

def main():
  # First experiment
  # lstm3lr_no_noise_experiment()

  # Second experiment
  # lstm3lr_residual_self_fed_experiment()

  # rebuttal experiments
  rebuttal_experiments()

if __name__ == "__main__":
  main()
