
import data_utils
import cPickle

actions = ["walking", "eating", "smoking", "discussion"]
models  = ["lstm3lr", "erd", "srnn"]

train_subjects = [1, 6, 7, 8, 9, 11]
data_dir = "../../rnn/data/ashesh"

for action in actions:

  train_set, complete_train = data_utils.load_data(data_dir, train_subjects, [action])

  data_mean, data_std, dim_to_ignore, _ = data_utils.normalization_stats(complete_train)

  data_stats = {}
  data_stats['mean'] = data_mean
  data_stats['std'] = data_std
  data_stats['ignore_dimensions'] = dim_to_ignore

  print( data_mean.shape )
  print( data_std.shape )

  #print( dim_to_ignore )

  # for model in models:
  #
  #   bpath = "/usr/local/src/RNNexp/structural_rnn/h3.6m/{0}_{1}/h36mstats.pik".format(model, action)
  #   cPickle.dump(data_stats, open(bpath,"wb"))
