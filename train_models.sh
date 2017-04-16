#!/bin/bash

for action in walking eating smoking discussion
do
  for learning_rate_decay_factor in [0.98, 0.97, 0.96, 0.95]
  do
     echo $action $learning_rate_decay_factor
     python translate.py --action $action --learning_rate_decay_factor $learning_rate_decay_factor --seq_length_in 25
  done
done

# do
#   echo $action
# done

#python trasnlate.py --action "salking"
