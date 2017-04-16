#!/bin/bash

# for action in walking eating smoking discussion
# do
#   for learning_rate_decay_factor in 1.0 0.99 #0.98 0.97 0.96 0.95
#   do
#     echo $action $learning_rate_decay_factor
#     # use_gpu means don't use the gpu -- i know
# 	  python translate.py --sample --use_gpu --try_to_load 10000 --action $action --learning_rate_decay_factor $learning_rate_decay_factor --seq_length_out 100 --seq_length_in 50
#   done
# done

for action in all
do
  for seq_out in 10
  #for seq_out in 10
  do
    for learning_rate in 0.01
    #for learning_rate in 0.005
    do
      echo $action $seq_in $seq_out $learning_rate
      python translate.py --sample --use_cpu --try_to_load 30000 --action $action --seq_length_out $seq_out  --learning_rate $learning_rate
    done
  done
done
