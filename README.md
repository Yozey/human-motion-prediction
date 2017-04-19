
## human-motion-prediction

This is the code for the paper

```
Julieta Martinez, Michael J. Black, Javier Romero.
On human motion prediction using recurrent neural networks. CVPR 17.
```

Our code runs under [Tensorflow](https://github.com/tensorflow/tensorflow/) 1.0 or later.

The code in this repository was written by [Julieta Martinez](https://github.com/una-dinosauria/) and [Javier Romero](https://github.com/libicocco/).

### Get this code and the data

First things first, clone this repo and get the human3.6m dataset on exponential map format.

```bash
git clone git@github.com:una-dinosauria/human-motion-prediction.git
cd human-motion-prediction
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

### Running average baselines

To reproduce the running average baseline results from our paper, run

`python src/baselines.py`

If you get the following error

``` python
Traceback (most recent call last):
  File "src/baselines.py", line 218, in <module>
    main()
  File "src/baselines.py", line 165, in main
    dtype=dtype)
  File "/home/jromero/src/human-motion-prediction/src/seq2seq_model.py", line 135, in __init__
    raise(ValueError, "unknown loss: %s" % loss_to_use)
ValueError
[1]    18026 segmentation fault (core dumped)  python src/baselines.py
```

do WHAT?

### RNN models

To train and reproduce the results of our models, use the following commands

| model      | arguments | notes |
| ---        | ---       | ---   |
| Sampling-based loss (SA) | `python src/translate.py --action walking --seq_length_out 25` | Realistic long-term motion, loss computed over 1 second. Training time in TITAN x: XXX seconds|
| Residual (SA)            | `python src/translate.py --residual_velocities --action walking` |  Training time in TITAN x: XXX seconds|
| Residual unsup. (MA)     | `python src/translate.py --residual_velocities --learning_rate 0.005 --omit_one_hot` |  Training time in TITAN x: XXX seconds|
| Residual sup. (MA)       | `python src/translate.py --residual_velocities --learning_rate 0.005` | best quantitative. Training time in TITAN x: XXX seconds|
| Untied       | `python src/translate.py --residual_velocities --learning_rate 0.005 --architecture basic` |  Training time in TITAN x: XXX seconds|


You can substitute the `--action walking` parameter for any action in

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

The code will log the error in Euler angles for each action to [tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). You can track the progress during training by typing `tensobard --logdir log` in the terminal and checking the board under [this](http://127.0.1.1:6006/) url in your browser.

### Visualization

TODO

### Citing

If you use our code, please cite our work

```
@inproceedings{martinez:motion17,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

### Acknowledgments

The pre-processed human 3.6m dataset and some of our evaluation code (specially under `src/data_utils.py`) was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399).

### Licence
MIT
