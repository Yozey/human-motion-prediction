
## human-motion-prediction

This is the code for the paper

```
Julieta Martinez, Michael J. Black, Javier Romero.
On human motion prediction using recurrent neural networks. CVPR 17.
```

Our code runs under [Tensorflow](https://github.com/tensorflow/tensorflow/) 1.0 or later.

The code in this repository was written by [una-dinosauria](https://github.com/una-dinosauria/) and [libicocco](https://github.com/libicocco/).

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

### RNN models

To train and reproduce the results of our models, use the following commands

| model      | arguments | notes |
| ---        | ---       | ---   |
| Sampling-based loss (SA) | `python src/translate.py --action walking` | Realistic long-term motion |
| Residual (SA)            | `python src/translate.py --residual_velocities --action walking` |  |
| Residual sup. (MA)       | `python src/translate.py --residual_velocities --learning_rate 0.005 --omit_one_hot` |  |
| Residual sup. (MA)       | `python src/translate.py --residual_velocities --learning_rate 0.005` | best quantitative |
| Untied       | `python src/translate.py --residual_velocities --learning_rate 0.005 --architecture basic` |  |


You can substitute the `--action walking` parameter for any action in 

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

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


### TODOS

* Remove "ashesh" from variable and function names
* Remove functions and variables that are not used
* Check that all results reproduce

