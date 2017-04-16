
### human-motion-prediction

This is the code for the paper

```
Julieta Martinez, Michael J. Black, Javier Romero. 
On human motion prediction using recurrent neural networks. CVPR 17.
```

Our code runs under [Tensorflow](https://github.com/tensorflow/tensorflow/) 1.0 or later.

The code in this repository was written by [una-dinosauria](https://github.com/una-dinosauria/).

#### Get the data

First of all, you need to get the human3.6m dataset on exponential map format.

```bash
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

#### Baselines

A major finding in our paper is that a set of simple baselines outperform the state of the art in human motion prediction. To reproduce these baseline results, run

`python predict_motion.py --run_baselines`

#### Sequence-to-sequence training

You can also reproduce our results on all the actions of our strongest model by running

`python predict_motion.py --multiaction --supervised`

#### Citing

If you use our code, please cite our work

```
@inproceedings{martinez:motion17,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

#### Licence
MIT
