# CoNLL2020
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![Commit Activity](https://img.shields.io/github/last-commit/stevend94/CoNLL2020?color=blue)
![Complete](https://img.shields.io/badge/Complete-90%25-green)

# Paper:
Code used in the paper [Analysing Word Representation in the Input and Output Layers of Neural Language Models](https://www.aclweb.org/anthology/2020.conll-1.36.pdf).  

# Progress:
This repository contains the majority of the code needed to run the experiments in the paper. A number of legacy models and packages are required to build and run the system, plus a number of benchmarks are required to replicate the results. Here, we use the lm_1b model which has moved to the tensorflow archive. 

# Model:
https://github.com/tensorflow/models/tree/archive/research/lm_1b

Follow instructions to download model, or run lm_1b.py --mode get_data

Prerequisites:
* Install TensorFlow 1.x.
* Install Keras
* Install pytorch
* GenSim
* fastText
* glove 

# Benchmarks:
These will be added to the pipeline, but include the following.
* vecto -> https://pypi.org/project/vecto/
* BrainBench -> http://www.langlearnlab.cs.uvic.ca/brainbench/
* SentEval -> https://github.com/facebookresearch/SentEval

While vecto and Brainbench are included in the src, SentEval experiments will require you to clone the repository and run the experiments yourself using the numpy files. The code needed to run evaluate the models on SentEval is included in the src file. SentEval will require you to install pytorch.


# Neural Language Model
Requires PennTreeBank dataset and preprocessing. Find the dataset at 

* train -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt
* test  -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt
* valid -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt
        
# Citation
```latex
@inproceedings{derby2020analysing,
  title={Analysing Word Representation from the Input and Output Embeddings in Neural Network Language Models},
  author={Derby, Steven and Miller, Paul and Devereux, Barry},
  booktitle={Proceedings of the 24th Conference on Computational Natural Language Learning},
  pages={442--454},
  year={2020}
}
```
