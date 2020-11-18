# CoNLL2020
![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/stevend94/CoNLL2020.js/graphs/commit-activity)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![Commit Activity](https://img.shields.io/github/commit-activity/m/stevend94/CoNLL2020)
![Complete](https://img.shields.io/badge/Complete-pending-orange)

# Paper:
Code used in the paper Analysing Word Representation in the Input and Output Layers of Neural Language Models.  

https://www.aclweb.org/anthology/2020.conll-1.36.pdf

# Progress:
Only some of the code if provided until we can find a good way to present and run the systems. A number of legacy models and packages are required to build and run the system, plus a number of benchmarks are required to replicate the results. Here, we use the lm_1b model which has moved to the tensorflow archive. 

# Model:
https://github.com/tensorflow/models/tree/archive/research/lm_1b

Follow instructions to download model, or run lm_1b.py --mode get_data

Prerequisites:
* Install TensorFlow 1.x.
* Install Keras
* GenSim
* fastText
* glove 

# Benchmarks:
These will be added to the pipeline, but include the following.
* vecto -> https://pypi.org/project/vecto/
* BrainBench -> http://www.langlearnlab.cs.uvic.ca/brainbench/
* SentEval -> https://github.com/facebookresearch/SentEval

# Neural Language Model
Requires PennTreeBank dataset and preprocessing. Find the dataset at 

* train -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt
* test  -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt
* valid -> https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt
        
