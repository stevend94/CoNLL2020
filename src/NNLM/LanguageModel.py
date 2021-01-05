# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is based on an early version of the tensorflow example on 
# language modeling with PTB (at /models/blob/master/tutorials/rnn/ptb/ptb_word_lm.py)
#
# Hakan Inan & Khashayar Khosravi
#

import reader3 as rdr
import tensorflow as tf
import collections
import math, sys
import time
import argparse
import numpy as np
import os
from collections import Counter
import requests

PENN_URLS = {
    'Train': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    'Test': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
    'Valid': "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt"
}

def none_pointer_copy(dictionary):
    new_dictionary = {}
    for key in list(dictionary):
        new_dictionary[key] = dictionary[key]

    return new_dictionary


def get_data():
    """
        Function to get PennTreeBank data (train, test and validation splits)
    """
    
    print("Downloading PennTreeBank Data from  https://raw.githubusercontent.com/wojzaremba/lstm/master/data ...  ", end = "")
    for url_key in PENN_URLS:
        url = PENN_URLS[url_key]
        data = requests.get(url, stream = True)
        file_name = url.split("/")[-1]
        
        with open('data/PennTreeBank/' + file_name, 'wb') as f:
            for chunk in data.iter_content(chunk_size = 1024):
                if chunk:
                    f.write(chunk)
                    
    print("Done")


def buildData(path, dsm):
    '''
        Function to process ptb dataset
    '''
    
    if len(os.listdir(path)) == 0:
        get_data()
        

    # Load all data from path
    with open(path + '/ptb.train.txt', 'r') as f:
        train = f.read().split('\n')

    with open(path + '/ptb.test.txt', 'r') as f:
        test = f.read().split('\n')

    with open(path +'/ptb.valid.txt', 'r') as f:
        valid = f.read().split('\n')

    # collect words that overlap
    words = list(dsm.keys())
    counter = Counter([x for sentence in train for x in sentence.split()])
    overlapping_vocab = list(set(list(counter.keys())).intersection(set(words)))
    identity_map = dict(zip(overlapping_vocab, overlapping_vocab))
    print('Number of Words:', len(overlapping_vocab))

    # process data
    processed_train = [[identity_map.setdefault(x, '<UNK>') for x in sentence.split()] for sentence in train]
    processed_test = [[identity_map.setdefault(x, '<UNK>') for x in sentence.split()] for sentence in test]
    processed_valid = [[identity_map.setdefault(x, '<UNK>') for x in sentence.split()] for sentence in valid]

    # write data back
    if not os.path.exists(path + '/processed'):
        os.mkdir(path + '/processed')
    with open(path + '/processed/ptb.train.txt', 'w') as f:
        f.write(' \n '.join([' '.join(x) for x in processed_train]))

    with open(path + '/processed/ptb.test.txt', 'w') as f:
        f.write(' \n '.join([' '.join(x) for x in processed_test]))

    with open(path + '/processed/ptb.valid.txt', 'w') as f:
        f.write(' \n '.join([' '.join(x) for x in processed_valid]))

        
        
class LanguageModel(object):
  """2-Layer LSTM language model"""

  def __init__(self, is_training, config, weight_embeddings, vocabulary ):
    self.config = config
    self.vocabulary = vocabulary

    self._input_data = tf.placeholder(tf.int32, [self.config['batch_size'], self.config['num_steps']])
    self._targets = tf.placeholder(tf.int32, [self.config['batch_size'], self.config['num_steps']])
    embedding_size = weight_embeddings.shape[1]

    with tf.name_scope('Language_Model'):
        # build embedding lookup table 
        with tf.variable_scope('embedding'):
                if self.config['model'] == 'output' or self.config['model'] == 'test':
                    embedding_matrix = tf.get_variable("embedding",
                                    [self.vocabulary, self.config['size']],
                                    dtype=tf.float32)
                else:
                    embedding_matrix = tf.constant(weight_embeddings, dtype = tf.float32)

        with tf.variable_scope('rnn'):
            lstms = [tf.contrib.rnn.BasicLSTMCell(size, forget_bias=0.0) for size in [self.config['size'], self.config['size']]]

            drops = [tf.contrib.rnn.DropoutWrapper(lstm, input_size=self.config['size'],
                                                        output_keep_prob=self.config['dropout'],
                                                        dtype=tf.float32) for lstm in lstms]
            cell = tf.nn.rnn_cell.MultiRNNCell(drops, state_is_tuple=True)

            self._initial_state = cell.zero_state(self.config['batch_size'], tf.float32)

    inputs = tf.nn.embedding_lookup(embedding_matrix, self.input_data)
    inputs = tf.nn.dropout(inputs, self.config['dropout'])
    inputs, state = tf.nn.dynamic_rnn(cell, inputs, initial_state=self._initial_state)

    if self.config['model'] == 'test':
         with tf.variable_scope('softmax_output'):
                    softmax_w = tf.get_variable(
                        "softmax_w", [self.config['size'], self.vocabulary], dtype=tf.float32)
                    softmax_b = tf.get_variable("softmax_b", [self.vocabulary], dtype=tf.float32)

    if self.config['model'] == 'input':
                # Set inputs as pretrained embeddings
        with tf.variable_scope('softmax_output'):
                    softmax_w = tf.get_variable(
                        "softmax_w", [embedding_size, self.vocabulary], dtype=tf.float32)
                    softmax_b = tf.get_variable("softmax_b", [self.vocabulary], dtype=tf.float32)
                    #inputs = tf.transpose(inputs, [1, 0, 2])

    if self.config['model'] == 'output':
        # Set outputs as pretrained embeddings
        softmax_w = tf.constant(weight_embeddings.T, dtype = tf.float32)
        softmax_b = tf.zeros(shape=[self.vocabulary], dtype=tf.float32, name="softmax_b")
            #inputs = tf.transpose(inputs, [1, 0, 2])


    if self.config['model'] == 'tied':
        # tie input embedding weights to output embedding weights which are non trainable
        softmax_w = tf.constant(weight_embeddings.T, dtype = tf.float32)
        softmax_b = tf.zeros(shape=[self.vocabulary], dtype=tf.float32, name="softmax_b")


    self.W = softmax_w
    self.b = softmax_b

    # calculate logits
    inputs = tf.reshape(inputs, [-1, self.config['size']])

    if not self.config['model'] == 'test':
      projection = tf.get_variable(
                "projection", [self.config['size'], embedding_size], dtype=tf.float32)
      inputs = tf.matmul(inputs, projection)

    logits = tf.nn.xw_plus_b(inputs, softmax_w, softmax_b)
    # Reshape logits to be a 3-D tensor for sequence loss
    logits = tf.reshape(logits, [self.config['batch_size'], self.config['num_steps'], self.vocabulary])

    loss = tf.contrib.seq2seq.sequence_loss(logits,
                                            self.targets,
                                            tf.ones([self.config['batch_size'], self.config['num_steps']], dtype=tf.float32),
                                            average_across_timesteps=False,
                                            average_across_batch=True)

    # labels = tf.reshape(self._targets, [-1])
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits,labels = labels)

    self._cost = cost = tf.reduce_sum(loss)

    self._lr = tf.Variable(0.0, trainable=False)
    self._perp = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return    

    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      self.config['clip_norm'])
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self.optimizer = optimizer
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def perp(self):
    return self._perp

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op

def run_epoch(session, m, data, eval_op):
  epoch_size = ((len(data) // m.config['batch_size']) - 1) // m.config['num_steps']
  start_time = time.time()
  costs = 0.0
  perps = 0.0
  iters = 0
  state = session.run(m.initial_state)
  
  epoch_size, iterator = rdr.ptb_iterator(data, m.config['batch_size'], m.config['num_steps'])
  for step, (x, y) in enumerate(iterator):

    perp, cost, state, _ = session.run([m.perp, m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})                                 

    costs += cost
    perps += perp
    iters += m.config['num_steps']

    print("%d / %d exp_cost: %.3f perplexity: %.3f speed: %.0f wps" %
            (step, epoch_size,np.exp(costs / iters), np.exp(perps / iters),
             iters * m.config['batch_size'] / (time.time() - start_time)), end = '\r')

  return np.exp(perps / iters)


def run(config, dsm):
    '''
        Run language model 
    '''
    train_data, valid_data, test_data, vocabulary, word_to_id, id_to_word  = rdr.ptb_raw_data(config['data_path'])

    # create LSTM layers
    embedding_size = 650
    weight_embeddings = np.zeros((1,650))
    if not config['model'] == 'test': 
        embedding_size = np.asarray(list(dsm.values())).shape[1]
        weight_embeddings = np.zeros((len(word_to_id), embedding_size))

        for word in word_to_id.keys():
            weight_embeddings[word_to_id[word],:] = dsm[word]
        

    # create validation and test configerations withourt
    test_config = {}
    valid_config = {}

    test_config = none_pointer_copy(config)
    valid_config = none_pointer_copy(config)
    test_config['batch_size'] = 1
    test_config['num_steps'] = 1
    test_config['dropout'] = 1.0
    valid_config['dropout'] = 1.0

    train_perplexity = []
    valid_perplexity = []
    test_perplexity = []

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config['init_val'], config['init_val'])
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LanguageModel(is_training=True, config=config, weight_embeddings = weight_embeddings, vocabulary = vocabulary)
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = LanguageModel(is_training=False, config=valid_config, weight_embeddings = weight_embeddings, vocabulary = vocabulary)
            mtest = LanguageModel(is_training=False, config=test_config, weight_embeddings = weight_embeddings, vocabulary = vocabulary)

        tf.initialize_all_variables().run()

        
        for i in range(config['epochs']):
            lr_decay = config['decay_rate']** max(i - config['max_epochs'], 0)
            m.assign_lr(session, config['learning_rate'] * lr_decay)    

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr),))
            train_perplexity.append(run_epoch(session, m, train_data, m.train_op))
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity[-1]))
            valid_perplexity.append(run_epoch(session, mvalid, valid_data, tf.no_op()))
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity[-1]))

        test_perplexity.append(run_epoch(session, mtest, test_data, tf.no_op()))
        print("Test Perplexity: %.3f" % test_perplexity[-1])
        return train_perplexity,  valid_perplexity, test_perplexity


if __name__ == "__main__":
    # Set up a few command line arguments for program
    parser = argparse.ArgumentParser(description = 'Language model module with several different modes for the final softmax layer.')

    parser.add_argument('--check_gpu', '-g', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to check if there is a gpu installed for tensorflow. If there is a GPU that is not detected then CUDDA may need to be installed'))

    parser.add_argument('--data_path', '-d', action = 'store',
                        required = False,
                        default = 'data/PennTreeBank/processed',
                        help = ('Function to get data for lm_1b model'))

    parser.add_argument('--size', '-s', action = 'store',
                        required = False,
                        default = '650',
                        help = ('Size of hidden layers of model'))

    parser.add_argument('--learning_rate', '-lr', action = 'store',
                        required = False,
                        default = '1.0',
                        help = ('Learning rate of languge model whem training.'))
    parser.add_argument('--batch_size', '-bs', action = 'store',
                        required = False,
                        default = '20',
                        help = ('Batch size for training.'))
    parser.add_argument('--num_steps', '-ns', action = 'store',
                        required = False,
                        default = '35',
                        help = ('Time step for unrolling lstms of language model.'))
    parser.add_argument('--dropout_keep', '-dk', action = 'store',
                        required = False,
                        default = '0.5',
                        help = ('Probability of keeping a value using dropout.'))

    parser.add_argument('--epochs', '-ep', action = 'store',
                        required = False,
                        default = '39',
                        help = ('Epochs used for training.'))

    parser.add_argument('--max_epochs', '-mep', action = 'store',
                        required = False,
                        default = '6',
                        help = ('Max epochs before learning decay schedule activates.'))

    parser.add_argument('--decay_rate', '-dr', action = 'store',
                        required = False,
                        default = '0.8',
                        help = ('Decay rate to decrease learning rate by after each epoch.'))

    parser.add_argument('--clip_norm', '-cn', action = 'store',
                        required = False,
                        default = '5.0',
                        help = ('Size of global clip norm of gradients.'))

    parser.add_argument('--init_value', '-iv', action = 'store',
                        required = False,
                        default = '0.05',
                        help = ('Range of values +- for the uniform random distribution initializer.'))

    parser.add_argument('--model', '-m', action = 'store',
                        required = False,
                        default = 'baseline',
                        help = ('Type of model to build, including "baseline", "tied", "tied+h" and "htied.'))
    
    parser.add_argument('--dsm_path', '-dsm', action = 'store',
                        required = False,
                        default = 'new',
                        help = ('Path to distributional Model.'))
    
    parser.add_argument('--save_path', '-sp', action = 'store',
                        required = False,
                        default = 'History',
                        help = ('Path to save history information.'))
    
    parser.add_argument('--build', '-bu', action = 'store',
                        required = False,
                        default = 'False',
                        help = ('Function to build data'))


    args = parser.parse_args()

    if args.check_gpu:
        print(tf.test.is_gpu_available())

    # build configuration file
    config = {
        'batch_size': int(args.batch_size),
        'num_steps': int(args.num_steps),
        'epochs': int(args.epochs),
        'data_path': args.data_path,
        'decay_rate': float(args.decay_rate),
        'learning_rate': float(args.learning_rate),
        'init_val': float(args.init_value),
        'model': args.model,
        'clip_norm': float(args.clip_norm),
        'max_epochs': int(args.max_epochs),
        'dropout': float(args.dropout_keep),
        'size': int(args.size)
    }
    
    name = args.model + '_' + args.dsm_path.split('/')[-1][:-4] 
    print(name)
    dsm = np.load(args.dsm_path).item()
    if args.build == 'False':
        train_perplexity,  valid_perplexity, test_perplexity = run(config = config, dsm = dsm)

        name = args.model + '_' + args.dsm_path.split('/')[-1][:-4] 

        history = {} 
        history['train'] = train_perplexity
        history['valid'] = valid_perplexity
        history['test'] = test_perplexity

        np.save(args.save_path + '/' + name + '_history.npy', history)
    else:
        buildData(path = args.data_path, dsm = dsm)
#     else:
#         raise Exception('No commands provided. Please choose one of the options.')

    
