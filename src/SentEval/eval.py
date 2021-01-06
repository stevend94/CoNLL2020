from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging
import argparse 

parser = argparse.ArgumentParser(description='Run SentEval on word embeddings')
parser.add_argument('--path', type=str, help='path to word embeddings (numpy format)')


# Set PATHs
PATH_TO_SENTEVAL = '.'
PATH_TO_DATA = './data'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_embeddings(path_to_vec, word2id):
    embeddings = {}
    
    numpy_vectors = np.load(path_to_vec).item()
    
    wvec_dim = np.asarray(list(numpy_vectors.values())).shape[1]
    
    for word in word2id:
        if word in numpy_vectors:
            embeddings[word] = numpy_vectors[word]
        else:
            embeddings[word] = np.zeros(wvec_dim)

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(embeddings), len(word2id)))
    return embeddings, wvec_dim


# SentEval prepare and batcher
def prepare(path_to_vec):
    
    def prepare_func(params, samples):
        _, params.word2id = create_dictionary(samples)
        params.word_vec, params.wvec_dim = get_embeddings(path_to_vec, params.word2id)
        return
    
    return prepare_func

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []

    for sent in batch:
        sentvec = []
        for word in sent:
            if word in params.word_vec:
                sentvec.append(params.word_vec[word])
        if not sentvec:
            vec = np.zeros(params.wvec_dim)
            sentvec.append(vec)
        sentvec = np.mean(sentvec, 0)
        embeddings.append(sentvec)

    embeddings = np.vstack(embeddings)
    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    
    args = parser.parse_args()
    
    se = senteval.engine.SE(params_senteval, batcher, prepare(args.path))
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
    
    name = args.path.split('/')[-1]
    np.save("results/" + name, results)
    
