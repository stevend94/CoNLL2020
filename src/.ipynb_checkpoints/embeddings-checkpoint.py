# script to construct word2vec model
import numpy as np
import os
import sys
from gensim.models import Word2Vec
import fasttext
from glove import Glove
from glove import Corpus
from multiprocessing import cpu_count
from keras.preprocessing.text import text_to_word_sequence
import argparse 


class TextIterator:
    def __init__(self, path):
        self.path = path
        self.length = self.num_lines()
        self.gen = self.build_generator()
        self.iter = 0
    
    def __iter__(self):
        return self
       
    
    def build_generator(self):
        with open(self.path, 'r') as f:
            for line in f:
                yield line.split()
    
    def num_lines(self):
        """Function to get the number of lines"""
        length = 0
        with open(self.path, 'r') as f:
            for line in f:
                length += 1
                
        return length
        
    
    def __next__(self):
        self.iter += 1
        next_iter = next(self.gen)
        if self.iter == self.length:
            self.iter = 0
            self.gen = self.build_generator()
            raise StopIteration()
            
        return next_iter
    
    

def combineDocs(path = 'data/training-monolingual.tokenized.shuffled', save_path = 'data/1b.txt'):
    '''
        Function to combine several documents in a folder into one long txt file
    '''
    files = os.listdir(path)

    for file in files:
        with open(path + '/' + file, 'r') as f:
            data = f.read().split('\n')

    new_data = []
    for d in data:
        new_data.append('<S> ' + d[:-2] + ' <\S>' + ' .')

    joint_new_data = '\n'.join(new_data)

    with open(save_path, 'a') as f:
        f.write(joint_new_data)

        
def textGenerator(path):
    '''
        Function to load, process and return.
    '''

    with open(path, 'r') as f:
        for line in f:
            yield line.split()
            

def buildCorpus(data_path = None, context_window = 5):
    # function that loads in wikipedia data and fits corpus model
    print('Fitting data...')

    # intialize and fit corpus
    corpus = Corpus()
    corpus.fit(textGenerator(data_path), window = context_window)
    return corpus


def trainGlove(path, no_components = 100, learning_rate = 0.05, epochs = 100, no_threads = 1, verbose = True, context_window = 5, save_path = 'outputs/Glove'):
    # function to load in and train GloVe model
    print('Training Glove Model...')
    glove = Glove(no_components = no_components, learning_rate = learning_rate)
    corpus = buildCorpus(path, context_window)    
    glove.fit(corpus.matrix, epochs = epochs, no_threads = no_threads, verbose = 1)
    glove.add_dictionary(corpus.dictionary)

    # glove.save(save_path + '/glove.model')
    
    with open('data/words.txt', 'r') as f:
        words = f.read().split('\n')[:-1]
        
    shared_words = list(set(words).intersection(set(list(corpus.dictionary))))

    glove_dict = {}
    for word in shared_words:
        glove_dict[word] = glove.word_vectors[glove.dictionary[word],:]
    
    np.save('DSMs/glove.npy', glove_dict)

    
def trainWord2Vec(data_path, embedding_size = 300, context_window = 5, min_count = 5, n_jobs = 1, save_path = 'outputs/Word2Vec'):
    # function that fits a word2vec using one text file
    print('Training Word2Vec Model...')
    
    text_iter = TextIterator(data_path)
    model = Word2Vec(text_iter,
                    size = embedding_size,
                    window = context_window,
                    min_count = min_count ,
                    workers = n_jobs,
                    sg = 1)

    model.save(save_path + '/word2vec_n' + str(embedding_size) + '_c' + str(context_window))
    saveWord2Vec(model, save_path, embedding_size, context_window)


def trainWord2VecMulti(data_path, embedding_size = 300, context_window = 5, min_count = 5, n_jobs = 1, save_path = 'outputs/Word2Vec'):
    # function that fits a word2vec using multiple sources of data from a folder
    print('Training Word2Vec Model...')
    data_files = os.listdir(data_path)

    # do initial training
    sys.stdout.write('Progress - 0 of ' + str(len(data_files)+1))
    model = Word2Vec([x for x in textGenerator(path = data_path + '/' + data_files[0])],
                    size = embedding_size,
                    window = context_window,
                    min_count = min_count ,
                    workers = n_jobs,
                    sg = 1)

    # now iteratively do the rest of the training
    for progress, path in enumerate(data_files[1:]):
        sys.stdout.write( '\r' + 'Progress - ' + str(progress+1) + ' of ' + str(len(data_files)+1))

        text_data = [x for x in textGenerator(path = data_path + '/' + path)]
        model.build_vocab(text_data, update = True)
        model.train(text_data, total_examples=model.corpus_count, epochs=model.iter)

    sys.stdout.write('\r' + 'Progress - ' + str(len(data_files)) + ' of ' + str(len(data_files)+1) + '\n')

    model.save(save_path + '/word2vec_n' + str(embedding_size) + '_c' + str(context_window))
    saveWord2Vec(model, save_path, embedding_size, context_window)

    
def trainFastText(data_path, embedding_size = 300, context_window = 5, min_count = 5, save_path = 'outputs/FastText'):
    # function that fits a fastext model, can only be in one file path
    print('Training FastText Model...')

    # train model
    model = fasttext.skipgram(data_path, save_path + '/fasttext', dim = embedding_size, ws = context_window, min_count = min_count)

    # save model
    with open('data/words.txt', 'r') as f:
        words = f.read().split('\n')[:-1]
        
    shared_words = list(set(words).intersection(set(model.words)))

    fasttext_dict = {}
    for word in shared_words:
        fasttext_dict[word] = model[word]

    np.save('DSMs/fasttext.npy', fasttext_dict)

    
def saveWord2Vec(w2v, save_path, embedding_size, context_window):
    # function used to save word2vec in dictionary format, save only words shared by image models
    with open('data/words.txt', 'r') as f:
        words = f.read().split('\n')[:-1]
        
    shared_words = set(words).intersection(set(w2v.wv.vocab))
    word2vec_matrix = np.zeros((len(shared_words), embedding_size))

    word2vec_dict = {}
    for word in shared_words:
        word2vec_dict[word] = w2v.wv[word]

    np.save('DSMs/word2vec.npy', word2vec_dict)

    
if __name__ == '__main__':
    # for this just use one source of data
    n_jobs = cpu_count()
    
    data_path = 'data/1b.txt'
    # Set up a few command line arguments for program
    parser = argparse.ArgumentParser(description = 'Module used to construct embedding models.')

    parser.add_argument('--word2vec', '-wv', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to word2vec model using default parameters.'))
    
    parser.add_argument('--fasttext', '-ft', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to fasttext model using default parameters.'))
    
    parser.add_argument('--glove', '-gl', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to glove model using default parameters.'))
    
    parser.add_argument('--combine', '-c', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to combine all documents in 1 billion word dataset.'))
    
    args = parser.parse_args()
    
    if args.word2vec:
        # build word2vec model and save as 
        if not os.path.exists('outputs/Word2Vec'):
            os.mkdir('outputs/Word2Vec')
        trainWord2Vec(data_path = 'data/1b.txt', 
                           embedding_size = 300, 
                           context_window = 5, 
                           min_count = 0, 
                           n_jobs = n_jobs, 
                           save_path = 'outputs/Word2Vec')
    elif args.fasttext:
        # build fasttext model and save as 
        if not os.path.exists('outputs/FastText'):
            os.mkdir('outputs/FastText')
        trainFastText(data_path = 'data/1b.txt', 
                           embedding_size = 300, 
                           context_window = 5, 
                           min_count = 0, 
                           save_path = 'outputs/FastText')
    elif args.glove:
        # build glove model and save as 
        if not os.path.exists('outputs/Glove'):
            os.mkdir('outputs/Glove')
        trainGlove(path = 'data/1b.txt', 
                    no_components = 300, 
                    context_window = 5, 
                    learning_rate = 0.05,
                    epochs = 40,
                    no_threads = n_jobs, 
                    save_path = 'outputs/Glove')
    elif args.combine:
        # combine 1 billion word dataset into one txt file for training
        combineDocs()
    else:
        raise Exception('No commands provided.')
        

