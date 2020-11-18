# Python file which has some utility function that can be used
import numpy as np
import os, errno, sys
import pickle
import requests
import argparse
from collections import defaultdict
import json
import pandas as pd

import vecto
import vecto.embeddings
from vecto.utils.fetch_benchmarks import fetch_benchmarks
from vecto.benchmarks.similarity import Similarity
        

def writeVectoEmbeddings(path):
    '''
        Function that will write vecto embeddings from a numpy dictionary with meta data automated. All data is dumped
        into the vecto experimental folder.
    '''

    model_name = path.split('/')[-1]
    print('Writing model', model_name)
    directory = 'experiments/Embeddings/' + model_name[:-4]
    save_path = directory+'/embeddings.txt'
    model = np.load(path).item()

    # check if directory exists first
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write embedding data
    with open(save_path, 'a', encoding = 'utf-8') as f:
        for word in list(model.keys()):
            string = word
            for num in model[word]:
                string += ' ' + str(num)
            f.write(string + '\n')

    # Write metadata
    metadata = {
    "class": "embeddings",
    "model": model_name,
    "uuid" : "f50ec0b7-f960-400d-91f0-c42a6d44e3d0",
    "context": "linear_unbound",
    }

    with open( directory + '/metadata.json', 'w') as outfile:
        json.dump(metadata, outfile)
        
        
def vectoBenchmark(path = '', save_path = '', files = None):
    '''
        Function used to benchmark all embedding models in the vecto experimental folder on
        intrinsic word similarity evaluation benchmarks.
    '''
    
    if save_path == '':
        save_path = 'outputs/intrinsic_scores.csv'
        
    benchmark_path = 'data/benchmarks/benchmarks/similarity/en'
    if path == '':
        path = 'experiments/Embeddings'
    if type(files) is not list:
        files = os.listdir(path)

    # if benchmarks are not there then get them
    if not os.path.exists('data/benchmarks'):
        fetch_benchmarks()

    evals = ['ws353_similarity',
         'ws353_relatedness',
         'simlex999',
         'men',
         'rw',
         'ws353']

    sim = Similarity()
    score_dictionary = defaultdict(dict)
    for f in files:
        print('Scoring', f)
        # Load vector space model and normalize
        vsm = vecto.embeddings.load_from_dir(path + '/' + f)
        vsm.normalize()

        
        for name in evals:
            score = sim.evaluate(vsm, sim.read_test_set(path = benchmark_path + '/' + name + '.txt'))[0:-1]
            print(name, '-', score)
            score_dictionary[f][name] = score[0]
            
            # save scores to txt file 
#             with open(save_path + '.txt', 'a') as g:
#                 g.write(f[:-4] + '\n')
#                 g.write(name + ' - ' + str(score) + '\n')
#                 g.write('\n')
        

        print('')
        
    df = pd.DataFrame(score_dictionary)
    df.to_csv(save_path)

    
def getIntrinsicWords():
	'''
		Function to get intrinsic words from evaluations
	'''
	
	path = 'data/benchmarks/benchmarks/similarity/en'
	
	text_files = [f for f in os.listdir(path) if '.txt' in f]
	
	words = []
	for file in text_file:
		with open(path + '/' + file, 'r') as f:
			data = f.read().split('\n')
    
    
if __name__ == '__main__':
    # Set up a few command line arguments for program
    parser = argparse.ArgumentParser(description = 'Module contains several tools that are handy for various tasks.')

    parser.add_argument('--check_gpu', '-g', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to check if there is a gpu installed for tensorflow. If there is a GPU that is not detected then CUDDA may need to be installed'))
    parser.add_argument('--data', '-d', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to get data for lm_1b model'))
    parser.add_argument('--build', '-b', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to build embeddings from lm_1b model'))
    parser.add_argument('--build_joint', '-bj', action = 'store',
                        required = False,
                        default = False,
                        help = ('Function to build joint embeddings from lm_1b model (requires you run build first)'))
    
    parser.add_argument('--save_path', '-sp', action = 'store',
                        required = False,
                        default = '',
                        help = ('Path to directory where you want the model to be saved.'))

    parser.add_argument('--vecto_write', '-vw', action = 'store',
                        required = False,
                        help = ('Used to turn a set of numpy embeddings models into vecto embedding format (must have experiment folder).'))

    parser.add_argument('--path', '-p', action = 'store',
                        required = False,
                        default = '',
                        help = ('General argument when a path is required.'))

    parser.add_argument('--vecto_bench', '-vb', action = 'store',
                        required = False,
                        help = ('Used to benchmark all embeddings in vecto experimental folder using word similarity benchmarks.'))

    parser.add_argument('--sent_progress', '-spr', action = 'store',
                        required = False,
                        help = ('Used to check progress of embedding extractor.'))
    
    parser.add_argument('--files', '-f', action = 'store',
                        required = False,
                        nargs='+',
                        help = ('Indicate files to process.'))
    
    parser.add_argument('--write_progress', '-wrp', action = 'store',
                        required = False,
                        default = '',
                        help = ('Used to check progress of embedding extractor.'))



    args = parser.parse_args()

    if args.vecto_write:
        writeVectoEmbeddings(path = args.path)
    elif args.vecto_bench:
        vectoBenchmark(save_path = args.save_path, files = args.files)
    else:
        raise Exception('No commands provided. Please choose one of the options.')
        