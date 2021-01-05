import numpy as np
import requests
import os
import argparse
import pickle

words_html = 'http://www.langlearnlab.cs.uvic.ca/brainbench/dictionary.txt'

def downloadWords(save_path = 'data'):
    r = requests.get(words_html)

    with open(save_path + '/Brainbench_words.txt', 'w') as f:
        f.write(r.text)

def filteredVec(model_path):
    model = np.load(model_path).item()
    model_words = list(model.keys())

    # get vectors corresponding to these words and save to txt format
    with open('data/FilteredVectors/' + model_path.split('/')[-1][:-4] + '.txt', 'w') as f:
        for word in model_words:
            string = word + ' ' + " ".join(str(x) for x in model[word])
            f.write(string.replace('\n', '') + '\n')

if __name__ == '__main__':
    # Set command line arguments
    parser = argparse.ArgumentParser(description = 'build brainbench compatable model.')

    parser.add_argument('--model', '-m1', action = 'store',
                        required = True,
                        help = ('Input path for model to correctly format.'))

    args = parser.parse_args()
    
    if not os.path.exists('data'):
        os.path.exists('data')
        os.path.exists('data/FilteredVectors')

    # check if brainbench dictionary is downloaded and if not, do so
    if not os.path.isfile('data/Brainbench_words.txt'):
        downloadWords()

    filteredVec(args.model)
    print('data/FilteredVectors/' + args.model.split('/')[-1][:-4] + '.txt')
