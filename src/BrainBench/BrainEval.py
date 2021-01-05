import numpy as np
import two_vs_two
import sys
import scipy.io as sio
import h5py
from scipy.stats.stats import pearsonr
import os

path = 'data/BrainScore.npy'

def printScores(scores):
    # function to print score similar to brainbench
    print("There are %d words found in BrainBench from the input" % scores[2])
    print("The fMRI score is %f" % scores[0])
    print("The MEG score is %f" % scores[1])

def checkDict():
    # check if brainscore dictionary exists, if not then create it
    if not os.path.exists(path):
        np.save(path, {})


def main():
    model_path = sys.argv[1]
    scores = two_vs_two.run_test(open(model_path, 'r'))
    printScores(scores)
    mod_dict = np.load(path).item()
    mod_dict[model_path.split('/')[-1][:-8]] = [scores[2], scores[0], scores[1]]

    np.save(path, mod_dict)

if __name__ == '__main__':
    checkDict()
    main()
