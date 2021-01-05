import numpy as np
import sys

def getFiltPath():
    path = sys.argv[1]

    print('data/FilteredVectors/' + path.split('/')[-1][:-4] + '_flt.txt')

if __name__ == '__main__':
    getFiltPath()
