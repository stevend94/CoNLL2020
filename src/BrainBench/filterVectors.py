import sys

dictionary = {}

for line in open(sys.argv[1], 'r'):
    dictionary[line.strip()] = 0

for line in sys.stdin:
    if line.strip().split()[0] in dictionary: 
        print(line.strip())
