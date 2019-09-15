'''
Split original csv into training and test partitions. (note - no validation set since I'm just trying to experiment with column skews.)
'''
import sys 

import numpy as np 
import pandas as pd

def main(argv):
    file_name = argv[1]

    dataset = pd.read_csv(file_name, delimiter=',', header=0)
    total_rows = len(dataset)
    train_set_size = int(.7 * total_rows)

    train = dataset.iloc[:train_set_size, :]
    assert(len(train) == train_set_size)

    test = dataset.iloc[train_set_size:, :]
    assert(len(test) == total_rows - train_set_size)

    train.to_csv(file_name + '.train', index=False)
    test.to_csv(file_name + '.test', index=False)

if __name__ == '__main__':
    main(sys.argv)