'''
This script takes in the destination file path and columns to skew by command line argument.
Used in 'generate_skews.sh' shell script.

ex usage: python3 sort_training_data_by_columns.py ../data/sorted_data_by_age_and_job.csv age job
'''
import sys

import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder

def read_dataset():
    return pd.read_csv('../data/bank-full-transformed.csv.train', 
                        delimiter=',', 
                        header=0)

def skew_dataset(dataset, cols):
    '''
    Sort dataset by cols

    Params:
    dataset - pd.DataFrame
    cols - List[str]
    '''
    skewed = dataset.sort_values(by=cols)
    return skewed

def main(argv):
    assert(len(argv) >= 3), "This script should take in at least two cmdline args: destination file path and column to sort by."
    dest_path = argv[1]
    cols = argv[2:]
    raw_dataset = read_dataset()
    skewed_dataset = skew_dataset(raw_dataset, cols)
    skewed_dataset.to_csv(dest_path, index=False)


if __name__ == '__main__':
    main(sys.argv)