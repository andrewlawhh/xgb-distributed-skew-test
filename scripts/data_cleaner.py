'''
This script transforms the data (bank marketing dataset from UCI ML database) https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#
to be more usable for training. For instance - replacing 'yes', 'no', 'success', 'failure' with 1 or 0 integer values, and replacing
'unknown' values to np.nan which XGBoost can understand as an 'empty' cell.

Additionally, one hot encoding was performed on categorical variables since XGBoost cannot handle categorical variables.
'''
import numpy as np
import pandas as pd
from category_encoders.one_hot import OneHotEncoder

def read_dataset():
    return pd.read_csv('../data/bank-full.csv', delimiter=',', header=0)

def transform_dataset(dataset : pd.DataFrame):
    enc = OneHotEncoder()
    cleaned_dataset = dataset.replace({'yes': 1,
                                     'no': 0,
                                     'success': 1,
                                     'failure': 0,
                                     'unknown': np.nan,
                                     'other': np.nan,
                                    })
    transformed = enc.fit_transform(cleaned_dataset)
    print(transformed)
    return transformed

def main():
    raw_dataset = read_dataset()
    transformed_dataset = transform_dataset(raw_dataset)
    transformed_dataset.to_csv('../data/bank-full-transformed.csv', 
                                index=False)


if __name__ == '__main__':
    main()