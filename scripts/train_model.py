'''
Train and save the model.
Take in training data src path (csv) and model name as command line arguments.
Used in 'train_all.sh' shell script.

ex. usage - python3 train_model.py ../data/loan-skew.csv.train loan_skew;
'''
import sys

import numpy as np 
import pandas as pd 
import xgboost as xgb

def load_training_data(src_path, rank, num_workers):
    dataset = pd.read_csv(src_path, delimiter=',', header=0)
    data, label = dataset.iloc[:,:-1], dataset.iloc[:,-1:]
    full_dtrain = xgb.DMatrix(data, label=label)

    if num_workers == 1:
        print('not distributed - loading full training set onto singular node')
        return full_dtrain

    if rank == 0:
        worker_dtrain = xgb.DMatrix(np.empty((1, 45)), label=np.ones((1, 1)))
    else:
        total_rows = full_dtrain.num_row()
        rows_per_worker = total_rows // num_workers
        
        start_dex = (rank - 1) * rows_per_worker
        end_dex = start_dex + rows_per_worker if rank != num_workers - 1 else total_rows

        worker_dtrain = full_dtrain.slice([i for i in range(start_dex, end_dex)])

    print("Mean label value for rank", rank, "=", np.mean(worker_dtrain.get_label(), dtype=np.float64))
    return worker_dtrain

def train(src_path, model_name):
    rank = xgb.rabit.get_rank()
    num_workers = xgb.rabit.get_world_size()

    dtrain = load_training_data(src_path, rank, num_workers)
    params = {'max_depth': 3, # default = 6
              'alpha': 0,  # default = 0
              'lambda': 1, # default = 1
              'eta': 0.3,  # default = 0.3
              'gamma': 0.3, # default = 0
              'objective': 'binary:hinge'
            }
    num_rounds = 25

    model = xgb.train(params, dtrain, num_rounds)
    model.save_model(''.join(['../models/', str(model_name), '.model']))
    model.dump_model(''.join(['../models/', str(model_name), '.txt_dump']))
    print('saved and dumped model for', model_name)
    
def main(argv):
    assert(len(argv) >= 3), "This script takes in source path of training data and model name."
    xgb.rabit.init()
    src_path = argv[1]
    model_name = argv[2]
    train(src_path, model_name)
    xgb.rabit.finalize()

if __name__ == '__main__':
    main(sys.argv)
