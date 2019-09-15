# Investigate effects of data skew across worker machines on distributed XGBoost

#### Directory Structure
The ./predictions/ directory contains predictions and evaluations of each setting.
The ./models/ directory contains XGBoost model binaries as well as human readable representations of each decision tree.

#### Experiment Information
Each model was trained using the same training parameters, found in the training function of ./scripts/train_model.py and
evaluated by the same metrics, namely (# wrong predictions)/(# total predictions).

The 'skew' on these non-categorical variables was performed by first sorting the dataset by that column, and then partitioning to
each worker machine in blocks of (1/num_workers).

The code for 'skew'-ing the data can be found in ./scripts/sort_training_data_by_columns.py, and the code for partitioning to each worker
can be found in ./scripts/train_model.py

This dataset was downloaded from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing#)

##### Results (run on local distributed XGBoost over 4 worker nodes and 1 tracker with dummy data)

**Control (No skew)**
eval-error:0.245060
{'false_positives': 490, 'false_negatives': 2834, 'total predictions': 13564}

**Label Skew **
eval-error:0.246461
{'false_positives': 501, 'false_negatives': 2842, 'total predictions': 13564}

**Age Skew**
eval-error:0.249115
{'false_positives': 465, 'false_negatives': 2914, 'total predictions': 13564}

**Balance Skew**
eval-error:0.246093
{'false_positives': 479, 'false_negatives': 2859, 'total predictions': 13564}

**Campaign Skew**
eval-error:0.246166
{'false_positives': 507, 'false_negatives': 2832, 'total predictions': 13564}

**Default Skew**
eval-error:0.249853
{'false_positives': 434, 'false_negatives': 2955, 'total predictions': 13564}

**Duration Skew**
eval-error:0.251401
{'false_positives': 462, 'false_negatives': 2948, 'total predictions': 13564}

**Housing Skew**
eval-error:0.244618
{'false_positives': 465, 'false_negatives': 2853, 'total predictions': 13564}

**Loan Skew**
eval-error:0.247272
{'false_positives': 429, 'false_negatives': 2925, 'total predictions': 13564}

**PDays Skew**
eval-error:0.244471
{'false_positives': 509, 'false_negatives': 2807, 'total predictions': 13564}

**POutcome Skew**
eval-error:0.249926
{'false_positives': 487, 'false_negatives': 2903, 'total predictions': 13564}

**Previous Skew**
eval-error:0.244471
{'false_positives': 509, 'false_negatives': 2807, 'total predictions': 13564}

