#!/bin/bash

# train control (no skew)
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/bank-full-transformed.csv.train bank_no_skew;

# train label skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/bank-full-transformed-label-skew.csv.train bank_label_skew;

# train age skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/age-skew.csv.train age_skew;

# train default skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/default-skew.csv.train default_skew;

# train balance skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/balance-skew.csv.train balance_skew;

# train housing skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/housing-skew.csv.train housing_skew;

# train loan skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/loan-skew.csv.train loan_skew;

# train duration skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/duration-skew.csv.train duration_skew;

# train campaign skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/campaign-skew.csv.train campaign_skew;

# train pdays skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/pdays-skew.csv.train pdays_skew;

# train previous skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/previous-skew.csv.train previous_skew;

# train poutcome skew data partition
../../dmlc-core/tracker/dmlc-submit --cluster local --num-workers $1 \
python3 train_model.py ../data/poutcome-skew.csv.train poutcome_skew;



