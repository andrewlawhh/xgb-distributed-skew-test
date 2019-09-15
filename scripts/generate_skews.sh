#!/bin/bash

# generate skews on all non-categorical values
python3 sort_training_data_by_columns.py ../data/age-skew.csv.train age;
python3 sort_training_data_by_columns.py ../data/default-skew.csv.train default;
python3 sort_training_data_by_columns.py ../data/balance-skew.csv.train balance;
python3 sort_training_data_by_columns.py ../data/housing-skew.csv.train housing;
python3 sort_training_data_by_columns.py ../data/loan-skew.csv.train loan;
python3 sort_training_data_by_columns.py ../data/duration-skew.csv.train duration;
python3 sort_training_data_by_columns.py ../data/campaign-skew.csv.train campaign;
python3 sort_training_data_by_columns.py ../data/pdays-skew.csv.train pdays;
python3 sort_training_data_by_columns.py ../data/previous-skew.csv.train previous;
python3 sort_training_data_by_columns.py ../data/poutcome-skew.csv.train poutcome;
