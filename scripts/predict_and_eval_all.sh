#!/bin/bash

# predict and eval - control
python3 predict_and_eval.py ../models/bank_no_skew.model;

# predict and eval - skews
python3 predict_and_eval.py ../models/bank_label_skew.model;
python3 predict_and_eval.py ../models/age_skew.model;
python3 predict_and_eval.py ../models/default_skew.model;
python3 predict_and_eval.py ../models/balance_skew.model;
python3 predict_and_eval.py ../models/housing_skew.model;
python3 predict_and_eval.py ../models/loan_skew.model;
python3 predict_and_eval.py ../models/duration_skew.model;
python3 predict_and_eval.py ../models/campaign_skew.model;
python3 predict_and_eval.py ../models/pdays_skew.model;
python3 predict_and_eval.py ../models/previous_skew.model;
python3 predict_and_eval.py ../models/poutcome_skew.model;
python3 predict_and_eval.py ../models/job_skew.model;
python3 predict_and_eval.py ../models/marital_skew.model;
python3 predict_and_eval.py ../models/education_skew.model;
python3 predict_and_eval.py ../models/contact_skew.model;
python3 predict_and_eval.py ../models/month_skew.model;
