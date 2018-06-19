#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 17:52:22 2018

@author: runyuwang
"""

import pandas as pd

# for code prototype, we use 20 percentage of data. 
# in the actual testing, we use full set of data
def read_file(file_name):
    num_line = sum((1 for i in open(file_name, 'rb')))
    return pd.read_csv(file_name, nrows= int(num_line/100)) 

app_train = read_file('application_train.csv')
bureau_bal = read_file('bureau_balance.csv')
bureau = read_file('bureau.csv')
credit_card_bal = read_file('credit_card_balance.csv')
pos_cash_bal = read_file('POS_CASH_balance.csv')
installments_pay = read_file('installments_payments.csv')
prev_app = read_file('previous_application.csv')
sample_sub = read_file('sample_submission.csv')

# app_test = pd.read_csv('application_test.csv')
homecredit_col_desc = pd.read_csv('HomeCredit_columns_description.csv', encoding='latin1')

print(app_train.groupby('TARGET').size())

temp_table = app_train.merge(bureau, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how = 'left')
temp_table = temp_table.merge(bureau_bal,left_on='SK_ID_BUREAU', right_on='SK_ID_BUREAU', how = 'left')

temp_table = temp_table.merge(prev_app, left_on='SK_ID_CURR', right_on='SK_ID_CURR', how = 'left')
temp_table = temp_table.merge(pos_cash_bal,left_on='SK_ID_PREV', right_on='SK_ID_PREV', how = 'left')
temp_table = temp_table.merge(installments_pay,left_on='SK_ID_PREV', right_on='SK_ID_PREV', how = 'left')
temp_table = temp_table.merge(credit_card_bal,left_on='SK_ID_PREV', right_on='SK_ID_PREV', how = 'left')