#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:18:46 2018

@author: runyuwang
"""

# dev variables
path = '/Users/runyuwang/Dropbox/MITB/Term 3/Applied Machine Learning/Project/Dataset/'
#path = ''../input/'
file_percent = 0.0001

import pandas as pd
import numpy as np
import gc


def reduce_mem_usage(df):
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    
    return df


def import_data(file):
    #create a dataframe and optimize its memory usage"""
    print(file)
    print('-' * 50)
    num_line = sum((1 for i in open(file, 'rb')))
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, nrows= int(num_line * file_percent))
    df = reduce_mem_usage(df)
    print('shape', df.shape)
    return df


def merge_table(left_table, right_table, key):
    temp_table = left_table.merge(right_table, left_on = key, right_on = key, how = 'left')
    
    return temp_table

def process_bureau(bureau):
    #create dummy columns for three cols
    bureau_credit_active_dum = pd.get_dummies(bureau.CREDIT_ACTIVE, prefix='cr_act_')
    bureau_credit_currency_dum = pd.get_dummies(bureau.CREDIT_CURRENCY, prefix='cr_ccy_')
    bureau_credit_type_dum = pd.get_dummies(bureau.CREDIT_TYPE, prefix='cr_type_')
    
    bureau_full = pd.concat([bureau, bureau_credit_active_dum, bureau_credit_currency_dum, bureau_credit_type_dum], axis=1)
    bureau_full = bureau_full.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)
    
    #check the the result and the meaning
    #compute the avg
    nb_bureau_per_curr = bureau_full[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()
    bureau_full['SK_ID_BUREAU'] = bureau_full['SK_ID_CURR'].map(nb_bureau_per_curr['SK_ID_BUREAU'])
    
    avg_bureau = bureau_full.groupby('SK_ID_CURR').mean()
        
    return avg_bureau

def process_bureau_bal(bureau_bal):
    bureau_bal_dummies = pd.get_dummies(bureau_bal.STATUS, prefix='bureau_bal_status')
    bureau_bal_concat = pd.concat([bureau_bal, bureau_bal_dummies], axis=1)
    bureau_bal = bureau_bal_concat.drop('STATUS', axis=1)

    #counting bureaus
    bureau_bal_sub = bureau_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']]
    bureau_counts = bureau_bal_sub.groupby('SK_ID_BUREAU').count()
    bureau_bal['buro_count'] = bureau_bal['SK_ID_BUREAU'].map(bureau_counts['MONTHS_BALANCE'])

    #averaging bureau bal
    avg_bureau_bal = bureau_bal.groupby('SK_ID_BUREAU',as_index=False).mean()

    avg_bureau_bal.columns = ['avg_buro_' + f_ for f_ in avg_bureau_bal.columns]
    
    return avg_bureau_bal

train = import_data(path + 'application_train.csv')
bureau = import_data(path + 'bureau.csv')
#bureau = process_bureau(bureau)
temp_table = merge_table(train,bureau,'SK_ID_CURR')
#del train
#del bureau
#gc.collect()

bureau_bal = import_data(path + 'bureau_balance.csv')
bureau_bal = process_bureau_bal(bureau_bal)
temp_table= merge_table(temp_table,bureau_bal,'SK_ID_BUREAU')
#del bureau_bal 
#gc.collect()

previous_app = import_data(path + 'previous_application.csv')
temp_table= merge_table(temp_table,previous_app,'SK_ID_CURR')
#del previous_app
#gc.collect()

pos_cash = import_data(path + 'POS_CASH_balance.csv')
temp_table= merge_table(temp_table,pos_cash,'SK_ID_PREV')
#del pos_cash
#gc.collect()

installments = import_data(path + 'installments_payments.csv')
temp_table= merge_table(temp_table,installments,'SK_ID_PREV')
#del installments
#gc.collect()

credit_card = import_data(path + 'credit_card_balance.csv')
temp_table= merge_table(temp_table,credit_card,'SK_ID_PREV')
#del credit_card
#gc.collect()

