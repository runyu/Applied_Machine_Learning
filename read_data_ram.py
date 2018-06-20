#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 20:18:46 2018

@author: runyuwang
"""

# dev variables
path = '/Users/runyuwang/Dropbox/MITB/Term 3/Applied Machine Learning/Project/Dataset/'
#path = ''../input/'
file_percent = 0.01

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


def merge_table(l_table, r_table, l_key, r_key, l_suffixes, r_suffixes):
    merged_table = l_table.merge(r_table, left_on = l_key, right_on = r_key, how = 'left',suffixes=(l_suffixes, r_suffixes))
    return merged_table

def process_bureau(bureau):
    #create dummy columns for three cols
    bureau_cr_act_dum = pd.get_dummies(bureau.CREDIT_ACTIVE, prefix='cr_act_')
    bureau_cr_ccy_dum = pd.get_dummies(bureau.CREDIT_CURRENCY, prefix='cr_ccy_')
    bureau_cr_typ_dum = pd.get_dummies(bureau.CREDIT_TYPE, prefix='cr_typ_')
    
    bureau_dum = pd.concat([bureau, bureau_cr_act_dum, bureau_cr_ccy_dum, bureau_cr_typ_dum], axis=1)
    bureau_dum = bureau_dum.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)
    
    #no averaging here
    return bureau_dum

def process_bureau_bal(bureau_bal):
    #create dummy columns for STATUS
    bureau_bal_dum = pd.get_dummies(bureau_bal.STATUS, prefix='bureau_bal_status')
    bureau_bal_concat = pd.concat([bureau_bal, bureau_bal_dum], axis=1)
    bureau_bal = bureau_bal_concat.drop('STATUS', axis=1)

    #counting bureaus
    bureau_bal_sub = bureau_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']]
    bureau_bal_counts = bureau_bal_sub.groupby('SK_ID_BUREAU').count()    
    bureau_bal['bureau_bal_count'] = bureau_bal['SK_ID_BUREAU'].map(bureau_bal_counts['MONTHS_BALANCE'])

    #averaging bureau bal
    avg_bureau_bal = bureau_bal.groupby('SK_ID_BUREAU',as_index=False).mean()
    avg_bureau_bal.columns = ['avg_bureau_' + f_ for f_ in avg_bureau_bal.columns]
    
    return avg_bureau_bal

def merge_bureau_bureau_bal(bureau, bureau_bal):
    #merge bureau and burea_bal
    bureau_full = merge_table(bureau, bureau_bal, 'SK_ID_BUREAU', 'avg_bureau_SK_ID_BUREAU','','_bur_bal')
    
    #counting bureau per SK_ID_CURR
    bureau_full_sub = bureau_full[['SK_ID_CURR', 'SK_ID_BUREAU']]
    count_bureau_per_curr = bureau_full_sub.groupby('SK_ID_CURR').count()
    bureau_full['count_bureau_per_curr'] = bureau_full['SK_ID_CURR'].map(count_bureau_per_curr['SK_ID_BUREAU'])
    
    #drop SK_ID_BUREAU, the meaning changes
    bureau_full = bureau_full.drop('SK_ID_BUREAU', axis=1)
    
    #averaging bureau
    avg_bureau = bureau_full.groupby('SK_ID_CURR',as_index=False).mean()
    return avg_bureau
    



bureau = import_data(path + 'bureau.csv')
bureau = process_bureau(bureau)

bureau_bal = import_data(path + 'bureau_balance.csv')
bureau_bal = process_bureau_bal(bureau_bal)

bureau_full = merge_bureau_bureau_bal(bureau, bureau_bal)




train = import_data(path + 'application_train.csv')
bureau = import_data(path + 'bureau.csv')
bureau = process_bureau(bureau)
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

