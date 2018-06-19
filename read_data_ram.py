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

corr_threshold = 0.8

def remove_high_correlation(df):

    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]  
    df = df.drop(to_drop, axis=1)  
    return df


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

    print('shape', df.shape)
    df = remove_high_correlation(df)
    return df


def import_data(file):
    #create a dataframe and optimize its memory usage"""
    print(file)
    print('-' * 50)
    num_line = sum((1 for i in open(file, 'rb')))
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, nrows= int(num_line * file_percent))
    df = reduce_mem_usage(df)
    print(df.shape)
    return df


def merge_table(left_table, right_table, key):
    temp_table = left_table.merge(right_table, left_on = key, right_on = key, how = 'left')
    
    return temp_table


train = import_data(path + 'application_train.csv')
bureau = import_data(path + 'bureau.csv')
temp_table = merge_table(train,bureau,'SK_ID_CURR')
#del train
#del bureau

bureau_balance = import_data(path + 'bureau_balance.csv')
temp_table= merge_table(temp_table,bureau_balance,'SK_ID_BUREAU')
#del bureau_balance 

previous_app = import_data(path + 'previous_application.csv')
temp_table= merge_table(temp_table,previous_app,'SK_ID_CURR')
#del previous_app

pos_cash = import_data(path + 'POS_CASH_balance.csv')
temp_table= merge_table(temp_table,pos_cash,'SK_ID_PREV')
#del pos_cash

installments = import_data(path + 'installments_payments.csv')
temp_table= merge_table(temp_table,installments,'SK_ID_PREV')
#del installments

credit_card = import_data(path + 'credit_card_balance.csv')
temp_table= merge_table(temp_table,credit_card,'SK_ID_PREV')
#del credit_card

