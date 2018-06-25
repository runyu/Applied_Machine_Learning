#reference: 
#https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm/code
#https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

#dev variables
path = '/Users/runyuwang/Dropbox/MITB/Term 3/Applied Machine Learning/Project/Dataset/'
#path = '../input/'
file_percent = 0.001

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
    return df


def import_data(file):
    #create a dataframe and optimize its memory usage"""
    print(file)
    print('-' * 50)
    num_line = sum((1 for i in open(file, 'rb')))
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True, nrows= int(num_line * file_percent))
    df = reduce_mem_usage(df)
    print('shape', df.shape, '\n')
    return df


def process_bureau_bal(bureau_bal):
    #create dummy columns for STATUS
    bureau_bal_dum = pd.get_dummies(bureau_bal.STATUS, prefix='bureau_bal_status_')
    bureau_bal_concat = pd.concat([bureau_bal, bureau_bal_dum], axis=1)
    bureau_bal = bureau_bal_concat.drop('STATUS', axis=1)

    #counting bureaus
    bureau_bal_sub = bureau_bal[['SK_ID_BUREAU', 'MONTHS_BALANCE']]
    bureau_bal_counts = bureau_bal_sub.groupby('SK_ID_BUREAU').count()    
    bureau_bal['bureau_bal_count'] = bureau_bal['SK_ID_BUREAU'].map(bureau_bal_counts['MONTHS_BALANCE'])

    #averaging bureau bal
    avg_bureau_bal = bureau_bal.groupby('SK_ID_BUREAU').mean()
    avg_bureau_bal.columns = ['avg_bureau_bal_' + f_ for f_ in avg_bureau_bal.columns]
    
    return avg_bureau_bal

bureau_bal = import_data(path + 'bureau_balance.csv')
bureau_bal = process_bureau_bal(bureau_bal)

def process_bureau(bureau):
    #create dummy columns for three cols
    bureau_cr_act_dum = pd.get_dummies(bureau.CREDIT_ACTIVE, prefix='cr_act_')
    bureau_cr_ccy_dum = pd.get_dummies(bureau.CREDIT_CURRENCY, prefix='cr_ccy_')
    bureau_cr_typ_dum = pd.get_dummies(bureau.CREDIT_TYPE, prefix='cr_typ_')
    
    bureau_dum = pd.concat([bureau, bureau_cr_act_dum, bureau_cr_ccy_dum, bureau_cr_typ_dum], axis=1)
    bureau_dum = bureau_dum.drop(['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE'], axis=1)
    
    #no averaging here
    return bureau_dum

bureau = import_data(path + 'bureau.csv')
bureau = process_bureau(bureau)

def merge_bureau_bureau_bal(bureau, bureau_bal):
    #merge bureau and burea_bal
    bureau_full = bureau.merge(right= bureau_bal.reset_index(), how= 'left', on = 'SK_ID_BUREAU', suffixes = ('','_bur_bal'))
    
    #counting number of bureau for every SK_ID_CURR
    bureau_full_sub = bureau_full[['SK_ID_CURR', 'SK_ID_BUREAU']]
    count_bureau_per_curr = bureau_full_sub.groupby('SK_ID_CURR').count()
    bureau_full['count_bureau_per_curr'] = bureau_full['SK_ID_CURR'].map(count_bureau_per_curr['SK_ID_BUREAU'])
    
    #drop SK_ID_BUREAU, SK_ID_BUREAU should not appear here. one row represents one SK_ID_CURR. 
    #one SK_ID_CURR can have multiple SK_ID_BUREAU
    bureau_full = bureau_full.drop('SK_ID_BUREAU', axis=1)
    
    #averaging bureau
    avg_bureau = bureau_full.groupby('SK_ID_CURR').mean()
    avg_bureau.columns = ['avg_bureau_' + f_ for f_ in avg_bureau.columns]
    return avg_bureau
    
avg_bureau = merge_bureau_bureau_bal(bureau, bureau_bal)
del bureau
del bureau_bal
gc.collect()

def process_prev(prev_app):
    #create dummies columns
    prev_cat_features = [
        f_ for f_ in prev_app.columns if prev_app[f_].dtype == 'object'
    ]
    
    prev_dum = pd.DataFrame()
    for f_ in prev_cat_features:
        prev_dum = pd.concat([prev_dum, pd.get_dummies(prev_app[f_], prefix=f_).astype(np.uint8)], axis=1)
    
    prev_app = pd.concat([prev_app, prev_dum], axis=1)
     
    #count number of prev for every SK_ID_CURR
    count_prev_per_curr = prev_app[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev_app['count_prev_per_curr'] = prev_app['SK_ID_CURR'].map(count_prev_per_curr['SK_ID_PREV'])   
    prev_app = prev_app.drop('SK_ID_PREV', axis = 1)
    
    #averaging prev
    avg_prev_app = prev_app.groupby('SK_ID_CURR').mean()
    avg_prev_app.columns = ['avg_prev_app_' + f_ for f_ in avg_prev_app.columns]
    
    return avg_prev_app

prev_app = import_data(path + 'previous_application.csv')
avg_prev_app = process_prev(prev_app)
del prev_app
gc.collect()

def process_pos_cash(pos_cash):
    #create dummies column
    pos_cash_dum = pd.get_dummies(pos_cash['NAME_CONTRACT_STATUS'], prefix = 'pos_cash_name_contract_status_')
    pos_cash = pd.concat([pos_cash, pos_cash_dum], axis=1)
    
    #count number of prev per curr
    count_prev_per_curr = pos_cash[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    pos_cash['count_prev_per_curr'] = pos_cash['SK_ID_CURR'].map(count_prev_per_curr['SK_ID_PREV'])
    pos_cash = pos_cash.drop('SK_ID_PREV', axis = 1)
    
    #averaging pos cash
    avg_pos_cash = pos_cash.groupby('SK_ID_CURR').mean()
    avg_pos_cash.columns = ['avg_pos_cash_' + f_ for f_ in avg_pos_cash.columns]
    
    return avg_pos_cash

pos_cash = import_data(path + 'POS_CASH_balance.csv')
avg_pos_cash = process_pos_cash(pos_cash)
del pos_cash
gc.collect()

def process_cc_bal(cc_bal): 
    #create dummies column
    cc_bal_dum = pd.get_dummies(cc_bal['NAME_CONTRACT_STATUS'], prefix='cc_bal_name_contract_status_')
    cc_bal = pd.concat([cc_bal, cc_bal_dum], axis=1)
    
    #count number of prev per curr
    count_prev_per_curr = cc_bal[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    cc_bal['count_prev_per_curr'] = cc_bal['SK_ID_CURR'].map(count_prev_per_curr['SK_ID_PREV'])
    cc_bal = cc_bal.drop('SK_ID_PREV', axis = 1)
    
    #compute average
    avg_cc_bal = cc_bal.groupby('SK_ID_CURR').mean()
    avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]
    return avg_cc_bal

cc_bal = import_data(path + 'credit_card_balance.csv')
avg_cc_bal = process_cc_bal(cc_bal)
del cc_bal
gc.collect()


def process_inst_pay(inst_pay):
    count_prev_per_curr = inst_pay[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    inst_pay['count_prev_per_curr'] = inst_pay['SK_ID_CURR'].map(count_prev_per_curr['SK_ID_PREV'])
    inst_pay = inst_pay.drop('SK_ID_PREV', axis = 1)
    
    avg_inst_pay = inst_pay.groupby('SK_ID_CURR').mean()
    avg_inst_pay.columns = ['inst_pay_' + f_ for f_ in avg_inst_pay.columns]
    return avg_inst_pay
    
inst_pay = import_data(path + 'installments_payments.csv')
avg_inst_pay = process_inst_pay(inst_pay)

del inst_pay
gc.collect()

train = import_data(path + 'application_train.csv')
test = import_data(path + 'application_test.csv')

y = train['TARGET']
del train['TARGET']

categorical_feats = [
    f for f in train.columns if train[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    train[f_], indexer = pd.factorize(train[f_])
    test[f_] = indexer.get_indexer(test[f_])

train = train.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_bureau.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_prev_app.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_prev_app.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_pos_cash.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_pos_cash.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

train = train.merge(right=avg_inst_pay.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_inst_pay.reset_index(), how='left', on='SK_ID_CURR')

del avg_bureau, avg_prev_app, avg_pos_cash, avg_cc_bal, avg_inst_pay
gc.collect()


