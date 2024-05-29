# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 00:31:36 2023

@author: alann
"""

import pandas as pd
import numpy as np
import pickle
import time
from pathlib import Path
import lightgbm
from lightgbm import LGBMRegressor
import joblib
import os
import gc
# =============================================================================
#                    Modelling and Prediction
# =============================================================================
# Model = LGBM
# Light GBM is a gradient boosting framework that uses tree based learning algorithm.
# Light GBM can handle the large size of data and takes lower memory to run. Another 
# reason of why Light GBM is popular is because it focuses on accuracy of results.

# it is not advisable to use LGBM on small datasets. Light GBM is sensitive to 
# overfitting and can easily overfit small data.

# use it only for data with 10,000+ rows.

# How it differs from other tree based algorithm?
# Light GBM grows tree vertically while other algorithm grows trees horizontally 
# meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. 
# It will choose the leaf with max delta loss to grow. When growing the same leaf, 
# Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

start_time = time.time()

# Import Data
# LIMITED RAM WE CANNOT LOAD 8GB DATA WITH  8GBRAM RAM---> TOO LARGE, WE GET MEMORY ERROR!!!
data_size = 'improved'
# path = r'C:\Users\alann\OneDrive\Documents\Python Scripts\retail_sales-forecasting\Data'
# data = pd.read_pickle(path + '/data_features_{}.pkl'.format(data_size))

# Set the path to the folder containing pickle files
folder_path = './chunks_folder'

# Get the list of files in the folder
file_list = os.listdir(folder_path)

# Initialize an empty list to store the loaded dataframes
dataframes = []

# Load pickle files and append them to the list
for file_name in file_list:
    if file_name.endswith('.pickle'):
        file_path = os.path.join(folder_path, file_name)
        
        # Load the pickle file in chunks
        df = pd.read_pickle(file_path)
        dataframes.append(df)
        
         # Optionally, append to the main DataFrame and then clear the list
        if len(dataframes) >= 10:  # For example, process 10 files at a time
            data = pd.concat(dataframes, ignore_index=True)
            dataframes = []  # Clear the list to free up memory
            gc.collect()  # Explicitly call garbage collection

# Concatenate any remaining dataframes
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
    dataframes = []  # Clear the list to free up memory
    gc.collect()  # Explicitly call garbage collection
data.info()

print("Dataframe shape: {}".format(data.shape))

# Start records from 55th
start_row = 55
# The time complexity for filtering a DataFrame based on a single column condition 
# is typically linear, or O(n), where n is the number of rows in the DataFrame.
# DataFrame is very large, the operation is computationally expensive.
# data = data[data['d'] >= start_row].copy()

# Solution
# lags introduce a lot of Null values, so I'll remove data for first 55 days as 
# I have introduced lags till 36 days.
data = data.loc[data['d'] >= start_row]
print('{:,} records for the prepared data'.format(len(data)))
data.info()

# Import key data
# Dictionaries of decoded data
folder_path = './Data'
LIST_NAME = ['d_id', 'd_item_id', 'd_dept_id', 'd_cat_id', 'd_store_id', 'd_state_id']
dict_data = {}

for list_name in LIST_NAME:
    dict_temp = pickle.load(open(folder_path + '\{}.p'.format(list_name), 'rb'))
    dict_data[list_name] = dict_temp
    del dict_temp
    

# SHOW FEATURES
# We'll split the features we created in groups so we can add them progressively 
# and measure the impact on the accuracy

# 1. Initial Features
INIT_FEAT = list(data.columns[0:21])


# 2.Lags and averages 
LAGAV_FEAT = list(data.columns[24:42])


# 3. Rolling Means and Rolling Means on lag
ROLLMEAN_FEAT = list(data.columns[42:52])


# 4. Trends and Rolling MAX
TREND_MAX_FEAT = list(data.columns[52:61])

# 5. Stock-Out and Store Closed
# SO_CLOSED_FEAT = list(data.columns[61:69])
SO_CLOSED_FEAT = list(['stock_out_id', 'store_closed'])

# 6. PRICE COMPARISON
PRICE_COMPARE = list(data.columns[21:24])

# Dictionary with different steps
dict_features = {
    'STEP_1': INIT_FEAT,
    'STEP_2': INIT_FEAT + LAGAV_FEAT,
    'STEP_3': INIT_FEAT + LAGAV_FEAT + ROLLMEAN_FEAT,
    'STEP_4': INIT_FEAT + LAGAV_FEAT + ROLLMEAN_FEAT + TREND_MAX_FEAT,
    'STEP_5': INIT_FEAT + LAGAV_FEAT + ROLLMEAN_FEAT + TREND_MAX_FEAT + SO_CLOSED_FEAT,
    'STEP_6': INIT_FEAT + LAGAV_FEAT + ROLLMEAN_FEAT + TREND_MAX_FEAT + SO_CLOSED_FEAT + PRICE_COMPARE  
    }

LIST_STEPS = ['STEP_1', 'STEP_2', 'STEP_3', 'STEP_4', 'STEP_5', 'STEP_6']
LIST_STEPS_NAME = ['INITIAL_DATA', 
                   'INITIAL + LAG + AVERAGES', 
                   'INITIAL + LAG + AVERAGES + ROLLING MEAN',
                   'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX',
                   'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX + STOCK-OUT AND STORE CLOSED',
                   'INITIAL + LAG AND AVERAGES + ROLLING MEAN + TREND AND ROLLING MAX + STOCK-OUT AND STORE CLOSED + PRICE COMPARISON']
dict_stepname = dict(zip(LIST_STEPS, LIST_STEPS_NAME))



#-----------TRAIN & TEST MODELS-------------#


# Create Validation and Test sets
# Splitting training & test data

# Validation Set
# valid = data[(data['d'] >= 1914) & (data['d'] < 1942) ][['id', 'd', 'sold']]
#valid = data.query("1914 <= d < 1942")[['id', 'd', 'sold']]

# Validation Prediction
# valid_preds = valid['sold']

# Test Set
# test = data[data['d'] >= 1942][['id', 'd', 'sold']]
# test = data.query("d >= 1942")[['id', 'd', 'sold']]

# Evaluation Prediction
# eval_preds = test['sold']



# Train and Test Models with different features
# Get the store ids
stores = data.store_id.unique()
d_store_id = dict_data['d_store_id']

# Dictionnary with errors each step
dict_error = {}


# Loop with the steps
for step in LIST_STEPS:
    # Folder for Models 
    print('*****Prediction for STEP: {}*****'.format(dict_stepname[step]))
    FOLDER_MODEL = './Model/{}/{}_Features_Improved/'.format(data_size, step)
    Path(FOLDER_MODEL).mkdir(parents=True, exist_ok=True)
    COLS_SCOPE = dict_features[step]
    
    # DataFrame with filter scope
    data_scope = data[COLS_SCOPE]
    
    # Validation Set
    # valid = data_scope[(data['d'] >= 1914) & (data['d'] < 1942) ][['id', 'd', 'sold']]
    valid = data_scope.query("1914 <= d < 1942")[['id', 'd', 'sold']]

    # Validation Prediction
    valid_set = valid['sold'] 
    valid_preds = valid['sold']
    
    # Test Set
    # test = data_scope[data['d'] >= 1942][['id', 'd', 'sold']]
    test = data_scope.query("d >= 1942")[['id', 'd', 'sold']]

    # Evaluation Prediction
    # eval_preds = test['sold']
    test_set = test['sold']
    
    # Validation + Predicition for all stores by step
    df_validpred = pd.DataFrame()
    
    # Test for prediction for next 28 days
    df_testpred = pd.DataFrame()
    
    
    
    # Loop for training a model for each store
    for store in stores:
        # Dataframe for each store
        # df = data_scope[data_scope['store_id'] == store]
        df = data_scope.query("store_id == @store")

        
        # Train Data until day = 1914
        # X_train, y_train = df[df['d'] < 1914].drop('sold',axis=1), df[df['d'] < 1914]['sold']
        X_train, y_train = df.query("d < 1914").drop('sold',axis=1), df.loc[df['d'] < 1914, 'sold']
        
        # Validation Day: 1914 to 1942
        # X_valid  = df[(df['d'] >= 1914) & (df['d'] < 1942)].drop('sold',axis=1)
        X_valid = df.query("1914 <= d < 1942").drop('sold', axis=1)
        # y_valid =  df[(df['d'] >= 1914) & (df['d'] < 1942)]['sold']
        y_valid = df.loc[(df['d'] >= 1914) & (df['d'] < 1942), 'sold']
        
        # X_test with 
        # X_test = df[df['d'] >= 1942].drop('sold', axis=1)
        X_test = df.query("d >= 1942").drop('sold', axis=1)
        
        
        
        # Train and validate
        model = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.3,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=8,
            num_leaves=50,
            min_child_weight=300
        )


        # Fit model
        # This parameter controls the verbosity of the training process. Setting 
        # it to 0 means silent mode, where no output is printed during training. 
        # If set to 1, it prints progress updates during training.
        model.fit(X_train, y_train, 
                  eval_set = [(X_train,y_train),(X_valid,y_valid)], 
                  eval_metric = 'rmse', 
                  verbose = 0, 
                  early_stopping_rounds = 20) 
        
        # Compute Prediction
        # The predicted values are then assigned to the valid_preds array at the 
        # corresponding indices obtained from X_valid.index.
        valid_pred = model.predict(X_valid)
        
        # The predicted values are assigned to the eval_preds array at the corresponding 
        # indices obtained from X_test.index.
        eval_pred = model.predict(X_test)
        
        # # Predict last 28 days
        # test_prediction = model.predict(X_test)
        
        # Test data 
        df_test = pd.DataFrame({
            'prediction_next28_days': eval_pred,
            'store': d_store_id[store]
            })
        
        # Actual Validation vs. Prediction
        df_valid = pd.DataFrame({
            'validation': valid_set[X_valid.index],
            'prediction': valid_pred,
            'store':d_store_id[store]
        })
        df_valid['error'] = df_valid['validation'] - df_valid['prediction']
        df_validpred = pd.concat([df_validpred, df_valid])
        df_testpred = pd.concat([df_testpred, df_test])
        
        # Save predictions by step
        df_valid.to_csv(FOLDER_MODEL + 'prediction_{}.csv'.format(step))
        df_test.to_csv(FOLDER_MODEL + 'test_{}.csv'.format(step))
        
        # Save model
        filename = FOLDER_MODEL + 'model_features_total-' + str(d_store_id[store]) + '.pkl'
        joblib.dump(model, filename)
        
        del model, X_train, y_train, X_valid, y_valid
    
    # Save Prediction for all stores
    df_validpred.to_csv(FOLDER_MODEL + 'prediction_{}.csv'.format(step))
    df_testpred.to_csv(FOLDER_MODEL + 'test_{}.csv'.format(step))
    
    # Compute Error
    valid_rmse = 100 * np.sqrt(
    np.mean((df_validpred.validation.values - df_validpred.prediction.values) ** 2)
    ) / np.mean(df_validpred.validation.values)
    # Add Error in a Dictionnary
    dict_error[step] = valid_rmse
    print("For {}: RMSE = {}".format(dict_stepname[step], valid_rmse))

# Final DataFrame with error for all stores by STEP
df_error = pd.DataFrame({
    'STEP': LIST_STEPS,
    'STEP_NAME': [dict_stepname[step] for step in LIST_STEPS],
    'rmse': [dict_error[step] for step in LIST_STEPS]
    })
df_error.to_excel(FOLDER_MODEL + 'df_error.xlsx')
df_error.head()




print("--- %s seconds ---" % (time.time() - start_time)) 
