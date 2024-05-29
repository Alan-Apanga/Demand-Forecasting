# -*- coding: utf-8 -*-
"""
Created on Sun May 28 22:42:26 2023

@author: alann
"""

import pandas as pd
import numpy as np
import pickle
import time
import os
#from pathlib import Path
#import lightgbm
#from lightgbm import LGBMRegressor
#import joblib


""" 
# Notebook inspired by the models shared:
# Link: https://www.kaggle.com/anshuls235/time-series-forecasting-eda-fe-modelling
# Link: https://www.kaggle.com/kneroma/m5-first-public-notebook-under-0-50/notebook


This analysis will be based on the M5 Forecasting dataset of Walmart stores sales 
records:

    - 1,913 days for the training set and 28 days for the evaluation set
    - 10 stores in 3 states (USA)
    - 3,049 unique in 10 stores
    - 3 main categories and 7 departments (sub-category)
    
The objective is to predict sales for all products in each store, in the following 
28 days right after the available dataset. We have to perform 30,490 forecasts 
for each day in the prediction horizon.

We’ll use the validation set to measure the performance of our model.

SUMMARY
I. Introduction
1. Data set
2. Initial Solution using LGBM
3. Features Analysis
II. Experiment
1. Additional features
2. Results
III. Conclusion and next steps

"""
start_time = time.time()
# =============================================================================
#               1. Fetch Data and Processing
# =============================================================================
# Sales Data: Training Set (1-1913) + Validation Set (1914-1941)
path = './product_segmentation/m5-forecasting-accuracy'
sales = pd.read_csv(path + '\sales_train_evaluation.csv')
sales.name = 'sales'
print("{:,} records for training data set".format(len(sales)))

# CALENDAR
calendar = pd.read_csv(path + '\calendar.csv')
calendar.name = 'calendar'
print('{:,} records for calendar data'.format(len(calendar)))

# SELLING PRICE
prices = pd.read_csv(path + '\sell_prices.csv')
prices.name = 'prices'
print("{:,} records for price data".format(len(prices)))

# ADD RECORDS FOR TESTING
for d in range(1942, 1970): 
    col = 'd_' + str(d)
    sales[col] = 0 
    sales[col] = sales[col].astype(np.int16)



# DOWNCASTING TO REDUCE MEMORY
# In this section I'll be downcasting the dataframes to reduce the amount of 
# storage used by them and also to expidite the operations performed on them.
# Refer to link above for more info

# Depending on your environment, pandas automatically creates int32, int64, 
# float32 or float64 columns for numeric ones. If you know the min or max value 
# of a column, you can use a subtype which is less memory consuming. You can also 
# use an unsigned subtype if there is no negative value.



def downcast(df):
    cols = df.dtypes.index.tolist()
    types = df.dtypes.values.tolist()
    np.object = object
    
    # Here are the different subtypes you can use: 
        # int8 / uint8 : consumes 1 byte of memory, range between -128/127 or 0/255
        # bool : consumes 1 byte, true or false
        # float16 / int16 / uint16: consumes 2 bytes of memory, range between -32768 and 32767 or 0/65535
        # float32 / int32 / uint32 : consumes 4 bytes of memory, range between -2147483648 and 2147483647
        # float64 / int64 / uint64: consumes 8 bytes of memory
    for i,t in enumerate(types):
        # Integer
        if 'int' in str(t):
            # Check if minimum and maximum are in the limit of int8
            if df[cols[i]].min() > np.iinfo(np.int8).min and df[cols[i]].max() < np.iinfo(np.int8).max:
                df[cols[i]] = df[cols[i]].astype(np.int8)
            # Check if minimum and maximum are in the limit of int16
            elif df[cols[i]].min() > np.iinfo(np.int16).min and df[cols[i]].max() < np.iinfo(np.int16).max:
                df[cols[i]] = df[cols[i]].astype(np.int16)
            # Check if minimum and maximum are in the limit of int32
            elif df[cols[i]].min() > np.iinfo(np.int32).min and df[cols[i]].max() < np.iinfo(np.int32).max:
                df[cols[i]] = df[cols[i]].astype(np.int32)
            # Choose int64
            else:
                df[cols[i]] = df[cols[i]].astype(np.int64)
        
        # Float
        elif 'float' in str(t):
            if df[cols[i]].min() > np.finfo(np.float16).min and df[cols[i]].max() < np.finfo(np.float16).max:
                df[cols[i]] = df[cols[i]].astype(np.float16)
            elif df[cols[i]].min() > np.finfo(np.float32).min and df[cols[i]].max() < np.finfo(np.float32).max:
                df[cols[i]] = df[cols[i]].astype(np.float32)
            else:
                df[cols[i]] = df[cols[i]].astype(np.float64)
        
        # Object
        elif t == np.object:
            if cols[i] == 'date':
                df[cols[i]] = pd.to_datetime(df[cols[i]], format='%Y-%m-%d')
            else:
                df[cols[i]] = df[cols[i]].astype('category')
    return df
                



# Additional Features: Pricing

# Relative difference between the current price of an item and its historical 
# average price, to highlight promotional offers’ impact.
    
# Price relative difference with the same item sold in other stores, to understand 
# whether or not the store has an attractive price.
    
# Price relative difference with other items sold in the same store and same product 
# category, to capture some cannibalization effects.

def improve_price():
    # Calculate Average price for all stores
    df_mean_store = pd.DataFrame(prices.groupby(['item_id', 'wm_yr_wk'])['sell_price'].mean())
    df_mean_store.columns = ['item_sells_price_avg']
    df_mean_store.reset_index(inplace = True)

    # Combine with calendar
    prices_new = pd.merge(prices, df_mean_store, on=['item_id', 'wm_yr_wk'], how='left', suffixes=('', '_y'))
    prices_new.drop(prices_new.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    # Price difference with same items in other stores
    prices_new['delta_price_all_rel'] = (prices_new['sell_price'] - 
                                         prices_new['item_sells_price_avg'])/prices_new['item_sells_price_avg']

    # Price difference with last week
    prices_new['item_store'] = prices_new['item_id'].astype(str) + '_' + prices_new['store_id'].astype(str)
    prices_new['item_store_change'] = prices_new["item_store"].shift() != prices_new["item_store"]
    # Price difference week n - week n-1
    prices_new['delta_price_weekn-1'] = (prices_new['sell_price']-
                                         prices_new['sell_price'].shift(1)).fillna(0)/prices_new['sell_price'].shift(1)
    prices_new['delta_price_weekn-1'] = prices_new['delta_price_weekn-1'].fillna(0) * (prices_new['item_store_change']==0)

    # Average price of the department by store
    prices_new['dept_id'] = prices_new.item_id.str[:-4]
    df_mean_cat = pd.DataFrame(prices_new.groupby(['dept_id', 'store_id', 'wm_yr_wk'])['sell_price'].mean())
    df_mean_cat.columns = ['dept_sells_price_avg']
    df_mean_cat.reset_index(inplace = True)
    # Combine with price dataset
    prices_new = pd.merge(prices_new, df_mean_cat, on=['dept_id', 'store_id', 'wm_yr_wk']
                          , how='left', suffixes=('', '_y'))
    prices_new.drop(prices_new.filter(regex='_y$').columns.tolist(),axis=1, inplace=True)

    # Cannibalisation: compare this item price with average of the department (category)
    prices_new['delta_price_cat_rel'] = (prices_new['sell_price'] - 
                                         prices_new['dept_sells_price_avg'])/prices_new['dept_sells_price_avg']                                               

    # Drop columns
    prices_new.drop(['item_sells_price_avg', 'item_store_change', 'item_store_change', 'dept_id', 'item_store',
                    'dept_sells_price_avg'], axis = 1, inplace = True)
    
    return prices_new


# Apply downcasting
sales = downcast(sales)      
calendar = downcast(calendar)  

prices = improve_price()
prices = downcast(prices)
 
# =============================================================================
#           2. Melt Data to reconstitute sales records                
# =============================================================================
# Melt
# Convert from wide to long format (records on row level)
# To make analysis of data in table easier, we can reshape the data into a more 
# computer-friendly form 
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], 
             var_name='d', value_name='sold').dropna()
print("{:,} records to combine with calendar and price".format(len(df)))
df.head()


# COMBINE DATA


# Merge / join. This returns only rows from left and right which share a common key
# If you specify how='left', then only keys from left are used, and missing data 
# from right is replaced by NaN.

# Combine with calendar
df = pd.merge(df, calendar, on='d', how='left')

# Combine with price
df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
print("{:,} records in df".format(len(df)))

del sales





"""                         Features Engineering     

# Time Series data must be re-framed as a supervised learning dataset before we 
# can start using machine learning algorithms:
    # There is no concept of input and output features in time series.
    # Instead, we must choose the variable to be predicted and use feature 
    # engineering to construct all of the inputs that will be used to make 
    # predictions for future time steps.  """

                

                
# =============================================================================
# # 1. LABEL ENCODING  
# =============================================================================
# Change Store id to category type
# Redundant
# df.store_id = df.store_id.astype('category')
# df.item_id = df.item_id.astype('category')
# df.cat_id = df.cat_id.astype('category')
# df.state_id = df.state_id.astype('category')
# df.id = df.id.astype('category')
# df.dept_id = df.dept_id.astype('category')

# Store the categories along with their codes
# d --> decoded
d_id = dict(zip(df.id.cat.codes, df.id))

# Item, Department and Categories
d_item_id = dict(zip(df.item_id.cat.codes, df.item_id))
d_dept_id = dict(zip(df.dept_id.cat.codes, df.dept_id))
d_cat_id = dict(zip(df.cat_id.cat.codes, df.cat_id))

# Stores and States
d_store_id = dict(zip(df.store_id.cat.codes, df.store_id))
d_state_id = dict(zip(df.state_id.cat.codes, df.state_id))

# Save as pickle file
path = './Data'
LIST_SAVE = [d_id, d_item_id, d_dept_id, d_cat_id, d_store_id, d_state_id]
LIST_NAME = ['d_id', 'd_item_id', 'd_dept_id', 'd_cat_id', 'd_store_id', 'd_state_id']
for list_save, list_name in zip(LIST_SAVE, LIST_NAME):
    # The wb indicates that the file is opened for writing in binary mode
    pickle.dump(list_save, open(path + '/{}.p'.format(list_name), 'wb'))
            

# MAPPING WITH CATEGORY CODES

# Remove d_ and transform to int (dates)   

# Takes a string x as input. It splits the string using the underscore ('_') 
# as a delimiter and returns the second element (index 1) of the resulting list.   
df.d = df['d'].apply(lambda x: x.split('_')[1]).astype(np.int16)
cols = df.dtypes.index.tolist()
types = df.dtypes.values.tolist()

# Transform categorical data to codes
for i,type in enumerate(types):
    if type.name == 'category':
        df[cols[i]] = df[cols[i]].cat.codes

# Drop Dates
df.drop('date',axis=1,inplace=True)
print("Dataframe shape: {}".format(df.shape))

df.info()
               
# =============================================================================
# 2. Introduce Lag              
# =============================================================================
# In the context of machine learning, lagging refers to a situation where the 
# input variables used for prediction or modeling are delayed or shifted in time 
# compared to the target variable. This delay or shift can occur due to various 
# reasons, such as data collection processes, real-time system constraints, or 
# inherent characteristics of the problem being modeled.

# Lagging is particularly relevant in time series analysis and forecasting tasks, 
# where the goal is to predict future values based on historical data. Time series 
# data often exhibits temporal dependencies, where the current value depends on 
# past values. However, in some cases, there may be a delay or lag between the 
# cause and effect relationship, or there might be a time gap before the impact 
# of certain factors is observed. In such situations, incorporating lagged variables 
# into the model can improve its performance by capturing the delayed dependencies.

# It's purely upto you how many lags you want to introduce.

# Introduce lags (days)
lags = [1, 2, 3, 7, 14, 28]

for lag in lags:
    df['sold_lag_'+str(lag)] = df.groupby(       
        ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        as_index= False
        )['sold'].shift(lag).astype(np.float16)


df.head()
                
                
# =============================================================================
# 3. Mean Encoding             
# =============================================================================
# From a mathematical point of view, mean encoding represents a probability of 
# your target variable, conditional on each value of the feature. In a way, it 
# embodies the target variable in its encoded value. I have calculated mean 
# encodings on the basis of following logical features :

# item
# state
# store
# category
# departmentbb

# category & department
# store & item
# category & item
# department & item
# state & store
# state, store and category
# store, category and department    

# Total Average Sales by: item, state, store, cat and dept
df['item_sold_avg'] = df.groupby('item_id')['sold'].transform('mean').astype(np.float16)
df['state_sold_avg'] = df.groupby('state_id')['sold'].transform('mean').astype(np.float16)
df['store_sold_avg'] = df.groupby('store_id')['sold'].transform('mean').astype(np.float16)
df['cat_sold_avg'] = df.groupby('cat_id')['sold'].transform('mean').astype(np.float16)
df['dept_sold_avg'] = df.groupby('dept_id')['sold'].transform('mean').astype(np.float16)          
                
# Sales average by 
df['cat_dept_sold_avg'] = df.groupby(['cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)
df['store_item_sold_avg'] = df.groupby(['store_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['cat_item_sold_avg'] = df.groupby(['cat_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['dept_item_sold_avg'] = df.groupby(['dept_id','item_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_sold_avg'] = df.groupby(['state_id','store_id'])['sold'].transform('mean').astype(np.float16)
df['state_store_cat_sold_avg'] = df.groupby(['state_id','store_id','cat_id'])['sold'].transform('mean').astype(np.float16)
df['store_cat_dept_sold_avg'] = df.groupby(['store_id','cat_id','dept_id'])['sold'].transform('mean').astype(np.float16)


# =============================================================================
# 4. Rolling Average (Window) on actual sales
# =============================================================================
# Days = 7
# We specify the window size as 3, which means that the rolling mean will be 
# calculated over the previous 3 values in the 'sold' column.
# Hence, we consider only the most recent values and ignore the past values
df['rolling_sold_mean'] = df.groupby(
    ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']    
    )['sold'].transform(
    lambda x: x.rolling(window=7).mean()
    ).astype(np.float16)

# Average for the last n days
for days in [3, 7, 14, 21, 28]:
    df['rolling_sold_mean_{}'.format(days)] = df.groupby(
    ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] 
    )['sold'].transform(
    lambda x: x.rolling(window=days).mean()   
    ).astype(np.float16)

        
# =============================================================================
# 5. Rolling Average (Window) on actual lag        
# =============================================================================
# rmean_7_7: rolling mean sales of a window size of 7 over column lag_7
# rmean_28_7: rolling mean sales of a window size of 28 over column lag_7
# rmean_7_28: rolling mean sales of a window size of 7 over column lag_28
# rmean_28_28: rolling mean sales of a window size of 28 over column lag_28


# Rolling Average on actual lag
for window, lag in zip([7, 7, 28, 28], [7, 28, 7, 28]):
    df['rolling_lag_{}_win_{}'.format(window, lag)] = df.groupby(
        ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        )['sold_lag_{}'.format(lag)].transform(
        lambda x: x.rolling(window=window).mean()    
        ).astype(np.float16)
df.head()


# =============================================================================
# 6. Trends
# =============================================================================
# I will be creating a selling trend feature, which will be some positive value 
# if the daily items sold are greater than the entire duration average ( d_1 - d_1969 ) 
# else negative

# By SKU (Store + Item Id)

# Daily Average by SKU (Item Id + Store)
df['daily_avg_sold'] = df.groupby(    
   ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd'] 
    )['sold'].transform('mean').astype(np.float16)


# Total Average by SKU (Item Id + Store)
df['avg_sold'] = df.groupby(
    ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']   
    )['sold'].transform('mean').astype(np.float16)


# Selling Trend
df['selling_trend'] = (df['daily_avg_sold'] - df['avg_sold']).astype(np.float16)

# Drop Columns 
df.drop(['daily_avg_sold', 'avg_sold'], axis=1, inplace=True)





"""              Additional Ideas for Features Engineering       """





# Trends by SKU for all stores
# By Item (All Stores)


# Daily Average by SKU (Item Id)
df['item_daily_avg_sold'] = df.groupby(
    # ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd']
    # store & state columns  excluded
    ['id', 'item_id', 'dept_id', 'cat_id', 'd']
    )['sold'].transform('mean').astype(np.float16)


# Total Average by SKU (Item Id)
df['item_avg_sold'] = df.groupby(
    # ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd']
    # store_id, state_id & d columns excluded
    ['id', 'item_id', 'dept_id', 'cat_id']   
    )['sold'].transform('mean').astype(np.float16)

# Selling Trend
df['item_selling_trend'] = (
    df['item_daily_avg_sold'] - df['item_avg_sold'] 
    ).astype(np.float16)





# Drop Columns 
df.drop(['item_daily_avg_sold', 'item_avg_sold'], axis=1, inplace=True)


# Rolling Average on actual lag
# for window, lag in zip([7, 7, 28, 28], [7, 28, 7, 28]):
#     df['rolling_lag_{}_win_{}'.format(window, lag)] = df.groupby(
#         ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
#         )['sold_lag_{}'.format(lag)].transform(
#         lambda x: x.rolling(window=window).mean()    
#         ).astype(np.float16)



# Rolling Max (Window): Last n days                     
df['rolling_sold_max'] = df.groupby(
    ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    )['sold'].transform(
    lambda x: x.rolling(window=7).max()
    ).astype(np.float16)


                  
# What is the maximum sales in the last the n days?
for days in [1, 2, 7, 14, 21, 28]:
    df['rolling_sold_max_{}'.format(days)] = df.groupby(
        ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        )['sold'].transform(
        lambda x: x.rolling(window=days).max()
        ).astype(np.float16) 
           
            





    # STOCK-OUTS: 
# Do you have stock availability issue that cause zero sales in the 
# last n days ?
# Assumption: we will suppose that if you do not have sales for 3 days in a row 
# you have a stock-out


# Sort the dataset by item_id and day
df.sort_values(['id', 'd'], ascending =[True, True], inplace = True)
df.head()

# Mapping id changed
# This code calculates a rolling sum of the previous 3 values in the 'zero_sale' 
# column and assigns 1 to the 'stock_out_id' column for each row where the 
# rolling sum equals 3, and 0 otherwise.
df['id_change'] = df['id'].diff().fillna(0)
print("{:,} unique id with {:,} id changes".format(df.id.nunique(), df['id_change'].sum()))

# Zero Sale 
# if condition met a value of 1 is stored in the zero_sale column
# '* 1' multiplies the boolean Series by 1, which converts 'True' values to 1 and 
# 'False' values to 0. This is done to convert the boolean values into integers.
df['zero_sale'] = (df['sold'] == 0) * (df['id_change']==False) * 1
df['stock_out_id'] = (
    df['zero_sale'].transform(
        # this lambda function calculates the sum of the previous 3 values
        lambda x: x.rolling(window=3).sum()
        # compares each transformed value to the integer 3. The result of this 
        # comparison is a boolean Series with 'True' for values that are equal 
        # to 3 and 'False' otherwise
        ).astype(np.float16) == 3 
    ) * 1

# Drop useless columns
df.drop(['id_change', 'zero_sale'], axis = 1, inplace = True)

# Stock-Outs in the last n days ?
for n_days in [1, 2 , 7]:
    df['stock_out_id_last_{}_days'.format(n_days)] = (
        df['stock_out_id'].transform(
            lambda x: x.rolling(window=n_days).sum()
            ).astype(np.float16)  >  0
        ) * 1
    
    
    
# STORE OPENING
# Was the store closed in the last n days ? 
# Assumption: if the store total sales is zero => closed

# Store Closed = Sales zero
df['store_closed'] = (
    df.groupby(['store_id', 'd'])['sold'].transform('sum').astype(np.float16) == 0
    ) * 1 

# Store Closed = Sales zero
for n_days in [1, 2, 7]: # closed the last week 
    df['store_closed_last_{}_days'.format(n_days)] = (
        df['store_closed'].transform(
            lambda x: x.rolling(window=n_days).sum()
            ).astype(np.float16) > 0
        ) * 1 
    




# =============================================================================
# SAVE DATA SET
# =============================================================================


# DATA TOO LARGE TO LOAD WHEN SAVED LIKE THIS!!!!!!
# path = r'C:\Users\alann\OneDrive\Documents\Python Scripts\retail_sales-forecasting\Data'
# df.to_pickle(path + '/data_features_improved.pkl')

def split_df_into_chunks(df, chunk_size):
    """
    Split a dataframe into chunks of specified size.
    """
    return [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]

def save_df_chunks_as_pickle_files(df, chunk_size, folder_path):
    """
    Split a dataframe into chunks and save each chunk as a separate pickle file in a folder.
    """
    chunks = split_df_into_chunks(df, chunk_size)

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Save each chunk as a pickle file
    for i, chunk in enumerate(chunks):
        file_path = os.path.join(folder_path, f"chunk_{i}.pickle")
        chunk.to_pickle(file_path)
        print(f"Saved chunk {i} to {file_path}")


chunk_size = 5000000  
folder_path = 'chunks_folder'
save_df_chunks_as_pickle_files(df, chunk_size, folder_path)


del df



print("--- %s seconds ---" % (time.time() - start_time)) 


















