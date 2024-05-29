# Demand-Forecasting

This analysis will be based on the M5 Forecasting dataset of Walmart stores sales records:

- 1,913 days for the training set and 28 days for the evaluation set
- 10 stores in 3 states (USA)
- 3,049 unique products in 10 stores
- 3 main categories and 7 departments (sub-category)
    
The objective is to predict sales for all products in each store, in the following 28 days right after the available dataset. We have to perform 30,490 forecasts for each day in the prediction horizon.

We’ll use the validation set to measure the performance of our model.

## SUMMARY
**I. Introduction**
1. Data set
2. Initial Solution using LGBM
3. Features Analysis

**II. Experiment**
1. Additional features
2. Results

**III. Conclusion and next steps**
