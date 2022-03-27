# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 23:51:23 2022

@author: Raghav_Agarwal
"""

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# |        Bike Sharing Demand Prediction for the hourly dataset        |
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------



# ----------------------------------------------------
# Description of Various Variables
# ----------------------------------------------------
# Season : 1-Spring, 2-Summer, 3-Fall, 4-Winter
# Holiday : 1-Yes, 2-No
# Weekday : 0-6 for Sunday to Saturday
# WorkingDay : 0-No, 1-Yes
# Weather :
#       1 - Clear, Few clouds
#       2 - Mist, CLoudy
#       3 - Light rain, light thunderstorm
#       4 - Heavy Rain, Snow
# Temp : Normalized temperature in celsius
# atemp : Normalized feeling temperature in celsius
# ----------------------------------------------------



# ---------------------------
# Step 0 - Import Libraries
# ---------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math



# --------------------------
# Step 1 - Read the data
# --------------------------

bikes = pd.read_csv('hour.csv')



# -------------------------------------------------------------------
# Step 2 - Prelim analysis and Feature selection
# -------------------------------------------------------------------
# We are excluding index, date, casual, registered columns because
# they are not required
# -------------------------------------------------------------------

bikes_prep = bikes.copy()
bikes_prep = bikes_prep.drop(['index', 'date', 'casual', 'registered'], axis=1)


# --------------------------------
# Basic check of missing values
# --------------------------------

print(bikes_prep.isnull().sum())
# it doesn't have any null values


# --------------------------------------------
# Visualize the data using pandas histogram
# --------------------------------------------

bikes_prep.hist(rwidth=0.9)
plt.tight_layout()



# ---------------------------------------------------------
# Step 3 - Data Visualization
# ---------------------------------------------------------


# ---------------------------------------------------------
# Visualizing the continuous features vs demand
# Continuous features - temp, atemp, humidity, windspeed
# Create a 2x2 subplot
# ---------------------------------------------------------

plt.subplot(2, 2, 1)
plt.title('Temperature vs Demand')
plt.scatter(bikes_prep['temp'], bikes_prep['demand'], s=2, c='b')

plt.subplot(2, 2, 2)
plt.title('aTemp vs Demand')
plt.scatter(bikes_prep['atemp'], bikes_prep['demand'], s=2, c='g')

plt.subplot(2, 2, 3)
plt.title('Humidity vs Demand')
plt.scatter(bikes_prep['humidity'], bikes_prep['demand'], s=2, c='m')

plt.subplot(2, 2, 4)
plt.title('Windspeed vs Demand')
plt.scatter(bikes_prep['windspeed'], bikes_prep['demand'], s=2, c='c')

plt.tight_layout()


# ---------------------------------------------------
# Visualizing the categorical features vs demand
# Categorical variables - season, month, holiday
# Create a 3x3 subplot
# ---------------------------------------------------

plt.subplot(3, 3, 1)
plt.title('Average demand per Season')

# Get unique values of season column using unique
cat_list = bikes_prep['season'].unique()

# Create average demand per season using groupby
cat_average = bikes_prep.groupby('season').mean()['demand']

colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 2)
plt.title('Average demand per Month')
cat_list = bikes_prep['month'].unique()
cat_average = bikes_prep.groupby('month').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 3)
plt.title('Average demand per Holiday')
cat_list = bikes_prep['holiday'].unique()
cat_average = bikes_prep.groupby('holiday').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 4)
plt.title('Average demand per Weekday')
cat_list = bikes_prep['weekday'].unique()
cat_average = bikes_prep.groupby('weekday').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 5)
plt.title('Average demand per Year')
cat_list = bikes_prep['year'].unique()
cat_average = bikes_prep.groupby('year').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 6)
plt.title('Average demand per Hour')
cat_list = bikes_prep['hour'].unique()
cat_average = bikes_prep.groupby('hour').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 7)
plt.title('Average demand per Workingday')
cat_list = bikes_prep['workingday'].unique()
cat_average = bikes_prep.groupby('workingday').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)


plt.subplot(3, 3, 8)
plt.title('Average demand per Weather')
cat_list = bikes_prep['weather'].unique()
cat_average = bikes_prep.groupby('weather').mean()['demand']
colors = ['g', 'r', 'm', 'b']
plt.bar(cat_list, cat_average, color = colors)

plt.tight_layout()



# -------------------------
# Check for outliers
# -------------------------

bikes_prep['demand'].describe()
bikes_prep['demand'].quantile([0.05, 0.10, 0.15, 0.25, 0.90, 0.95, 0.99])



# ---------------------------------------------------------
# Step 4 - Check Multiple Linear Regression Assumptions
# ---------------------------------------------------------

# ---------------------------------------------------------------------
# Linearity using correlation coefficient matrix using corr function
# ---------------------------------------------------------------------
correlation = bikes_prep[['temp', 'atemp', 'humidity', 'windspeed', 'demand']].corr()

# dropping features based on data visualization and correlation matrix
bikes_prep = bikes_prep.drop(['weekday', 'year', 'workingday', 'atemp', 'windspeed'], axis=1)


# -------------------------------------------------------------
# Check the autocorrelation in demand using acorr function
# -------------------------------------------------------------

# for checking autocorrelation, we need to convert the column to float
df1 = pd.to_numeric(bikes_prep['demand'], downcast='float')

# plotting the autocorrelation graph
plt.acorr(df1, maxlags=12)



# -------------------------------------
# Step 6 - Modify/Create new features
# -------------------------------------
# Log Normalise the feature demand
# -------------------------------------

df1 = bikes_prep['demand']
# Doing log transformation by using log function of numpy
df2 = np.log(df1)

plt.figure()
plt.hist(df1, rwidth=0.9, bins = 20)

plt.figure()
plt.hist(df2, rwidth=0.9, bins = 20)

# After plotting, we found we are getting kind of normal distribution
# so we will convert the actual demand column by log transformation
bikes_prep['demand'] = np.log(bikes_prep['demand'])


# ----------------------------------------------
# Handling Autocorrelation in demand column
# ----------------------------------------------

t_1 = bikes_prep['demand'].shift(+1).to_frame()
t_1.columns = ['t-1']
t_2 = bikes_prep['demand'].shift(+2).to_frame()
t_2.columns = ['t-2']
t_3 = bikes_prep['demand'].shift(+3).to_frame()
t_3.columns = ['t-3']

bikes_prep_lag = pd.concat([bikes_prep, t_1, t_2, t_3], axis = 1)

bikes_prep_lag = bikes_prep_lag.dropna()



# --------------------------------------------------
# Step 7 - Create dummy variables and drop first
#          to avoid dummy variables trap
#          using get_dummies function
# --------------------------------------------------
# features - season, holiday, weather, month, hour
# --------------------------------------------------

# for get_dummies to work, we need to change the data type
# to category for those columns

bikes_prep_lag.dtypes

bikes_prep_lag['season'] = bikes_prep_lag['season'].astype('category')
bikes_prep_lag['holiday'] = bikes_prep_lag['holiday'].astype('category')
bikes_prep_lag['weather'] = bikes_prep_lag['weather'].astype('category')
bikes_prep_lag['month'] = bikes_prep_lag['month'].astype('category')
bikes_prep_lag['hour'] = bikes_prep_lag['hour'].astype('category')

bikes_prep_lag = pd.get_dummies(bikes_prep_lag, drop_first=True)



# ----------------------------------------------
# Step 8 - Create Train and test split
# ----------------------------------------------

# Split the data into independent and dependent (X and Y)
# and then into train and test data

# from sklearn.model import train_test_split
# X_train, X_test, Y_train, Y_test = \
#     train_test_split(X, Y, test_size=0.4, random_state=1234)

# however demand is time dependent or time series
# so we cannot randomly select the test and train data
# so we need to take the data from start, or middle or end
# to have complete data for a particular period

Y = bikes_prep_lag[['demand']]
X = bikes_prep_lag.drop(['demand'], axis=1)

# Create the training size as 70% of data
tr_size = 0.7 * len(X)
tr_size = int(tr_size)

X_train = X.values[0:tr_size]
X_test  = X.values[tr_size:len(X)]

Y_train = Y.values[0:tr_size]
Y_test  = Y.values[tr_size:len(Y)]



# ----------------------------------
# Step 9 - Fit and Score the model
# ----------------------------------

from sklearn.linear_model import LinearRegression

std_reg = LinearRegression()
std_reg.fit(X_train, Y_train)

r2_train = std_reg.score(X_train, Y_train)
r2_test  = std_reg.score(X_test, Y_test)

# Create Y predictions
Y_predict = std_reg.predict(X_test)


from sklearn.metrics import mean_squared_error
rmse = math.sqrt(mean_squared_error(Y_test, Y_predict))



# ------------------------------------------------------------------
# Final Step - Calculate RMSLE and compare results
# ------------------------------------------------------------------
# We converted demand (Y) to log while doing normal distribution
# To do RMSLE, we need to convert it back to original
# We can convert log values back to original by taking exponent
# ------------------------------------------------------------------

Y_test_e    = []
Y_predict_e = []

for i in range(0, len(Y_test)):
    Y_test_e.append(math.exp(Y_test[i]))
    Y_predict_e.append(math.exp(Y_predict[i]))
    
# Calculation of RMSLE

log_sq_sum = 0.0

for i in range(0, len(Y_test_e)):
    log_a      = math.log(Y_test_e[i] + 1)
    log_p      = math.log(Y_predict_e[i] + 1)
    log_diff   = (log_p - log_a)**2
    log_sq_sum = log_sq_sum + log_diff
    
rmsle = math.sqrt(log_sq_sum / len(Y_test_e))

print("Root Mean Squared Logarithmic Error =", rmsle)


