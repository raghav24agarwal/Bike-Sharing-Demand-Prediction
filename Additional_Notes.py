# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 00:12:55 2022

@author: Raghav_Agarwal
"""

# --------------------------------------------------------------
# Important points
# --------------------------------------------------------------
# 1. Predicted vaiable 'demand' is not normally distributed
# --------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------
# Conclusions from plots of continuous features
# ----------------------------------------------------------------------------------------------------
# 1. As temp inc., demand goes up. Same is case for atemp.
# 2. The plots of temp and atemp are similar which means high correlation b/w them
# 3. Upto some point windspeed doesn't affects demand. Beyond that, as windspeed inc., demand dec.
# 4. Humidity is spread all over the plot, so very little change in demand with change in humidity
# ----------------------------------------------------------------------------------------------------



# -----------------------------------------------------------------------------------------------------
# Conclusion from plots of categorical features
# -----------------------------------------------------------------------------------------------------
# 1. Demand varies as season changes. Lowest during spring, highest during fall
# 2. Demand varies as month changes. Months during summer show higher demand
# 3. Demand is higher on non-holiday
# 4. Demand doesn't changes based on weekday
# 5. Demand is increasing year by year, not sure since having only two years
# 6. Demand is low past midnight and early morning and then high and then again low
# Since there is lot of variation based on hours, plot it bigger. Demand is highest at 8am and 5pm
# It's time series type of data, which means demand based on time interval
# 7. Demand doesn't changes based on workingday
# 8. Demand is highest during clear weather and lowest during rain/snow
# -----------------------------------------------------------------------------------------------------




# ------------------------------------------------------------
# Features to be dropped after data visualization
# ------------------------------------------------------------
# 1. weekday - demand doesn't changes based on weekday
# 2. year - insufficient data
# 3. workingday - demand doesn't changes based on workingday
# ------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------------
# Data Visualization Analysis Result
# ----------------------------------------------------------------------------------------------------------
# 1. Demand is not normally distributed
# 2. Temp and demand appears to have direct correlation
# 3. The plot for temp and atemp are almost identical, so we should check for multi-collinearity here
# 4. Humidity and windspeed affect demand but need more statistical analysis.
# We can check for correlation coeffcient for more clarity
# 5. There is variation in demand based on - season, month, holiday, hour, weather.
# 6. We also concluded demand is more at 8am and then 4pm/5pm
# 7. No significant change in demand due to weekday or workingday
# 8. Year wise growth pattern cannot be considered due to limited no of years
# ----------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------------------------------
# Conclusion from Correlation matrix
# --------------------------------------------------------------------------------------------
# 1. temp and atemp have near perfect linear correlation, this violates multicollinearity
# So we need to drop atemp
# 2. temp doesn't have a good correlation with humidity and windspeed, which is good
# 3. temp have a sufficient linear correlation with demand, which is good
# 4. humidity have some linear correlation with windspeed, which is not good,
# humidity has sufficient linear correlation with demand, 
# however windspeed doesn't have good linear correlation with demand
# So we are going to drop windspeed
# --------------------------------------------------------------------------------------------



# ------------------------------------------------------------
# Features to be dropped after checking correlation matrix
# ------------------------------------------------------------
# 1. atemp
# 2. windspeed
# ------------------------------------------------------------



# ---------------------------------------------------------------------
# Conclusion from autocorrelation graph
# ---------------------------------------------------------------------
# It does have very high autocorrelation with previous three values
# If its been any other indpendent column, we simply have dropped it,
# But since its the one dependent column, we cannot drop it.
# ---------------------------------------------------------------------
