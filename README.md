
# Bike Sharing Demand Prediction

In this project, I have prepared a model for predicting bike sharing demand using Multiple Linear Regression. The RMSLE score of this model is 0.356.


## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Description of Variables](#description-of-variables)
- [Steps Involved](#steps-involved)
- [Technologies Used](#technologies-used)
- [Conclusions](#conclusions)
- [Contact](#contact)


## Overview

Bike sharing systems are a means of renting bicycles where the process of obtaining membership, rental, and bike return is automated via a network of kiosk locations throughout a city. Using these systems, people are able rent a bike from a one location and return it to a different place on an as-needed basis. Currently, there are over 500 bike-sharing programs around the world.

[Actual Link to Kaggle](https://www.kaggle.com/c/bike-sharing-demand)


## Problem Statement

Here, we are asked to combine historical usage patterns with weather data in order to forecast bike rental demand in the Capital Bikeshare program in Washington, D.C.
## Description of Variables

- Season : 1-Spring, 2-Summer, 3-Fall, 4-Winter
- Holiday : 1-Yes, 2-No
- Weekday : 0-6 for Sunday to Saturday
- WorkingDay : 0-No, 1-Yes
- Weather :
    - 1-Clear, Few clouds
    - 2-Mist, CLoudy
    - 3-Light rain, light thunderstorm
    - 4-Heavy Rain, Snow
- Temp : Normalized temperature in celsius
- atemp : Normalized feeling temperature in celsius
## Steps Involved

- Read the data
- Feature Selection
- Data Visualization
- Check Multiple Linear Regression Assumptions
- Create/Modify new Features
- Train Test Split
- Fit and Score the model
## Technologies Used

- Python 3
- Spyder
- Data Analysis libraries: Numpy, Pandas, Math
- Visualization library: Matplotlib
- Machine Learning library: Sklearn


## Conclusions

- Y_predict and Y_test have very minimal difference.
- R-squared of training dataset was 0.919 and of test dataset was 0.928, which are very good.
- The RMSE (Root Mean Squared Error) of the model was 0.381.
- The RMSLE (Root Mean Squared Logarithmic Error) of the model was 0.356.
## Contact

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/raghav-agarwal-/)