# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:05:31 2019

@author: geomlk

This function reads the processed Car_following_df_2d.csv file and try to propose a 
deep car following model to predicts the acceleration rate at the next frame (1/25s)
"""

#import pandas as pd
import numpy as np


#load data
prject_path = '/Users/MinhKieu/Documents/Research/highD/'
filename = prject_path + "data/Car_following_df_AM_v2.csv"
df = np.loadtxt(filename, delimiter=',')

#Fit a Random Forest to the data
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

sc = StandardScaler()
X_train = sc.fit_transform(df[:,:-1])
RF = RandomForestRegressor(n_estimators=10, random_state=1)
RF.fit(X_train,df[:,31])


