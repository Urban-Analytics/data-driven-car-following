# -*- coding: utf-8 -*-
"""
@author: Minh Kieu, University of Leeds

This function develops a lane-changing model from the data
It uses Random Forest to perform the classication of lane change or not



"""

# load all the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Step 1: load data abd process the data for modelling
prject_path = '/Users/MinhKieu/Documents/Github/data-driven-car-following/'
filename = prject_path + "data/Car_following_df.csv"
dataset = np.genfromtxt(filename, delimiter=',', dtype=None)

names=('drivingDirection','time_hour','width','height', 'class', 'minXSpeed',
        'maxXSpeed','meanXSpeed',
        'Speed3', 'Distance_Headway3',
        'Time_Headway3' ,'Time_to_Collision3','Preceeding_Speed3','Left_Pre_X3',
        'Left_Pre_Speed3','Left_Al_X3','Left_Al_Speed3','Left_Fol_X3','Left_Fol_Speed3',
        'Right_Pre_X3','Right_Pre_Speed3','Right_Al_X3','Right_Al_Speed3','Right_Fol_X3',
        'Right_Fol_Speed3','traffic_density3','traffic_speed3',
        'Speed2', 'Distance_Headway2',
        'Time_Headway2', 'Time_to_Collision2', 'Preceeding_Speed2', 'Left_Pre_X2',
        'Left_Pre_Speed2', 'Left_Al_X2', 'Left_Al_Speed2', 'Left_Fol_X2', 'Left_Fol_Speed2',
        'Right_Pre_X2', 'Right_Pre_Speed2', 'Right_Al_X2', 'Right_Al_Speed2', 'Right_Fol_X2',
        'Right_Fol_Speed2', 'traffic_density2', 'traffic_speed2',
        'Speed1', 'Distance_Headway1',
        'Time_Headway1', 'Time_to_Collision1', 'Preceeding_Speed1', 'Left_Pre_X1',
        'Left_Pre_Speed1', 'Left_Al_X1', 'Left_Al_Speed1', 'Left_Fol_X1', 'Left_Fol_Speed1',
        'Right_Pre_X1', 'Right_Pre_Speed1', 'Right_Al_X1', 'Right_Al_Speed1', 'Right_Fol_X1',
        'Right_Fol_Speed1', 'traffic_density1', 'traffic_speed1')


## process the data to consider static vs dynamic variables, and also consider several time steps

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Labels are the values we want to predict
acceleration = np.array(dataset[:, dataset.shape[1] - 1])  # last column: acceleration
lanechange = np.array(dataset[:, dataset.shape[1] - 2])  # second last column: lane change (binary)

target = lanechange

# Remove the labels from the features
dataset = dataset[:, 7:-2]
# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(dataset, target, test_size=0.25,
                                                                            random_state=42)

##########
# Step 2: Develop and train the Random Forest model
#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(train_features,train_labels)

test_predictions=clf.predict(test_features)
# Probabilities for each class
rf_probs = clf.predict_proba(test_features)[:, 1]

pd.crosstab(test_labels, test_predictions, rownames=['Actual Result'], colnames=['Predicted Result'])

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(test_labels, test_predictions))

##########
# Step 4: Find important features
from sklearn.metrics import r2_score
from rfpimp import permutation_importances

def r2(clf, train_features, train_labels):
    return r2_score(train_labels, clf.predict(train_features))

perm_imp_rfpimp = permutation_importances(clf, pd.DataFrame(train_features,columns=names[7:]), pd.DataFrame(train_labels), r2)

perm_imp_rfpimp=perm_imp_rfpimp.iloc[:,0]

df_plt = perm_imp_rfpimp[perm_imp_rfpimp>0]

import matplotlib.pyplot as plt
import seaborn as sns
# Creating a bar plot
sns.barplot(x=df_plt, y=df_plt.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features")
#plt.legend()
plt.savefig(prject_path + "figures/Important_features.pdf", bbox_inches='tight')
plt.show()

