# -*- coding: utf-8 -*-
"""
@author: Minh Kieu, University of Leeds

This function reads the processed Car_following_df_2d.csv file and try to propose a 
deep car following model to predicts the acceleration rate at the next time interval (1s)

Note that this model try to predict the acceleration only, so it's a separated
car-following model (no lane changing)

The lower the loss, the better a model (unless the model has over-fitted to the training data). 
The loss is calculated on training and validation and its interperation is how well the model 
is doing for these two sets. Unlike accuracy, loss is not a percentage. 
It is a summation of the errors made for each example in training or validation sets.

In the case of neural networks, the loss is usually negative log-likelihood and 
residual sum of squares for classification and regression respectively. 
Then naturally, the main objective in a learning model is to reduce (minimize) 
the loss function's value with respect to the model's parameters by changing 
the weight vector values through different optimization methods, 
such as backpropagation in neural networks.

Loss value implies how well or poorly a certain model behaves after each 
iteration of optimization. Ideally, one would expect the reduction of loss 
after each, or several, iteration(s).

The accuracy of a model is usually determined after the model parameters are 
learned and fixed and no learning is taking place. Then the test samples are 
fed to the model and the number of mistakes (zero-one loss) the model makes 
are recorded, after comparison to the true targets. Then the percentage of 
misclassification is calculated.

For example, if the number of test samples is 1000 and model classifies 
952 of those correctly, then the model's accuracy is 95.2%.

More about loss vs accuracy: 
https://stackoverflow.com/questions/34518656/how-to-interpret-loss-and-accuracy-for-a-machine-learning-model

"""

#load all the required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from pandas import read_csv
import tensorflow.keras
#from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from sklearn.preprocessing import MinMaxScaler
#from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Step 1: load data abd process the data for modelling
prject_path = '/Users/MinhKieu/Documents/Github/data-driven-car-following/'
filename = prject_path + "data/Car_following_df.csv"
dataset = np.genfromtxt(filename, delimiter=',',dtype=None)

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
acceleration = np.array(dataset[:,dataset.shape[1]-1])  #last column: acceleration
lanechange = np.array(dataset[:,dataset.shape[1]-2])   #second last column: lane change (binary)

target = acceleration

# Remove the labels from the features
dataset= dataset[:,7:-2]
# Using Skicit-learn to split data into training and testing sets
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(dataset, target, test_size = 0.25, random_state = 42)


##########
# Step 2: Develop and train the Deep Learning model

def build_model():
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=train_features.shape[1]))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])
    return model

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# Display training progress by printing a single dot for each completed epoch
class PrintDot(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 100

history = model.fit(
  train_features, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop])


##########
# Step 3: Analyse the modelling progress

#score: first is the mean squared error, second is the accuracy in percentage
# for instance accuracy = 0.94 means 94% of accuracy
#score = model.evaluate(test_features, test_labels, batch_size=128)

#hist = pd.DataFrame(history.history)
#hist['epoch'] = history.epoch

#model.summary()


def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Validation Error')
  #plt.ylim([0,5])
  plt.legend()
  plt.savefig(prject_path + "figures/NN_mae.pdf", bbox_inches='tight')

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Validation Error')
  #plt.ylim([0,100])
  plt.legend() 
  plt.savefig(prject_path + "figures/NN_mse.pdf", bbox_inches='tight')

  plt.show()

plot_history(history)


##########
# Step 4: Test the models

#loss, mae, mse = model.evaluate(test_features, test_labels, verbose=2)

#print("Testing set Mean Abs Error: {:5.2f}".format(mae))

test_predictions = model.predict(test_features)

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
plt.savefig(prject_path + "figures/scatter_pred.pdf", bbox_inches='tight')
plt.show()
#_ = plt.plot([-100, 100], [-100, 100])

#plt.figure()
#error = test_predictions - test_labels
#plt.hist(error, bins = 25)
#plt.xlabel("Prediction Error")
#_ = plt.ylabel("Count")
#plt.savefig(prject_path + "figures/error_bins.pdf", bbox_inches='tight')
#plt.show()

