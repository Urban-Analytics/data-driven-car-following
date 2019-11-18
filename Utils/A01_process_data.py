"""
Author: Minh Kieu, University of Leeds, Oct 2019
 this function processes the HighD data to Input-Output data for a car-following model

In the data, we would have the following files:

1. Recording Meta Information (XX_recordingMeta.csv)
This file contains metadata for each recording. The metadata provides a general overview, e.g. of the time of recording,
the highway section considered and the total number of vehicles recorded.

2.Track Meta Information (XX_tracksMeta.csv)
This file contains an overview of all tracks. For each track there are summary values like the distance covered or
the average speed. The purpose of this file is to allow to filter tracks e.g. by class or driving direction.

3.Tracks (XX_tracks.csv)
This file contains all time dependent values for each track. Information such as current velocities, viewing ranges
and information about surrounding vehicles are included.

This function processes all of these files

TODO:
1. Empirical analysis: Identify cases/situations in the data
2. Throw the data to a DL
3. Simulate and see if it's reproduce the data
4. If not, then we separate the cases to each model
5. 


1. Distance to the leading vehicle
2. Driver heterogeneity : different models for each class of drivers
3. Cooperative & uncooperative behaviours
4. Find out whether a car nearby is changing their lanes
5. Rare situations: in the data or not in the data but the model needs to
give estimates


"""

#import pickle
import pandas as pd
import numpy as np
#import os

minSec =10 # in seconds, we focus on vehicles that stay at least 40s in the data

#prject_path = '~/Documents/Research/highD/'
prject_path = 'C:/Research/highD/'
dynamic_col_to_use = [0,6,12,13,14,15,24]
static_col_to_use = [1,2,6,9,10,11]
nearby_index = [2,6]

uniqueID = 0 #give an unique ID to the vehicle being processed

Location = 2  #focus only on the location number 2 in the dataset
L = 0.424  # length of the location under study (in km)
NumLane = 2

"""
STAGE A: First, we process data into a line-by-line dataset of all related information
"""

print("Stage A")
Car_following_df = []
for i in range(1,60):
    print("currently at file: " + str(i))
    #file names:
    if i <10:
        record_name = prject_path + "data/0" + str(i) + "_recordingMeta.csv"
        tracksMeta_name = prject_path + "./data/0" + str(i) + "_tracksMeta.csv"
        track_name = prject_path + "./data/0" + str(i) + "_tracks.csv"
    else:
        record_name = prject_path + "./data/" + str(i) + "_recordingMeta.csv"
        tracksMeta_name = prject_path + "./data/" + str(i) + "_tracksMeta.csv"
        track_name = prject_path + "./data/" + str(i) + "_tracks.csv"

    #Step A.1: Read the Record Metadata
    recordMeta_df = pd.read_csv(record_name)
    #only take the data in the morning (if we take the whole day there will be >1M data lines)
    #if int(recordMeta_df["startTime"][0][1]) >12:
    #    continue
    timestamp =  pd.to_datetime(recordMeta_df["startTime"][0],format='%H:%M')
    time_hour =np.array(timestamp.hour+timestamp.minute/60)
    #only take data if it's on our location of interests
    
    if recordMeta_df["locationId"][0] != Location:
        continue
    
    #Step A.2: Read the tracksMeta data (summary about each vehicle)
    tracksMeta_df = pd.read_csv(tracksMeta_name)
    #Read the track data (individual vehicle data)
    all_track_df = pd.read_csv(track_name)
    #loop through the tracksMeta line-by-line, each line is a vehicle
    for l in range(0,len(tracksMeta_df.index)):
        trackID = tracksMeta_df["id"][l]
        drivingDirection = tracksMeta_df["drivingDirection"][l]  #1 for upper lanes (drive to the left), and 2 for lower lanes (drive to the right)
        numFrames = tracksMeta_df["numFrames"][l]
        
        if numFrames < recordMeta_df["frameRate"][0]*minSec:  #only focus to vehicles that we can observed for more than minSec seconds
            continue
        #sanity check
        if trackID != tracksMeta_df.iloc[l,0]:
            print("The trackID is not the same at line: " + str(l))
        
        ############################################################        
        # find all the static data of the vehicle (e.g. vehicle length, class, etc)
        static_df_track = np.array(tracksMeta_df.iloc[l,static_col_to_use])
        # convert categorical to binary variable (e.g Car vs Truck)
        if static_df_track[2]=='Car':
            static_df_track[2]=0
        else: static_df_track[2]=1 #otherwise it should be a truck           
        #convert to float for speed
        static_df_track=static_df_track.astype(float)
        #META DATA OF static_df_float: width, height, class, minXSpeed,
        #maxXSpeed,meanXSpeed
        
        #Step A.3: Find the dynamic features of each vehicle
        track_df = all_track_df[all_track_df["id"]==trackID].reset_index(drop=True)
        # on the upper half of the video, the speed and acceleration is negative 
        # because it uses universal positioning
        # we need to convert it to the otherway around
        if drivingDirection==1:
            track_df["xVelocity"]=-track_df["xVelocity"]
            track_df["xAcceleration"]=-track_df["xAcceleration"]
            track_df["precedingXVelocity"]=-track_df["precedingXVelocity"]
        
        # loop through each line in the track data
        for t in range(0,len(track_df.index)-1,recordMeta_df["frameRate"][0]):  #loop by each second
            #print('currently looking at line:' + str(t))

            #################################################################            
            # collect all the dynamic vehicle data (e.g. position, speed, etc)
            dynamic_df_track = np.array(track_df.iloc[t,dynamic_col_to_use])
            # META DATA OF dynamic_df_track: XSpeed, Distance Headway,
            #Time Headway, Time to Collision, Preceeding XSpeed
            frameID = dynamic_df_track[0]
            laneID = dynamic_df_track[-1]
            
            #Step A.4: Find traffic-related variables: Density and traffic mean speed
            if drivingDirection==1:
                traffic_density = len(all_track_df[(all_track_df["frame"]==frameID) & (all_track_df["laneId"] < NumLane+2)]) / (L*NumLane)
            else: traffic_density = len(all_track_df[(all_track_df["frame"]==frameID) & (all_track_df["laneId"] > NumLane+1)]) / (L*NumLane)
            
            if drivingDirection==1:
                traffic_speed = -np.mean(all_track_df.loc[(all_track_df["frame"]==frameID) & (all_track_df["laneId"] < NumLane+2),"xVelocity"])
            else: traffic_speed = np.mean(all_track_df.loc[(all_track_df["frame"]==frameID) & (all_track_df["laneId"] > NumLane+1),"xVelocity"])
            
            
            #Step A.5: Now look at the all_track_df data to find the location 
            #and speed of surrounding vehicles
            #for each vehicle we keep [x_location,speed]
            
            if track_df["leftPrecedingId"][t]!=0:
                leftPreceding_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftPrecedingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftPreceding_df[0] = np.abs(leftPreceding_df[0]-track_df["x"][t])
                leftPreceding_df[1]= np.abs(leftPreceding_df[1])
            else: leftPreceding_df = np.array([0,0])

            if track_df["leftFollowingId"][t]!=0:
                leftFollowing_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftFollowingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftFollowing_df[0] = np.abs(leftFollowing_df[0]-track_df["x"][t])
                leftFollowing_df[1] = np.abs(leftFollowing_df[1])
            else: leftFollowing_df = np.array([0,0])

            if track_df["leftAlongsideId"][t]!=0:
                leftAlongside_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["leftAlongsideId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                leftAlongside_df[0] = np.abs(leftAlongside_df[0]-track_df["x"][t])
                leftAlongside_df[1] = np.abs(leftAlongside_df[1])
            else: leftAlongside_df = np.array([0,0])
            

            if track_df["rightPrecedingId"][t]!=0:
                rightPreceding_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightPrecedingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightPreceding_df[0] = np.abs(rightPreceding_df[0]-track_df["x"][t])
                rightPreceding_df[1] = np.abs(rightPreceding_df[1])
            else: rightPreceding_df = np.array([0,0])

            if track_df["rightAlongsideId"][t]!=0:
                rightAlongside_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightAlongsideId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightAlongside_df[0] = np.abs(rightAlongside_df[0]-track_df["x"][t]) 
                rightAlongside_df[1] = np.abs(rightAlongside_df[1])
            else: rightAlongside_df = np.array([0,0])

            if track_df["rightFollowingId"][t]!=0:
                rightFollowing_df = np.array(all_track_df.loc[(all_track_df["id"] == track_df["rightFollowingId"][t]) & (all_track_df["frame"] == track_df["frame"][t])].values[0][nearby_index])
                rightFollowing_df[0] = np.abs(rightFollowing_df[0]-track_df["x"][t])
                rightFollowing_df[1] = np.abs(rightFollowing_df[1])
            else: rightFollowing_df = np.array([0,0])

            #Step A.6: Now combine all the data together
        
            #The output of the car-following model is the acceleration
            Acceleration = np.array(track_df.loc[t+1,"xAcceleration"])
            # Combine the whole line of data
            line_df = np.hstack([uniqueID,frameID,drivingDirection,time_hour,static_df_track,dynamic_df_track[1:-1],leftPreceding_df,leftAlongside_df,leftFollowing_df,rightPreceding_df,rightAlongside_df,rightFollowing_df,traffic_density,traffic_speed,laneID,Acceleration])
            # METADATA OF THE WHOLE DATAFRAME:
            # uniqueID,frameID,drivingDirection,time_hour,width, height, class, minXSpeed,
            #maxXSpeed,meanXSpeed,XSpeed,Distance Headway, Time Headway, Time to Collision, Preceeding XSpeed,
            #LaneID,leftPreceding_df,leftAlongside_df,leftFollowing_df,
            #rightPreceding_df,rightAlongside_df (each as Xpos and Xspeed), Output (Acceleration)
            
            
            Car_following_df.append(line_df)
        
        uniqueID += 1
        print(len(Car_following_df))
        
    
#import pickle
#with open('Car_following_df.pickle', 'wb') as f:
#    pickle.dump(Car_following_df_2d, f)
#save csv file as well
Car_following_df_2d = np.vstack(Car_following_df)        
#np.savetxt("Car_following_df_AM.csv", Car_following_df_2d,fmt='%6.2f', delimiter=",")    

np.savetxt("Car_following_df_raw.csv", Car_following_df_2d,fmt='%5.2f', delimiter=",")   

"""
STAGE B: next, we process the data such that data from previous time steps are also included in the features
"""
print("Stage B")

static_index = [2,3,4,5,6,7,8,9]

list_veh = np.unique(Car_following_df_2d[:,0])
DL_df = []
#loop through each vehicle in the processed data
for v in list_veh:
    Veh_df = Car_following_df_2d[Car_following_df_2d[:,0]==v,:]  #take out the vehicle data to analyse
    
   
    for l in range(0,len(Veh_df)-3):
        #take static data only
        static_df = Veh_df[l,static_index]
        #now look at 03 time steps ahead and take all the dynamic information
        dynamic1 = Veh_df[l,static_index[-1]+1:-2]
        dynamic2 = Veh_df[l+1,static_index[-1]+1:-2]
        dynamic3 = Veh_df[l+2,static_index[-1]+1:-2]
        #combine all the static and dynamic data
        line_df = np.hstack([static_df,dynamic1,dynamic2,dynamic3,np.abs(Veh_df[l+3,-2]-Veh_df[l,-2]),Veh_df[l+3,-1]])
        #write to a large list
        DL_df.append(line_df)
        
    print(v)
#convert the list into a 2D dataframe
DL_df_2D = np.vstack(DL_df)
#write to data file
np.savetxt("Car_following_df.csv", DL_df_2D,fmt='%5.2f', delimiter=",")   
                     
                     
                
        



