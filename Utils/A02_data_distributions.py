"""
Author: Minh Kieu, University of Leeds, Oct 2019
This function reads the raw HighD data to prepare an individual data for 
population synthesis for the ABM model

Outputs: Each line of data is an individual vehicle/driver with

1. Vehicle Type: Binary (Car vs Trucks)
2. Start time: appearance time in the data (h): Float
3. Start Lane: initial lane: Integer
4. Width: Float (m)
5. Length: Float (m)
6. Desired Speed: Float (km/h)
7. Max Acceleration: Float (m/s2)
8. Min Distance Headway: Float (m)
9. Min Time Headway: Float (s)
10. Min Time to Collision: Float (s) 

In the data, we would need the following file:

1.Track Meta Information (XX_tracksMeta.csv)
This file contains an overview of all tracks. For each track there are summary values like the distance covered or
the average speed. The purpose of this file is to allow to filter tracks e.g. by class or driving direction.

    
"""

#import pickle
import pandas as pd
import numpy as np
#import os



#prject_path = '~/Documents/Research/highD/'
prject_path = 'C:/Research/highD/'
dynamic_col_to_use = [0,6,12,13,14,15,24]
static_col_to_use = [1,2,6,9,10,11,12,13,14,15]
nearby_index = [2,6]


Location = 2  #focus only on the location number 2 in the dataset
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

    #Step 1: Read the Record Metadata
    recordMeta_df = pd.read_csv(record_name)
    #only take the data where time is between 6-10AM (morning peak)
    #if int(recordMeta_df["startTime"][0][1]) <6 or int(recordMeta_df["startTime"][0][1]) >9:
    #    continue
    timestamp =  pd.to_datetime(recordMeta_df["startTime"][0],format='%H:%M')
    time_hour =np.array(timestamp.hour+timestamp.minute/60)
    #only take data if it's on our location of interests
    
    if recordMeta_df["locationId"][0] != Location:
        continue
    
    #Step 2: Read the tracksMeta data (summary about each vehicle)
    tracksMeta_df = pd.read_csv(tracksMeta_name)
    #Read the track data (individual vehicle data)
    all_track_df = pd.read_csv(track_name)
    #loop through the tracksMeta line-by-line, each line is a vehicle
    for l in range(0,len(tracksMeta_df.index)):
        trackID = tracksMeta_df["id"][l]
        drivingDirection = tracksMeta_df["drivingDirection"][l]
        numFrames = tracksMeta_df["numFrames"][l]
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
        #maxXSpeed,meanXSpeed,min Distance Headway (DHW), min Time Headway (THW)
        #, min Time to Collision (TTC), number of lane changes         
        
        #Step 3: Find the dynamic features of each vehicle
        track_df = all_track_df[all_track_df["id"]==trackID].reset_index(drop=True)
        # on the upper half of the video, the speed and acceleration is negative 
        # because it uses universal positioning
        # we need to convert it to the otherway around
        if drivingDirection==1:
            track_df["xVelocity"]=-track_df["xVelocity"]
            track_df["xAcceleration"]=-track_df["xAcceleration"]
            track_df["precedingXVelocity"]=-track_df["precedingXVelocity"]
        
        # loop through each line in the track data
        for t in range(0,len(track_df.index)-1):
            #print('currently looking at line:' + str(t))

            #################################################################            
            # collect all the dynamic vehicle data (e.g. position, speed, etc)
            dynamic_df_track = np.array(track_df.iloc[t,dynamic_col_to_use])
            # META DATA OF dynamic_df_track: frame,XSpeed, Distance Headway, 
            #Time Headway, Time to Collision, Preceeding XSpeed,LaneID           
            
            #Step 4: Now look at the all_track_df data to find the location 
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

            #Step 5: Now combine all the data together
        
            #The output of the car-following model is the acceleration
            Output = np.array(track_df.loc[t+1,"xAcceleration"])
            # Combine the whole line of data
            line_df = np.hstack([i,trackID,numFrames,drivingDirection,time_hour,dynamic_df_track,leftPreceding_df,leftAlongside_df,leftFollowing_df,rightPreceding_df,rightAlongside_df,rightFollowing_df,static_df_track,Output])
            # METADATA OF THE WHOLE DATAFRAME:
            # fileid,trackID,numFrames,drivingDirection,time_hour,frame,XSpeed, 
            #Distance Headway, Time Headway, Time to Collision, Preceeding XSpeed,
            #LaneID,maxXSpeed,meanXSpeed,min Distance Headway (DHW), 
            #min Time Headway (THW), min Time to Collision (TTC), 
            #number of lane changes,leftPreceding_df,leftAlongside_df,leftFollowing_df,
            #rightPreceding_df,rightAlongside_df (each as Xpos and Xspeed), Output (Acceleration)
            
            
            Car_following_df.append(line_df)
            
        print(len(Car_following_df))
    
#save pickle file
#with open('Car_following_df.pickle', 'wb') as f:
#    pickle.dump(Car_following_df, f)
#save csv file as well
Car_following_df_2d = np.vstack(Car_following_df)        
np.savetxt("Car_following_df_2d.csv", Car_following_df_2d,fmt='%10.3f', delimiter=",")    
