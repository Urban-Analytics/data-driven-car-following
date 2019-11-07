"""
Author: Minh Kieu, University of Leeds, Oct 2019
This function reads the raw HighD data to prepare an individual data for 
population synthesis for the ABM model

Outputs: Each line of data is an individual vehicle/driver with

Variables from Track Meta and 

1. Vehicle Type: Binary (Car vs Trucks)
2. Start time: appearance time in the data (h): Float
3. Width: Float (m)
4. Length: Float (m)

Variables from Tracks data: Take only leading vehicles because they are
unrestricted

5. Max Speed: Float (km/h)
6. Start Lane: initial lane: Integer
7. Max Acceleration: Float (m/s2)

In the data, we would need the following file:

- Track Meta Information (XX_tracksMeta.csv)
This file contains an overview of all tracks. For each track there are summary values like the distance covered or
the average speed. The purpose of this file is to allow to filter tracks e.g. by class or driving direction.

- Tracks (XX_tracks.csv)
This file contains all time dependent values for each track. Information such as current velocities, viewing ranges
and information about surrounding vehicles are included.
    
"""

#import pickle
import pandas as pd
import numpy as np
#import os



prject_path = '~/Documents/Research/highD/'
#prject_path = 'C:/Research/highD/'
following_ratio_threshold = 0.3
unique_count=1
#Location = 2  #focus only on the location number 2 in the dataset
Veh_features = []
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

    #only take data if it's on our location of interests
    #if recordMeta_df["locationId"][0] != Location:
    #    continue

    timestamp =  pd.to_datetime(recordMeta_df["startTime"][0],format='%H:%M')
    time_hour =np.array(timestamp.hour+timestamp.minute/60)
    
    #Step 2: Read the tracksMeta data (summary about each vehicle)
    tracksMeta_df = pd.read_csv(tracksMeta_name)
    #Read the track data (individual vehicle data)
    all_track_df = pd.read_csv(track_name)
    #loop through the tracksMeta line-by-line, each line is a vehicle
    for l in range(0,len(tracksMeta_df.index)):
        trackID = tracksMeta_df["id"][l]

        #Step 3: Find the dynamic features of each vehicle
        track_df = all_track_df[all_track_df["id"]==trackID].reset_index(drop=True)

        #following ratio: the amount of time  the vehicle is following some other vehicle
        if len(track_df[track_df["precedingId"]!=0])/len(track_df) > following_ratio_threshold:
            continue
        
        
        
        startlane = track_df["laneId"][0]

        drivingDirection = tracksMeta_df["drivingDirection"][l]

        if drivingDirection==1:
            track_df["xVelocity"]=-track_df["xVelocity"]
            track_df["xAcceleration"]=-track_df["xAcceleration"]
            track_df["precedingXVelocity"]=-track_df["precedingXVelocity"]
        
        startSpeed = track_df["xVelocity"][0]
        time_start = time_hour + tracksMeta_df["initialFrame"][l]/(recordMeta_df["frameRate"][0]*3600)

        length = tracksMeta_df["width"][l]      
        width = tracksMeta_df["height"][l] 
        
        if tracksMeta_df["class"][l]=='Car':
            veh_class=0
        else: veh_class=1
        
        #Take the time where the vehicle is actually the leading vehicle
        lead_df = track_df[track_df["precedingId"]==0]
        
        maxSpeed = np.max(lead_df["xVelocity"])
        maxAcceleration = np.max(lead_df["xAcceleration"])
        
        # Combine the whole line of data
        line_df = np.hstack([unique_count,recordMeta_df["locationId"][0],drivingDirection,time_start,startlane,startSpeed,length,width,veh_class,maxSpeed,maxAcceleration])
            
        unique_count+=1    
        Veh_features.append(line_df)
            
        print(len(Veh_features))
    
#save pickle file
#with open('Car_following_df.pickle', 'wb') as f:
#    pickle.dump(Car_following_df, f)
#save csv file as well
Veh_features = np.vstack(Veh_features)        
np.savetxt("Veh_features.csv", Veh_features,fmt='%10.5f', delimiter=",")    
