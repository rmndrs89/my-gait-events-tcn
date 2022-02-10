import pandas as pd
import os
import numpy as np
from utils.preprocessing import resamp1d, create_sequences

def select_data(df_data, df_channels, tracked_points=[], incl_magn=False):
    col_names = df_channels.loc[df_channels["tracked_point"].isin(tracked_points)]["name"].values
    df_data = df_data[col_names]
    
    if incl_magn == True:
        return df_data
    else:
        col_names = [col_name for col_name in df_data.columns if not "MAGN" in col_name]
        return df_data[col_names]

def get_gait_events(df_events):
    return {"ICL": df_events[df_events["event_type"]=="initial_contact_left"]["onset"].values-1, 
            "FCL": df_events[df_events["event_type"]=="final_contact_left"]["onset"].values-1, 
            "ICR": df_events[df_events["event_type"]=="initial_contact_right"]["onset"].values-1, 
            "FCR": df_events[df_events["event_type"]=="final_contact_right"]["onset"].values-1}

def get_labels(num_time_steps, events, classification_task="events"):
    # Map event type for index
    map_events_to_int = {"ICL": 0, "FCL": 1, "ICR": 2, "FCR": 3}
    
    # Initialize zeros array
    labels = np.zeros((num_time_steps, len(events))) if classification_task == "events" else np.zeros((num_time_steps, len(events)//2))    
    
    if classification_task == "events":    
        # Iterate over the events dictionary
        for event_type, indices in events.items():
            for i in range(len(indices)):
                labels[indices[i], map_events_to_int[event_type]] = 1.0
    else:
        if events["FCL"][0] < events["ICL"][0]:
            arr = events["ICL"] - events["FCL"][:len(events["ICL"])]
            for i in range(len(arr)):
                labels[events["FCL"][i]:events["ICL"][i], map_events_to_int["ICL"]//2] = 1.0
        else:
            arr = events["ICL"][1:] - events["FCL"][:len(events["ICL"][1:])]
            for i in range(len(arr)):
                labels[events["FCL"][i]:events["ICL"][i+1], map_events_to_int["ICL"]//2] = 1.0
        if events["FCR"][0] < events["ICR"][0]:
            arr = events["ICR"] - events["FCR"][:len(events["ICR"])]
            for i in range(len(arr)):
                labels[events["FCR"][i]:events["ICR"][i], map_events_to_int["ICR"]//2] = 1.0
        else:
            arr = events["ICR"][1:] - events["FCR"][:len(events["ICR"][1:])]
            for i in range(len(arr)):
                labels[events["FCR"][i]:events["ICR"][i+1], map_events_to_int["ICR"]//2] = 1.0
    return labels

def load_data_from_file(filename, tracked_points=[], incl_magn=False, normalize=True, classification_task="events", win_len=200, step_len=200):
    if len(tracked_points) == 0:
        return
    
    # Load data
    df_data = pd.read_csv(filename, sep="\t", header=0)
    
    # Load channels info
    df_channels = pd.read_csv(filename.replace("_motion.tsv", "_channels.tsv"), sep="\t", header=0)
    
    # Check sampling frequency
    if df_channels["sampling_frequency"].iloc[0] != 200:
        X = df_data.to_numpy()
        X = resamp1d(X, df_channels["sampling_frequency"].iloc[0], 200)
        df_data = pd.DataFrame(data=X, columns=df_data.columns)
        del X
    
    # Select data from given tracked points, and magnetometer readings, if wanted
    df_select = select_data(df_data, df_channels, tracked_points=tracked_points, incl_magn=incl_magn)
    
    # Normalize
    if normalize:
        df_select = ( df_select - df_select.mean() ) / df_select.std()
    
    # Determine start and end of current task trial
    if os.path.isfile(filename.replace("_tracksys-imu_motion.tsv", "_events.tsv")):
        df_events = pd.read_csv(filename.replace("_tracksys-imu_motion.tsv", "_events.tsv"), sep="\t", header=0)
        indx_start = df_events[df_events["event_type"]=="start"]["onset"].values[0] - 1
        indx_stop  = df_events[df_events["event_type"]=="stop"]["onset"].values[0]
        
        # Get indices corresponding to gait events
        events = get_gait_events(df_events=df_events)
        
        # Get labels
        labels = get_labels(len(df_select), events, classification_task=classification_task)
    else:
        return
    
    # Combine (normalized) data and labels in array
    data = np.concatenate((df_select.to_numpy(), labels), axis=1)
    
    # Create sequences of equal length
    sequences = create_sequences(data, win_len=win_len, step_len=step_len)
    
    # Split data and labels
    if classification_task == "events":
        data, labels = sequences[:,:,:-4], sequences[:,:,-4:]
    else:
        data, labels = sequences[:,:,:-2], sequences[:,:,-2:]
    return data, labels