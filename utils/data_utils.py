import pandas as pd
import os, re
import numpy as np
from utils.preprocessing import resamp1d, create_sequences

def split_groups(df_subjects):
    """Splits the subject ids into three different sets:
    (1) training set,
    (2) validation set,
    (3) test set

    Parameters
    ----------
    df_subjects : pandas DataFrame
        A pandas DataFrame with the subjects' demographics infos.

    Returns
    -------
    train_ids, val_ids, test_ids : list, list, list
        A list of subject ids for each of the three sets.
    """
    # Create groupby objects by participant type and gender
    groups = df_subjects.groupby(["participant_type", "gender"])
    train_ids, val_ids, test_ids = [], [], []
    
    # For each groupby object ..
    # .. assign one-third to the validation set, one-third to the test set
    # .. and the remaining subject ids to the training set
    for dmy, df_group in groups:
        num_val_examples = len(df_group)//3
        num_test_examples = len(df_group)//3
        num_train_examples = len(df_group) - num_test_examples - num_val_examples
        indx_random = np.arange(len(df_group))
        np.random.shuffle(indx_random)
        train_ids += df_group.iloc[indx_random[:num_train_examples]]["sub"].tolist()
        test_ids += df_group.iloc[indx_random[num_train_examples:num_train_examples+num_test_examples]]["sub"].tolist()
        val_ids += df_group.iloc[indx_random[num_train_examples+num_test_examples:]]["sub"].tolist()
    
    # Add prefix
    train_ids = ["sub-"+train_id for train_id in train_ids]
    val_ids = ["sub-"+val_id for val_id in val_ids]
    test_ids = ["sub-"+test_id for test_id in test_ids]
    return train_ids, val_ids, test_ids

def select_data(df_data, df_channels, tracked_points=[], incl_magn=False):
    """Select the data columns that are included in the analysis.
    
    The tracked points correspond to the device body location, and
    we can choose whether to include magnetometer readings in the analysis.

    Parameters
    ----------
    df_data : pandas DataFrame
        A pandas DataFrame that contains the IMU sensor data.
    df_channels : pandas DataFrame
        A pandas DataFrame that contains infos about the IMU sensor channels.
    tracked_points : list, optional
        A list of tracked points, by default []
    incl_magn : bool, optional
        Whether to include magnetometer readings, by default False

    Returns
    -------
    _ : pandas DataFrame
        A pandas DataFrame with only IMU sensor data from the given tracked points.
    """
    col_names = df_channels.loc[df_channels["tracked_point"].isin(tracked_points)]["name"].values
    df_data = df_data[col_names]
    
    if incl_magn == True:
        return df_data
    else:
        col_names = [col_name for col_name in df_data.columns if not "MAGN" in col_name]
        return df_data[col_names]

def get_gait_events(df_events):
    """Get indices of annotated gait events.

    Parameters
    ----------
    df_events : pandas DataFrame
        A pandas DataFrame with annotated gait events.

    Returns
    -------
    _ : dict
        A dictionary with for each type of gait events an array of indices corresponding to the gait events.
    """
    indx_start = df_events[df_events["event_type"]=="start"]["onset"].values[0] - 1
    return {"ICL": df_events[df_events["event_type"]=="initial_contact_left"]["onset"].values-1 - indx_start, 
            "FCL": df_events[df_events["event_type"]=="final_contact_left"]["onset"].values-1 - indx_start, 
            "ICR": df_events[df_events["event_type"]=="initial_contact_right"]["onset"].values-1 - indx_start, 
            "FCR": df_events[df_events["event_type"]=="final_contact_right"]["onset"].values-1 - indx_start}

def get_labels(num_time_steps, events, classification_task="events"):
    """Get labels for the classification task.

    Parameters
    ----------
    num_time_steps : _type_
        _description_
    events : _type_
        _description_
    classification_task : str, optional
        _description_, by default "events"

    Returns
    -------
    labels : (num_time_steps, K) array_like
        A numpy array with the labels for the gait events or phases.
    """
    # Map event type to index
    map_events_to_int = {"ICL": 0, "FCL": 1, "ICR": 2, "FCR": 3}
    
    # Initialize zeros array
    labels = np.zeros((num_time_steps, len(events))) if classification_task == "events" else np.zeros((num_time_steps, len(events)//2))    
    
    if classification_task == "events":    
        # Iterate over the events dictionary
        for event_type, indices in events.items():
            for i in range(len(indices)):
                labels[indices[i], map_events_to_int[event_type]] = 1.0
    else:
        # Labels the swing phase as the positive class (-> 1.0)
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

def load_data_from_file(filename, tracked_points=[], incl_magn=False, normalize=True, classification_task="events", win_len=200, training=False):
    """Load data from a single data file.

    Parameters
    ----------
    filename : str
        The filename of the data file that contains the sensor data.
    tracked_points : list, optional
        A list of tracked points to be included, by default []
    incl_magn : bool, optional
        Whether to include magnetometer readings, by default False
    normalize : bool, optional
        Whether to normalize the sensor data, by default True
    classification_task : str, optional
        Which classification task, by default "events". 
        Currently, choose between "events" or "phases", where ..
        .. "events" means individual samples are labelled 0 or 1.
        .. "phases" means the swing phase is labelled 1.
    win_len : int, optional
        The window length used to generate sequences of equal length, by default 200
    training : bool, optional
        Whether it concerns data for training, by default False
        If it concerns training data, then we use a sliding window with 
        an overlap of circa 75% to artificially increase the number of training examples.

    Returns
    -------
    data, labels : (batch_size, win_len, num_channels), (batch_size, win_len, num_classes)
        Numpy array for the sensor data, and corresponding labels.
    """
    if len(tracked_points) == 0:
        return
    
    # Load data
    df_data = pd.read_csv(filename, sep="\t", header=0)
    
    # Load channels info
    df_channels = pd.read_csv(filename.replace("_motion", "_channels"), sep="\t", header=0)
    
    # Check sampling frequency
    if df_channels["sampling_frequency"].iloc[0] != 200:
        X = df_data.to_numpy()
        X = resamp1d(X, df_channels["sampling_frequency"].iloc[0], 200)
        df_data = pd.DataFrame(data=X, columns=df_data.columns)
        del X
    
    # Select data from given tracked points, and magnetometer readings, if wanted
    df_select = select_data(df_data, df_channels, tracked_points=tracked_points, incl_magn=incl_magn)
       
    # Determine start and end of current task trial
    df_events = pd.read_csv(filename.replace("_tracksys-imu_motion", "_events"), sep="\t", header=0)
    indx_start = df_events[df_events["event_type"]=="start"]["onset"].values[0] - 1
    indx_stop  = df_events[df_events["event_type"]=="stop"]["onset"].values[0]
    df_events = df_events.loc[(df_events["onset"]>=indx_start) & (df_events["onset"]<=indx_stop)]
    df_select = df_select.iloc[indx_start:indx_stop]
    
    # Normalize
    if normalize:
        df_select = ( df_select - df_select.mean() ) / df_select.std()
    
    # Get indices corresponding to gait events
    events = get_gait_events(df_events=df_events)
    
    # Get labels
    labels = get_labels(len(df_select), events, classification_task=classification_task)
    
    # Combine (normalized) data and labels in array
    data = np.concatenate((df_select.to_numpy(), labels), axis=1)
    
    # Create sequences of equal length
    if training == True:
        sequences = create_sequences(data, win_len=win_len, step_len=win_len//4)
    else:
        sequences = create_sequences(data, win_len=win_len, step_len=win_len)
    
    # Split data and labels
    if classification_task == "events":
        data, labels = sequences[:,:,:-4], sequences[:,:,-4:]
    else:
        data, labels = sequences[:,:,:-2], sequences[:,:,-2:]
    return data, labels

def get_dataset(path, sub_ids, tracked_points, incl_magn, normalize, classification_task, win_len, training=False):
    """Get a dataset.

    Parameters
    ----------
    path : str
        The absolute or relative path to the root directory where data files are stored.
    sub_ids : list
        A list of subject ids to be included in the current dataset.
    tracked_points : list
        A list of tracked points.
    incl_magn : bool
        Whether to include magnetometer readings.
    normalize : bool
        Whether to normalize the sensor data.
    classification_task : str
        The type of classification task, either "events" or "phases".
    win_len : int
        The length of the sliding window used to create sequences of equal length.
    training : bool, optional
        Whether it concerns data for training, by default False

    Returns
    -------
    (data, labels, list_filenames) : (batch_size, win_len, num_channels) array_like, (batch_size, win_len, num_classes) array_like, list
        Numpy array for the sensor data, and corresponding labels, as well as a list of filenames.
    """
    
    # Initialize output arrays -- X: data array, y: labels array, f: filenames list
    data, labels, list_filenames = [], [], []
    
    # Loop over the subject ids    
    for (i_sub_id, sub_id) in enumerate(sub_ids):
        
        # Get a list of event filenames -- only process files for which we have annotated gait events
        filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if ("_task-walk" in fname) and ("_events.tsv" in fname)]
        
        # Loop over the filenames
        for (i_filename, filename) in enumerate(filenames):   
                       
            # Check if data from the tracked points are available
            df_imu_channels = pd.read_csv(os.path.join(path, sub_id, "motion", filename.replace("_events", "_tracksys-imu_channels")), sep="\t", header=0)
            if not(any([re.search(tracked_point, chan) for chan in df_imu_channels["name"].tolist() for tracked_point in tracked_points])):
                continue
            
            # Load data from file
            X, y = load_data_from_file(filename=os.path.join(path, sub_id, "motion", filename.replace("_events", "_tracksys-imu_motion")),
                                       tracked_points=tracked_points,
                                       incl_magn=incl_magn,
                                       normalize=normalize, 
                                       classification_task=classification_task,
                                       win_len=win_len,
                                       training=training)
                
            # Accumulate data over multiple files
            if len(data)>0:
                data = np.concatenate((data, np.concatenate((X[:,:,:X.shape[-1]//len(tracked_points)], X[:,:,X.shape[-1]//len(tracked_points):]), axis=0)), axis=0)
                labels = np.concatenate((labels, np.concatenate((y[:,:,:y.shape[-1]//len(tracked_points)], y[:,:,y.shape[-1]//len(tracked_points):]), axis=0)), axis=0)
            else:
                data = np.concatenate((X[:,:,:X.shape[-1]//len(tracked_points)], X[:,:,X.shape[-1]//len(tracked_points):]), axis=0)
                labels = np.concatenate((y[:,:,:y.shape[-1]//len(tracked_points)], y[:,:,y.shape[-1]//len(tracked_points):]), axis=0)
            list_filenames += [filename for _ in range(X.shape[0]*len(tracked_points))]
    return (data, labels, list_filenames)