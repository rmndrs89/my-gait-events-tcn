import os
import pandas as pd
import numpy as np
from .preprocessing import resamp1d

def split_ids(path, by=["gender", "participant_type"]):
    """Splits the subjects intro three separate groups:
    (1) training set
    (2) validation set
    (3) test set

    Parameters
    ----------
    path : str
        Absolute or relative path to the root directory.
    by : list, optional
        A list of variables for which to stratify for, by default ["gender", "participant_type"]

    Returns
    -------
    train_ids, val_ids, test_ids : list, list, list
        A list of sujects ids, including the prefix `sub-`, for each set.
    """
    # Loop over the subject ids for which there is a data folder
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-pp")]
    keep_ids = []
    for sub_id in sub_ids:
        fnames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        # If there is an `events` file for the walking trials
        if len(fnames) > 0:
            keep_ids.append(sub_id)
    
    # Get dataframe with demographics of all subjects
    df_subjects = pd.read_csv(os.path.join(path, "participants.tsv"), sep="\t", header=0)
    
    # Select only subjects with annotated gait events
    df_select = df_subjects.loc[df_subjects["sub"].isin([sub_id[-5:] for sub_id in keep_ids])]
    
    # Group subjects by gender and participant type (i.e., disease or diagnosis)
    train_ids, val_ids, test_ids = [], [], []
    groups = df_select.groupby(by=by)
    for grp, df_group in groups:
        num_val_instances = len(df_group)//3
        num_test_instances = len(df_group)//3
        num_train_instances = len(df_group) - num_val_instances - num_test_instances
        ix_random = np.arange(len(df_group))
        np.random.shuffle(ix_random)
        train_ids += df_group.iloc[ix_random[:num_train_instances]]["sub"].tolist()
        test_ids += df_group.iloc[ix_random[num_train_instances:num_train_instances+num_test_instances]]["sub"].tolist()
        val_ids += df_group.iloc[ix_random[num_train_instances+num_test_instances:]]["sub"].tolist()
    
    # Add prefix
    train_ids = ["sub-"+train_id for train_id in train_ids]
    val_ids = ["sub-"+val_id for val_id in val_ids]
    test_ids = ["sub-"+test_id for test_id in test_ids]
    return train_ids, val_ids, test_ids

def load_file(filename, tracked_points=[], incl_magn=False, normalize=True, visualize=False):
    """Process the data from a single data file.
    For each file, check if we have both marker data, IMU data, and an events file.
    Then for each of the tracked points, make an instance with data and targets.

    Parameters
    ----------
    filename : str
        Absolute or relative path to the events file.
    tracked_points : list, optional
        A list of tracked points corresponding to the body segments, by default []
    incl_magn : bool, optional
        Whether to include magnetometer readings as well, by default False
    normalize : bool, optional
        Whether to normalize the sensor data, by default True
    visualize : bool, optional
        Whether to generate a plot, by default False

    Returns
    -------
    instances : list
        A list of instances, where each instance is a dictionary on itself.
        The dictionary entries are:
            filename_prefix : str
                The prefix of the filename holding infos on the subject id, walking task, and run type.
            tracked_point : str
                The tracked point, or body segment the IMU was attached to.
            data : (N, D) numpy array
                A numpy array with N time steps across D sensor channels.
            targets : dictionary
                A dictionary with entries for the initial and final contacts:
                initial_contact_left/right : (N, 1) array
                final_contact_left/right : (N, 1) array
    """
    # Fixed params
    fwhm = 40//5
    
    # Preliminary checks
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-omc_motion.tsv'):s} does not exist. Skip file.")
        return
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-imu_motion.tsv'):s} does not exist. Skip file.")
        return
    if len(tracked_points) == 0:
        print(f"No tracked points were defined.")
        return
    
    # Split filename
    _, fname = os.path.split(filename)
    
    # Get data from markers and inertial measurement units
    df_omc = pd.read_csv(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv"), sep="\t", header=0)
    df_imu = pd.read_csv(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv"), sep="\t", header=0)

    # If sampling frequencies do not match, then upsample IMU sensor data
    df_imu_channels = pd.read_csv(filename.replace("_events.tsv", "_tracksys-imu_channels.tsv"), sep="\t", header=0)
    if df_imu_channels["sampling_frequency"].iloc[0] != 200:
        X = df_imu.to_numpy()
        X = resamp1d(X, df_imu_channels["sampling_frequency"].iloc[0], 200)
        df_imu = pd.DataFrame(data=X, columns=df_imu.columns)
        del X
    
    # Get annotated events
    df_events = pd.read_csv(filename, sep="\t", header=0)
    ix_start  = df_events[df_events["event_type"]=="start"]["onset"].values[0]
    ix_end    = df_events[df_events["event_type"]=="stop"]["onset"].values[0]
    
    # Get target values
    targets = {}
    for event_type in df_events["event_type"].unique():
        if (event_type != "start") and (event_type != "stop"):
            indx = df_events[df_events["event_type"]==event_type]["onset"].values
            probs= np.zeros((len(df_omc),))
            for i in range(len(indx)):
                x = np.exp(-((4 * np.log(2) * (np.arange(len(df_omc)) - indx[i])**2)/(fwhm**2)))
                x[x < 0.1] = 0.0
                probs += x
            targets[event_type] = np.expand_dims(probs, axis=-1)
            
    # Segment target values from start to end
    for k in targets.keys():
        targets[k] = targets[k][ix_start:ix_end]
    
    # Initialize list to store data and targets
    instances = []
    
    # For each tracked point
    for tracked_point in tracked_points:
        
        # Get the corresponding columns
        if incl_magn == True:
            col_names = [col_name for col_name in df_imu.columns if (tracked_point in col_name) and ("_MAGN" in col_name)]
        else:
            col_names = [col_name for col_name in df_imu.columns if (tracked_point in col_name) and ("_MAGN" not in col_name)]
        
        # If no columns were found
        if len(col_names) == 0:
            continue  # to next tracked point
            
        # Get the IMU sensor data
        data = df_imu[col_names].iloc[ix_start:ix_end].to_numpy()
        if any(data.std(axis=0) < 1e-8):
            continue # to next tracked point
        
        # Normalize
        if normalize == True:
            data = ( data - data.mean(axis=0, keepdims=True) ) / data.std(axis=0, keepdims=True)
        
        # Get only targets from corresponding limb
        if "left_" in tracked_point:
            targets_ = {key.replace("_left",""): value for key, value in targets.items() if "left" in key}
        elif "right_" in tracked_point:
            targets_ = {key.replace("_right",""): value for key, value in targets.items() if "right" in key}
        
        # Append to list
        instances.append({
            "filename_prefix": fname.replace("_events.tsv", ""),
            "tracked_point": tracked_point,
            "data": data,
            "targets": targets_
        })
    return instances

def load_dataset(path, sub_ids=[], tracked_points=[], incl_magn=False, normalize=True):
    """Generate a dataset of instances by processing all available data files for a list subject ids.

    Parameters
    ----------
    path : str
        Absolute or relative path to the root directory.
    sub_ids : list, optional
        A list of subject ids, by default []
    tracked_points : list, optional
        A list of tracked points corresponding to the body segments, by default []
    incl_magn : bool, optional
        Whether to include magnetometer readings as well, by default False
    normalize : bool, optional
        Whether to normalize the sensor data, by default True
    visualize : bool, optional
        Whether to generate a plot, by default False

    Returns
    -------
    ds : list
        A list of instances, where each example is a dictionary on itself.
        The dictionary entries are:
            filename_prefix : str
                The prefix of the filename holding infos on the subject id, walking task, and run type.
            tracked_point : str
                The tracked point, or body segment the IMU was attached to.
            data : (N, D) numpy array
                A numpy array with N time steps across D sensor channels.
            targets : dictionary
                A dictionary with entries for the initial and final contacts:
                initial_contact_left/right : (N, 1) array
                final_contact_left/right : (N, 1) array
    """
    if len(tracked_points)<1:
        print(f"No tracked points were specified. Abort program.")
        return
    if len(sub_ids)==0:
        print(f"No subject ids were specified. Include all subjects from `{path:s}`.")
        sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-")]
    
    kwargs = {
        "tracked_points": tracked_points,
        "incl_magn": incl_magn,
        "normalize": normalize
    }
    
    # Initialize empty list to store the dataset
    ds = []

    # Loop over subject ids
    for sub_id in sub_ids:

        # Get a list of event filenames
        filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]

        # Loop over files
        for ix_filename, filename in enumerate(filenames):
            
            # Get instances from current data file
            instances = load_file(
                os.path.join(path, sub_id, "motion", filename),
                **kwargs
            )
            ds += instances
    return ds