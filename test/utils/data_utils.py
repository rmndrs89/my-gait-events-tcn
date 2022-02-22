import enum
import os
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from .preprocessing import resamp1d, normalize_data
from .plot_utils import plot_targets

def load_file(filename, tracked_points=[], normalize=True, visualize=False):
    # Fixed params
    fwhm = 40//5
    
    # Preliminary checks
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-omc_motion.tsv'):s} does not exist. Skip file.")
        return
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-imu_motion.tsv'):s} does not exist. Skip file.")
        return
        
    # Get data from markers and inertial measurement units
    df_omc = pd.read_csv(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv"), sep="\t", header=0)
    df_imu = pd.read_csv(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv"), sep="\t", header=0)

    # Check if tracked points are available for current trial
    for tracked_point in tracked_points:
        cols = [col for col in df_imu.columns if tracked_point in col]
        if len(cols) == 0:
            print(f"{filename:s} contains no data for (at least) the {tracked_point:s} sensor. Skip file.")
            return

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
    
    # Initialize dict to store target values
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
    
    # Call function to plot
    if visualize:
        plot_targets(df_omc, targets, ix_start=ix_start, ix_end=ix_end)
    
    # Get data from tracked points of interest
    select_cols = []
    for tracked_point in tracked_points:
        select_cols += [col for col in df_imu.columns if tracked_point in col]
    data = df_imu[select_cols].iloc[ix_start:ix_end].to_numpy()
    
    # Normalize
    if normalize:
        normalized_data = normalize_data(data)
        if normalized_data is not None:
            data = normalized_data
        else:
            return
    
    # Segment target time series from start to end
    for k in targets.keys():
        targets[k] = targets[k][ix_start:ix_end]
    return data, targets

def load_dataset(path, sub_ids=[], tracked_points=[], normalize=True):
    if len(tracked_points)<1:
        print(f"No tracked points were specified. Abort program.")
        return
    if len(sub_ids)==0:
        sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-")]
    
    # Initialize empty list to store (filename, data, target) tuples
    ds = []

    # Loop over subject ids
    for (_, sub_id) in enumerate(sub_ids):

        # Get a list of event filenames
        event_filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]

        # Loop over files
        for (_, filename) in enumerate(event_filenames):
            # Get filename prefix
            filename_prefix = filename.replace("_events.tsv", "")

            # Get response
            # ... https://pythoncircle.com/post/708/solving-python-error-typeerror-nonetype-object-is-not-iterable/ 
            res = load_file(os.path.join(path, sub_id, "motion", filename), tracked_points=tracked_points, normalize=normalize)
            if res is not None:
                # If response is not of NoneType
                (data, targets) = res
                
                # Append to list
                ds.append({"filename_prefix": filename_prefix, "data": data, "targets": targets})
    return ds