import os
from matplotlib.pyplot import plot
import pandas as pd
import numpy as np
from .plot_utils import plot_targets

def load_file(filename, tracked_points=[], visualize=False):
    # Fixed params
    fwhm = 40//5
    
    # Preliminary checks
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-omc_motion.tsv'):s} does not exist. Skip file.")
        return
    if not os.path.isfile(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv")):
        print(f"{filename.replace('_events.tsv', '_tracksys-imu_motion.tsv'):s} does not exist. Skip file.")
        return
    if len(tracked_points)<1:
        print(f"No tracked points were specified. Abort program.")
        return
    
    # Get data from markers and inertial measurement units
    df_omc = pd.read_csv(filename.replace("_events.tsv", "_tracksys-omc_motion.tsv"), sep="\t", header=0)
    df_imu = pd.read_csv(filename.replace("_events.tsv", "_tracksys-imu_motion.tsv"), sep="\t", header=0)
    
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
    
    # Segment target time series from start to end
    for k in targets.keys():
        targets[k] = targets[k][ix_start:ix_end]
    return data, targets