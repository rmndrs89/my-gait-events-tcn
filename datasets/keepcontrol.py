"""Keep Control Validation Study."""
import os
import _pickle as cp
from utils.data_utils import get_dataset, split_groups
import pandas as pd
import numpy as np

def select_subs(path, sub_ids):
    """Select only subjects for which we have annotated gait events.
    
    Note:
    That is currently only the straight walking trials at preferred, fast and slow speed.

    Parameters
    ----------
    path : str
        The absolute or relative path to the data files.
    sub_ids : list
        A list of subject ids for which there is a data folder.

    Returns
    -------
    _ : list
        A list of subject ids to include in the analysis.
    """
    delete_sub_ids = []
    for (i_sub_id, sub_id) in enumerate(sub_ids):
        event_filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if ("_task-walk" in fname) and ("_events.tsv" in fname)]
        if len(event_filenames) < 1:
            delete_sub_ids += [sub_id]
    return [sub_id for sub_id in sub_ids if sub_id not in delete_sub_ids]

def load_data(path="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata", 
              filename="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv",
              tracked_points=[],
              incl_magn=False,
              normalize=True,
              classification_task="events", 
              win_len=200, 
              seed=None):
    """Loads the Keep Control Validation study dataset.
    
    """
    # Check if dataset has already been prepared
    derivatives_path = os.path.join(os.path.split(path)[0], "derivatives", "motion", "doe")
    derivatives_filename = os.path.join(derivatives_path, f"run_class-{classification_task:s}_winLen-{win_len:3d}.pkl")
    if os.path.isfile(derivatives_filename):
        with open(derivatives_filename, 'rb') as infile:
            print(f"Load dataset from pickle ...")
            ds_train, ds_val, ds_test = cp.load(infile)
    else:       
        if len(tracked_points) == 0:
            print(f"No tracked points were defined. Exit program.")
            return
        seed = np.random.seed(123) if seed is None else seed  # for reproducible results
        print(f"Creating dataset ...")
            
        # Subjects demographics dataframe
        df_subjects = pd.read_csv(filename, sep="\t", header=0)
            
        # List of subject ids from the data folders
        sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-pp")]
        sub_ids = select_subs(path, sub_ids)
        
        # Only retain infos from subjects for whom we have annotated gait events
        df_subjects = df_subjects.loc[df_subjects["sub"].isin([s[-5:] for s in sub_ids])]
        
        # Split subject ids in train, validation and test set -- stratified by participant type and gender
        train_ids, val_ids, test_ids = split_groups(df_subjects=df_subjects)
            
        # Generate train and validation set
        (train_data, train_labels, train_filenames) = get_dataset(path, 
                                                                train_ids,
                                                                tracked_points=tracked_points,
                                                                incl_magn=incl_magn,
                                                                normalize=normalize,
                                                                classification_task=classification_task,
                                                                win_len=win_len,
                                                                training=True)
        (val_data, val_labels, val_filenames) = get_dataset(path, 
                                                            val_ids,
                                                            tracked_points=tracked_points,
                                                            incl_magn=incl_magn,
                                                            normalize=normalize,
                                                            classification_task=classification_task,
                                                            win_len=win_len)
        ds_train = (train_data, train_labels, train_filenames, train_ids)
        ds_val = (val_data, val_labels, val_filenames, val_ids)
        ds_test = (test_ids)
        
        # Dump to pickle
        with open(derivatives_filename, 'wb') as outfile:
            cp.dump([ds_train, ds_val, ds_test], outfile)
    return ds_train, ds_val, ds_test