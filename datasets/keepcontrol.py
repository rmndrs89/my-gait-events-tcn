"""Keep Control Validation Study."""
import os, re
from utils.data_utils import load_data_from_file, make_train_val_set, split_groups
import pandas as pd
import numpy as np

def select_subs(path, sub_ids):
    delete_sub_ids = []
    for (i_sub_id, sub_id) in enumerate(sub_ids):
        event_filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if (re.search("sub-pp[0-9]{3}_task-walk[a-zA-Z]*_events.tsv", fname) is not None) or (re.search("sub-pp[0-9]{3}_task-walk[a-zA-Z]+_run-[a-zA-Z]+_events.tsv", fname) is not None)]
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
              step_len=200, 
              seed=None):
    """Loads the Keep Control Validation study dataset.
    
    """
    seed = np.random.seed(123) if seed is None else seed
    
    # Subjects demographics dataframe
    df_subjects = pd.read_csv(filename, sep="\t", header=0)
        
    # List of subject ids from the data folders
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-pp")]
    sub_ids = select_subs(path, sub_ids)
    
    # Only retain infos from subjects
    df_subjects = df_subjects.loc[df_subjects["sub"].isin([s[-5:] for s in sub_ids])]
    
    # Split subject ids in train, validation and test set -- stratified by participant type and gender
    train_ids, val_ids, test_ids = split_groups(df_participants=df_subjects, sub_ids=sub_ids)
        
    
    # # Generate train and validation set
    # (train_data, train_labels, train_filenames) = make_train_val_set(path, 
    #                                                                  train_ids,
    #                                                                  tracked_points=tracked_points,
    #                                                                  incl_magn=incl_magn,
    #                                                                  normalize=normalize,
    #                                                                  classification_task=classification_task,
    #                                                                  win_len=win_len,
    #                                                                  step_len=step_len)
    # (val_data, val_labels, val_filenames) = make_train_val_set(path, 
    #                                                            val_ids,
    #                                                            tracked_points=tracked_points,
    #                                                            incl_magn=incl_magn,
    #                                                            normalize=normalize,
    #                                                            classification_task=classification_task,
    #                                                            win_len=win_len,
    #                                                            step_len=step_len)        
    return train_ids, val_ids, test_ids #(train_data, train_labels, train_filenames, train_ids), (val_data, val_labels, val_filenames, val_ids), (test_ids)