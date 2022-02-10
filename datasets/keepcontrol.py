"""Keep Control Validation Study."""
import enum
import os
from utils.data_utils import load_data_from_file

def load_data(path="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata", 
              filename="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv", 
              tracked_points=[],
              incl_magn=False,
              normalize=True,
              classification_task="events", 
              win_len=200, 
              step_len=200):
    """Loads the Keep Control Validation study dataset.
    
    """
    # Get a list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(path) if sub_id.startswith("sub-pp")]
    
    # Loop over the subject ids
    for (i_sub_id, sub_id) in enumerate(sub_ids[:1]):
        
        # Get a list of files
        filenames = [fname for fname in os.listdir(os.path.join(path, sub_id, "motion")) if fname.endswith("_tracksys-imu_motion.tsv")]
        
        # Loop over the filenames
        for (i_filename, filename) in enumerate(filenames):
            
            # Load data from file
            data, labels = load_data_from_file(filename=os.path.join(path, sub_id, "motion", filename), 
                                               tracked_points=tracked_points, 
                                               incl_magn=incl_magn,
                                               normalize=normalize,
                                               classification_task=classification_task, 
                                               win_len=win_len,
                                               step_len=step_len)
            
    return data, labels