from fileinput import filename
from locale import normalize
import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.data_utils import load_dataset, load_file, split_ids, split_left_right
from utils.plot_utils import plot_targets
from utils.preprocessing import create_batch_sequences, create_sequences
from scipy.signal import find_peaks
import timeit

def main():
    # Get list of all subject for which we have any kind of data
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-pp")]
    
    # Split the subject ids into separate lists
    train_ids, val_ids, test_ids = split_ids(ROOT_DIR)
    
    # Get a training dataset
    ds_train = []
    start_time = timeit.default_timer()
    for sub_id in train_ids:
        filenames = [fname for fname in os.listdir(os.path.join(ROOT_DIR, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        for ix_filename, filename in enumerate(filenames):
            examples = load_file(
                os.path.join(ROOT_DIR, sub_id, "motion", filename),
                tracked_points=TRACKED_POINTS,
                incl_magn=INCL_MAGN,
                normalize=NORMALIZE
            )
            ds_train += examples
    elapsed_time = timeit.default_timer() - start_time
    print(f"Training dataset completed, run {elapsed_time}")
    
    # Get a validation dataset
    ds_val = []
    for sub_id in val_ids:
        filenames = [fname for fname in os.listdir(os.path.join(ROOT_DIR, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        for ix_filename, filename in enumerate(filenames):
            examples = load_file(
                os.path.join(ROOT_DIR, sub_id, "motion", filename),
                tracked_points=TRACKED_POINTS,
                incl_magn=INCL_MAGN,
                normalize=NORMALIZE
            )
            ds_val += examples
    elapsed_time = timeit.default_timer() - start_time
    print(f"Validation dataset completed, run {elapsed_time}")
    
    # Get a test dataset
    ds_test = []
    for sub_id in test_ids:
        filenames = [fname for fname in os.listdir(os.path.join(ROOT_DIR, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        for ix_filename, filename in enumerate(filenames):
            examples = load_file(
                os.path.join(ROOT_DIR, sub_id, "motion", filename),
                tracked_points=TRACKED_POINTS,
                incl_magn=INCL_MAGN,
                normalize=NORMALIZE
            )
            ds_test += examples
    elapsed_time = timeit.default_timer() - start_time
    print(f"Test dataset completed, run {elapsed_time}")
    return

if __name__ == "__main__":
    ROOT_DIR = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    # TRACKED_POINTS = ["left_ankle", "right_ankle", "left_shank", "right_shank"]
    TRACKED_POINTS = ["left_ankle", "right_ankle", "left_shank", "right_shank"]
    INCL_MAGN = False
    NORMALIZE = True
    WIN_LEN = 400
    STEP_LEN = 200
    main()