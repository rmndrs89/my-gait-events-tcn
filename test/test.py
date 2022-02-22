from locale import normalize
import os, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.data_utils import load_dataset ,load_file
from utils.plot_utils import plot_targets
from utils.preprocessing import create_batch_sequences, create_sequences
from scipy.signal import find_peaks

def main():
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-pp")]
    # data, targets = load_file(
    #     os.path.join(ROOT_DIR, "sub-pp007", "motion", "sub-pp007_task-walkSlow_events.tsv"),
    #     tracked_points=TRACKED_POINTS,
    #     normalize=True,
    #     visualize=False
    # )
    for sub_id in sub_ids:

        event_filenames = [fname for fname in os.listdir(os.path.join(ROOT_DIR, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        for fname in event_filenames:
            print(f"Load file: {fname:s}")
            res = load_file(
                os.path.join(ROOT_DIR, sub_id, "motion", fname),
                tracked_points=TRACKED_POINTS, 
                normalize=True,
                visualize=False
            )
            if res is not None:
                (data, targets) = res
    return

if __name__ == "__main__":
    ROOT_DIR = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    main()