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

def main():
    sub_ids = [sub_id for sub_id in os.listdir(ROOT_DIR) if sub_id.startswith("sub-pp")]
    train_ids, val_ids, test_ids = split_ids(ROOT_DIR)
    print(f"# train examples: {len(train_ids):d}")
    print(f"# val examples: {len(val_ids):d}")
    print(f"# test examples: {len(test_ids):d}")
    ds_train = load_dataset(ROOT_DIR, sub_ids=train_ids, tracked_points=TRACKED_POINTS, normalize=True, split_lr=True)
    train_data, train_targets, train_examples = create_batch_sequences(ds_train, win_len=WIN_LEN, step_len=STEP_LEN)
    # data, targets = load_file(
    #     os.path.join(ROOT_DIR, "sub-pp007", "motion", "sub-pp007_task-walkSlow_events.tsv"),
    #     tracked_points=TRACKED_POINTS,
    #     normalize=True,
    #     visualize=False
    # )
    # for sub_id in sub_ids:

    #     event_filenames = [fname for fname in os.listdir(os.path.join(ROOT_DIR, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
    #     for fname in event_filenames:
    #         print(f"Load file: {fname:s}")
    #         res = load_file(
    #             os.path.join(ROOT_DIR, sub_id, "motion", fname),
    #             tracked_points=TRACKED_POINTS, 
    #             normalize=True,
    #             visualize=False
    #         )
    #         if res is not None:
    #             (data, targets) = res
    return

if __name__ == "__main__":
    ROOT_DIR = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    WIN_LEN = 400
    STEP_LEN = 200
    main()