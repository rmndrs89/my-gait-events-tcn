import os
from datasets import keepcontrol

def main():
    # (train_data, train_labels, train_filenames, train_ids), (val_data, val_labels, val_filenames, val_ids), (test_ids) = keepcontrol.load_data(path="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata",
    #                                                                                                                                            filename="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv",
    #                                                                                                                                            tracked_points=["left_ankle", "right_ankle"], 
    #                                                                                                                                            classification_task="phases", 
    #                                                                                                                                            win_len=400)
    (train_ids, val_ids, test_ids) = keepcontrol.load_data(path="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata",
                                                           filename="/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv",
                                                           tracked_points=["left_ankle", "right_ankle"], 
                                                           classification_task="phases", 
                                                           win_len=400)
    return

if __name__ == "__main__":
    main()