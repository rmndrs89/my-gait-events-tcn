from winreg import EnumValue
from datasets import keepcontrol
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from utils.models import get_base_model
from sklearn.utils.class_weight import compute_class_weight

def main():
    # Load dataset
    ds_train, ds_val, ds_test = keepcontrol.load_data(path=PATH,
                                                      filename=DEMOGRAPHICS_FILE,
                                                      tracked_points=TRACKED_POINTS,
                                                      incl_magn=False,
                                                      classification_task=CLASSIFICATION_TASK,
                                                      win_len=WIN_LEN)

    # Separate data and labels
    (train_data, train_labels, train_filenames, train_ids) = ds_train
    (val_data, val_labels, val_filenames, val_ids) = ds_val

    # Shape: (batch_size, win_len, num_channels)
    print(f"Shape of train data: {train_data.shape}")
    print(f"Shape of train labels: {train_labels.shape}")
    print(f"Shape of val data: {val_data.shape}")
    print(f"Shape of val labels: {val_labels.shape}")

    # Convert labels to sparse categorical arrays
    if CLASSIFICATION_TASK == "events":    
        train_targets = np.zeros((train_labels.shape[0], train_labels.shape[1], 1), dtype=int)
        for m in range(train_labels.shape[-1]):
            indx = np.argwhere(train_labels[:,:,m]==1)
            for i in range(indx.shape[0]):
                train_targets[indx[i][0]][indx[i][1],0] = m+1

        val_targets = np.zeros((val_labels.shape[0], val_labels.shape[1], 1), dtype=int)
        for m in range(val_labels.shape[-1]):
            indx = np.argwhere(val_labels[:,:,m]==1)
            for i in range(indx.shape[0]):
                val_targets[indx[i][0]][indx[i][1],0] = m+1
    
    # Convert labels to one-hot encoded categorical arrays
    y_train = keras.utils.to_categorical(train_targets)
    y_val = keras.utils.to_categorical(val_targets)
    
    # Build model
    tcn_model = get_base_model(train_data.shape[1:])
    tcn_model.summary()

    class_weights = compute_class_weight("balanced", classes=np.unique(train_targets), y=np.ravel(train_targets))
    class_weights = dict(enumerate(class_weights))
    print(f"Class weights: {class_weights}")

    # Fit model
    history = tcn_model.fit(
        x=train_data,
        y=y_train,
        epochs=5,
        batch_size=32,
        validation_data=(val_data, y_val),
        shuffle=True,
        class_weight=class_weights
    )
    return

if __name__ == "__main__":
    # Global variables
    PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    DEMOGRAPHICS_FILE = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata\\participants.tsv"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    CLASSIFICATION_TASK = "events"
    WIN_LEN = 400
    
    # Call main function
    main()