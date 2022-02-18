from sklearn.utils import shuffle
from datasets import keepcontrol
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary
from utils.models import get_base_model, TCNHyperModel
from utils.losses import MyWeightedBinaryCrossentropy
from sklearn.utils.class_weight import compute_class_weight
import keras_tuner as kt

def tune(train_data, train_targets, validation_data, weights=None):
    print(f"Start hyperparameter tuning ...")
    # Define tuner    
    tuner = kt.RandomSearch(
        hypermodel=TCNHyperModel(
            input_shape=train_data.shape[1:],
            num_classes=len(train_targets),
            weights=weights
        ),
        objective="val_loss",
        max_trials=MAX_TRIALS,
        executions_per_trial=EXECUTIONS_PER_TRIAL,
        overwrite=True,
        directory="/home/robbin/Desktop/tuning",
        project_name="alpha"
    )
    
    # Search hyperparameter space
    tuner.search(
        train_data,
        train_targets,
        epochs=EPOCHS,
        validation_data=validation_data,
        shuffle=True,
        callbacks=[EARLY_STOPPING, REDUCE_LR],
        verbose=0
    )
    return tuner
    
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
    
    # Re-organize labels -> targets
    train_targets, val_targets = {}, {}
    for i in range(train_labels.shape[-1]):
        train_targets[f"outputs_{i+1}"] = np.expand_dims(train_labels[:,:,i], axis=-1)
        val_targets[f"outputs_{i+1}"] = np.expand_dims(val_labels[:,:,i], axis=-1)
    
    # Hyperparameter tuning
    tuner = tune(
        train_data=train_data,
        train_targets=train_targets,
        validation_data=(val_data, val_targets)
    )
    
    # Get hyperparameters
    best_hps = tuner.get_best_hyperparameters()[0]
    print(f"Found best model architecture")
    print(f"    # filters: {2**best_hps.get('nb_filters'):d}")
    print(f"    kernel size: {best_hps.get('kernel_size'):d}")
    print(f"    padding: {best_hps.get('padding'):s}")
    print(f"    dilations: {[2**i for i in range(best_hps.get('dilations'))]}")
    
    # Build model
    optim_model = tuner.hypermodel.build(best_hps)
    
    # Concatenate training and validation data
    train_val_data = np.concatenate((train_data, val_data), axis=0)
    train_val_targets = {}
    for k, v in train_targets.items():
        train_val_targets[k] = np.concatenate((train_targets[k], val_targets[k]), axis=0)
    
    # Fit optimized model to combined training and validation data
    history = optim_model.fit(
        x=train_val_data,
        y=train_val_targets,
        batch_size=64, 
        epochs=EPOCHS,
        shuffle=True
    )
    return

if __name__ == "__main__":
    # Global variables
    PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    DEMOGRAPHICS_FILE = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata\\participants.tsv"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    CLASSIFICATION_TASK = "events"
    WIN_LEN = 400
    
    # Hyperparameter tuning
    EPOCHS = 20
    MAX_TRIALS = 15
    EXECUTIONS_PER_TRIAL = 3
    OUTPUT_DIR = "/home/robr/Desktop/tuning"
    
    # Training callbacks
    EARLY_STOPPING = keras.callbacks.EarlyStopping(
        patience=5,
        monitor="val_loss"
    )
    REDUCE_LR = keras.callbacks.ReduceLROnPlateau(
        patience=5,
        monitor="val_loss",
        factor=0.1
    )
    
    # Call main function
    main()