# Import libraries
from datasets import keepcontrol
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tcn import TCN, tcn_full_summary
from utils.losses import MyWeightedBinaryCrossentropy
from utils.evaluate import compare_events
from scipy.signal import find_peaks
from utils.data_utils import select_data, get_gait_events, get_labels
from utils.preprocessing import resamp1d
from datetime import datetime

def main():
    # Load datasets
    ds_train, ds_val, ds_test = keepcontrol.load_data(
        path=PATH,
        filename=DEMOGRAPHICS_FILE,
        tracked_points=TRACKED_POINTS,
        incl_magn=False,
        classification_task=CLASSIFICATION_TASK,
        win_len=WIN_LEN
    )
    
    # Load model, set compile=False because custom loss cannot be loaded
    tcn_model = keras.models.load_model(CHECKPOINT_FILEPATH, compile=False)

    # Compile the model, using same as before
    LOSSES, METRICS = {}, {}
    for i in range(NUM_CLASSES):
        LOSSES[f"outputs_{i+1}"] = MyWeightedBinaryCrossentropy()
        METRICS[f"outputs_{i+1}"] = keras.metrics.BinaryAccuracy()
    tcn_model.compile(loss=LOSSES, optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=METRICS)
    
    # Output variables
    out_sub_ids = []
    out_filenames = []
    out_event_types = []
    out_reference_timings = []
    out_predicted_timings = []

    test_ids = ds_test
    for (i_sub_id, sub_id) in enumerate(test_ids[39:]):    
        test_filenames = [fname for fname in os.listdir(os.path.join(PATH, sub_id, "motion")) if (fname.endswith("_events.tsv")) and ("_task-walk" in fname)]
        
        for (i_test_filename, test_filename) in enumerate(test_filenames):
            
            # Load the IMU motion and channels files
            df_imu = pd.read_csv(
                os.path.join(PATH, sub_id, "motion", test_filename.replace("_events.tsv", "_tracksys-imu_motion.tsv")), 
                sep="\t", 
                header=0
            )
            df_imu_channels = pd.read_csv(
                os.path.join(PATH, sub_id, "motion", test_filename.replace("_events.tsv", "_tracksys-imu_channels.tsv")), 
                sep="\t", 
                header=0
            )
            
            # Resample if necessary
            if df_imu_channels["sampling_frequency"].iloc[0] != 200:
                X = df_imu.to_numpy()
                X = resamp1d(X, df_imu_channels["sampling_frequency"].iloc[0], 200)
                df_imu = pd.DataFrame(data=X, columns=df_imu.columns)
                del X
            
            # Select data from given tracked points
            df_select = select_data(df_imu, df_imu_channels, tracked_points=TRACKED_POINTS, incl_magn=False)
            
            # Determine start and end of current task trial
            df_events = pd.read_csv(os.path.join(PATH, sub_id, "motion", test_filename), sep="\t", header=0)
            indx_start = df_events[df_events["event_type"]=="start"]["onset"].values[0] - 1
            indx_stop = df_events[df_events["event_type"]=="stop"]["onset"].values[0]
            df_events = df_events.loc[(df_events["onset"]>=indx_start) & (df_events["onset"]<=indx_stop)]
            df_select = df_select.iloc[indx_start:indx_stop]
            
            # Normalize
            normalize = True
            if normalize:
                df_select = ( df_select - df_select.mean() ) / df_select.std()
            
            # Get indices corresponding to gait events
            events = get_gait_events(df_events=df_events)
            
            # Get labels
            labels = get_labels(len(df_select), events, classification_task=CLASSIFICATION_TASK)
            
            # Convert data to numpy array
            data = df_select.to_numpy()
            
            # Split left/right
            data = np.stack([data[:,:data.shape[-1]//len(TRACKED_POINTS)], data[:,data.shape[-1]//len(TRACKED_POINTS):]], axis=0)
            labels = np.stack([labels[:,:labels.shape[-1]//len(TRACKED_POINTS)], labels[:,labels.shape[-1]//len(TRACKED_POINTS):]], axis=0)
            
            # Make predictions
            predictions = tcn_model.predict(data)
            
            # Get indices from annotated and predicted events
            ix_true_ICL = np.argwhere(labels[0][:,0]==1)[:,0]
            ix_true_FCL = np.argwhere(labels[0][:,1]==1)[:,0]
            ix_true_ICR = np.argwhere(labels[1][:,0]==1)[:,0]
            ix_true_FCR = np.argwhere(labels[1][:,1]==1)[:,0]

            ix_pred_ICL, pk_props_ICL = find_peaks(predictions[0][0][:,0], height=0.5, distance=50)
            ix_pred_FCL, pk_props_FCL = find_peaks(predictions[1][0][:,0], height=0.5, distance=50)
            ix_pred_ICR, pk_props_ICR = find_peaks(predictions[0][1][:,0], height=0.5, distance=50)
            ix_pred_FCR, pk_props_FCR = find_peaks(predictions[1][1][:,0], height=0.5, distance=50)
            
            # For each gait event, determine the time error
            ann2pred_ICL, pred2ann_ICL, time_difference_ICL = compare_events(ix_true_ICL, ix_pred_ICL)
            ann2pred_FCL, pred2ann_FCL, time_difference_FCL = compare_events(ix_true_FCL, ix_pred_FCL)
            ann2pred_ICR, pred2ann_ICR, time_difference_ICR = compare_events(ix_true_ICR, ix_pred_ICR)
            ann2pred_FCR, pred2ann_FCR, time_difference_FCR = compare_events(ix_true_FCR, ix_pred_FCR)
            if (ann2pred_ICL is None) or (pred2ann_ICL is None):
                continue
            if (ann2pred_FCL is None) or (pred2ann_FCL is None):
                continue
            if (ann2pred_ICR is None) or (pred2ann_ICR is None):
                continue
            if (ann2pred_FCR is None) or (pred2ann_FCR is None):
                continue
            
            # Left Initial Contacts
            for i in range(len(ix_true_ICL)-1, -1, -1):
                if ann2pred_ICL[i] >- 999:
                    out_predicted_timings.append(ix_pred_ICL[ann2pred_ICL[i]])
                    ix_pred_ICL = np.delete(ix_pred_ICL, ann2pred_ICL[i])
                    pred2ann_ICL = np.delete(pred2ann_ICL, ann2pred_ICL[i])
                else:
                    out_predicted_timings.append(np.nan)
                out_reference_timings.append(ix_true_ICL[i])
                out_event_types.append("ICL")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
            for i in range(len(ix_pred_ICL)-1, -1, -1):
                if pred2ann_ICL[i] > -999:
                    out_reference_timings.append(ix_true_ICL[pred2ann_ICL[i]])
                    ix_true_ICL = np.delete(ix_true_ICL, pred2ann_ICL[i])
                    ann2pred_ICL = np.delete(ann2pred_ICL, pred2ann_ICL[i])
                else:
                    out_reference_timings.append(np.nan)
                out_predicted_timings.append(ix_pred_ICL[i])
                out_event_types.append("ICL")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
                
            # Left Final Contacts
            for i in range(len(ix_true_FCL)-1, -1, -1):
                if ann2pred_FCL[i] >- 999:
                    out_predicted_timings.append(ix_pred_FCL[ann2pred_FCL[i]])
                    ix_pred_FCL = np.delete(ix_pred_FCL, ann2pred_FCL[i])
                    pred2ann_FCL = np.delete(pred2ann_FCL, ann2pred_FCL[i])
                else:
                    out_predicted_timings.append(np.nan)
                out_reference_timings.append(ix_true_FCL[i])
                out_event_types.append("FCL")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
            for i in range(len(ix_pred_FCL)-1, -1, -1):
                if pred2ann_FCL[i] > -999:
                    out_reference_timings.append(ix_true_FCL[pred2ann_FCL[i]])
                    ix_true_FCL = np.delete(ix_true_FCL, pred2ann_FCL[i])
                    ann2pred_FCL = np.delete(ann2pred_FCL, pred2ann_FCL[i])
                else:
                    out_reference_timings.append(np.nan)
                out_predicted_timings.append(ix_pred_FCL[i])
                out_event_types.append("FCL")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
        
            # Right Initial Contacts
            for i in range(len(ix_true_ICR)-1, -1, -1):
                if ann2pred_ICR[i] >- 999:
                    out_predicted_timings.append(ix_pred_ICR[ann2pred_ICR[i]])
                    ix_pred_ICR = np.delete(ix_pred_ICR, ann2pred_ICR[i])
                    pred2ann_ICR = np.delete(pred2ann_ICR, ann2pred_ICR[i])
                else:
                    out_predicted_timings.append(np.nan)
                out_reference_timings.append(ix_true_ICR[i])
                out_event_types.append("ICR")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
            for i in range(len(ix_pred_ICR)-1, -1, -1):
                if pred2ann_ICR[i] > -999:
                    out_reference_timings.append(ix_true_ICR[pred2ann_ICR[i]])
                    ix_true_ICR = np.delete(ix_true_ICR, pred2ann_ICR[i])
                    ann2pred_ICR = np.delete(ann2pred_ICR, pred2ann_ICR[i])
                else:
                    out_reference_timings.append(np.nan)
                out_predicted_timings.append(ix_pred_ICR[i])
                out_event_types.append("ICR")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
                
            # Right Final Contacts
            for i in range(len(ix_true_FCR)-1, -1, -1):
                if ann2pred_FCR[i] >- 999:
                    out_predicted_timings.append(ix_pred_FCR[ann2pred_FCR[i]])
                    ix_pred_FCR = np.delete(ix_pred_FCR, ann2pred_FCR[i])
                    pred2ann_FCR = np.delete(pred2ann_FCR, ann2pred_FCR[i])
                else:
                    out_predicted_timings.append(np.nan)
                out_reference_timings.append(ix_true_FCR[i])
                out_event_types.append("FCR")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
            for i in range(len(ix_pred_FCR)-1, -1, -1):
                if pred2ann_FCR[i] > -999:
                    out_reference_timings.append(ix_true_FCR[pred2ann_FCR[i]])
                    ix_true_FCR = np.delete(ix_true_FCR, pred2ann_FCR[i])
                    ann2pred_FCR = np.delete(ann2pred_FCR, pred2ann_FCR[i])
                else:
                    out_reference_timings.append(np.nan)
                out_predicted_timings.append(ix_pred_FCR[i])
                out_event_types.append("FCR")
                out_filenames.append(test_filename)
                out_sub_ids.append(sub_id)
    df_out = pd.DataFrame({"sub": out_sub_ids, "filename": out_filenames, "event_type": out_event_types, "ref": out_reference_timings, "pred": out_predicted_timings})
    df_out.to_csv(os.path.join("/home/robbin/Desktop", datetime.now().strftime("%Y%m%d%H%M%S")+".tsv"), sep="\t")
    return
    


if __name__ == "__main__":
    
    # Global variables
    PATH = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata"
    DEMOGRAPHICS_FILE = "/mnt/neurogeriatrics_data/Keep Control/Data/lab dataset/rawdata/participants.tsv" if sys.platform == "linux" else "Z:\\Keep Control\\Data\\lab dataset\\rawdata\\participants.tsv"
    TRACKED_POINTS = ["left_ankle", "right_ankle"]
    CLASSIFICATION_TASK = "events"
    WIN_LEN = 400
    DERIVATIVES_PATH = os.path.join(os.path.split(PATH)[0], "derivatives", "motion", "doe")
    CHECKPOINT_FILEPATH = os.path.join(os.path.split(PATH)[0], "derivatives", "motion", "doe", "models")

    # TODO: should be inferred from data
    INPUT_SHAPE = (None, 6)
    NUM_CLASSES = 2
    
    main()