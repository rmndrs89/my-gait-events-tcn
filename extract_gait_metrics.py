import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os

# Settings
ROOT_DIR = "."
FILENAME = "results_v3.tsv"

# Load data
df = pd.read_csv(os.path.join(ROOT_DIR, FILENAME), header=0, sep="\t")

# 
df_gait_metrics = {"sub_id": [],
                    "task": [],
                    "tracked_point": [],
                    "ix_ref": [],
                    "ix_pred": [],
                    "stride_time_ref": [],
                    "stride_time_pred": [],
                    "stance_time_ref": [],
                    "stance_time_pred": [],
                    "swing_time_ref": [],
                    "swing_time_pred": []}

for sub_id in df["sub_id"].unique():
    # print(f"{sub_id}")
    df_sel = df.loc[df["sub_id"]==sub_id]
    for task in df_sel["task"].unique():
        # print(f"    {task:s}")
        df_sel = df.loc[(df["sub_id"]==sub_id) & (df["task"]==task)]
        for run in df_sel["run"].unique():
            if not isinstance(run, str):
                for tracked_point in df_sel["tracked_point"].unique():
                    # print(f"        {tracked_point}")
                    df_sel = df.loc[(df["sub_id"]==sub_id) & (df["task"]==task) & (df["tracked_point"]==tracked_point)]
                    if not(df_sel[["ix_ref", "ix_pred"]].isna().any().any()):
                        ix_IC_ref = np.sort(df_sel.loc[df_sel["event_type"]=="IC"]["ix_ref"].values[:])
                        ix_IC_pred = np.sort(df_sel.loc[df_sel["event_type"]=="IC"]["ix_pred"].values[:])
                        ix_FC_ref = np.sort(df_sel.loc[df_sel["event_type"]=="FC"]["ix_ref"].values[:])
                        ix_FC_pred = np.sort(df_sel.loc[df_sel["event_type"]=="FC"]["ix_pred"].values[:])
                        if len(ix_IC_ref)==0 or len(ix_IC_pred)==0 or len(ix_FC_ref)==0 or len(ix_FC_pred)==0:
                            continue
                        for i in range(len(ix_IC_ref)-1):
                            stride_time_ref = ( ix_IC_ref[i+1] - ix_IC_ref[i] ) / 200
                            stride_time_pred = ( ix_IC_pred[i+1] - ix_IC_pred[i] ) / 200
                            f = np.argwhere(ix_FC_ref > ix_IC_ref[i])[:,0][0]
                            stance_time_ref = ( ix_FC_ref[f] - ix_IC_ref[i] ) / 200
                            stance_time_pred = ( ix_FC_pred[f] - ix_IC_pred[i] ) / 200
                            swing_time_ref = ( ix_IC_ref[i+1] - ix_FC_ref[f] ) / 200
                            swing_time_pred = ( ix_IC_pred[i+1] - ix_FC_pred[f] ) / 200
                            
                            # Add to dict
                            df_gait_metrics["sub_id"].append(sub_id)
                            df_gait_metrics["task"].append(task)
                            df_gait_metrics["tracked_point"].append(tracked_point)
                            df_gait_metrics["ix_ref"].append(ix_IC_ref[i])
                            df_gait_metrics["ix_pred"].append(ix_IC_pred[i])
                            df_gait_metrics["stride_time_ref"].append(stride_time_ref)
                            df_gait_metrics["stride_time_pred"].append(stride_time_pred)
                            df_gait_metrics["stance_time_ref"].append(stance_time_ref)
                            df_gait_metrics["stance_time_pred"].append(stance_time_pred)
                            df_gait_metrics["swing_time_ref"].append(swing_time_ref)
                            df_gait_metrics["swing_time_pred"].append(swing_time_pred)
            else:
                df_sel = df.loc[(df["sub_id"]==sub_id) & (df["task"]==task) & (df["run"]==run)]
                for tracked_point in df_sel["tracked_point"].unique():
                    # print(f"        {tracked_point}")
                    df_sel = df.loc[(df["sub_id"]==sub_id) & (df["task"]==task) & (df["run"]==run) & (df["tracked_point"]==tracked_point)]
                    if not(df_sel[["ix_ref", "ix_pred"]].isna().any().any()):
                        ix_IC_ref = np.sort(df_sel.loc[df_sel["event_type"]=="IC"]["ix_ref"].values[:])
                        ix_IC_pred = np.sort(df_sel.loc[df_sel["event_type"]=="IC"]["ix_pred"].values[:])
                        ix_FC_ref = np.sort(df_sel.loc[df_sel["event_type"]=="FC"]["ix_ref"].values[:])
                        ix_FC_pred = np.sort(df_sel.loc[df_sel["event_type"]=="FC"]["ix_pred"].values[:])
                        if len(ix_IC_ref)==0 or len(ix_IC_pred)==0 or len(ix_FC_ref)==0 or len(ix_FC_pred)==0:
                            continue
                        for i in range(len(ix_IC_ref)-1):
                            stride_time_ref = ( ix_IC_ref[i+1] - ix_IC_ref[i] ) / 200
                            stride_time_pred = ( ix_IC_pred[i+1] - ix_IC_pred[i] ) / 200
                            f = np.argwhere(ix_FC_ref > ix_IC_ref[i])[:,0][0]
                            stance_time_ref = ( ix_FC_ref[f] - ix_IC_ref[i] ) / 200
                            stance_time_pred = ( ix_FC_pred[f] - ix_IC_pred[i] ) / 200
                            swing_time_ref = ( ix_IC_ref[i+1] - ix_FC_ref[f] ) / 200
                            swing_time_pred = ( ix_IC_pred[i+1] - ix_FC_pred[f] ) / 200
                            
                            # Add to dict
                            df_gait_metrics["sub_id"].append(sub_id)
                            df_gait_metrics["task"].append(task)
                            df_gait_metrics["tracked_point"].append(tracked_point)
                            df_gait_metrics["ix_ref"].append(ix_IC_ref[i])
                            df_gait_metrics["ix_pred"].append(ix_IC_pred[i])
                            df_gait_metrics["stride_time_ref"].append(stride_time_ref)
                            df_gait_metrics["stride_time_pred"].append(stride_time_pred)
                            df_gait_metrics["stance_time_ref"].append(stance_time_ref)
                            df_gait_metrics["stance_time_pred"].append(stance_time_pred)
                            df_gait_metrics["swing_time_ref"].append(swing_time_ref)
                            df_gait_metrics["swing_time_pred"].append(swing_time_pred) 