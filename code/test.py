import argparse
import os
import pandas as pd
import numpy as np
from utils import load_data
import matplotlib.pyplot as plt

def main(base_dir="", participants_file="", seed=None):
    if seed is None:
        np.random.seed(123)
    else:
        np.random.seed(seed)
        
    # Get list of subject ids for which we have data
    sub_ids = [sub_id for sub_id in os.listdir(base_dir) if sub_id.startswith("sub-pp")]
    
    # Get demographics data from all participants
    df_participants = pd.read_csv(participants_file, sep="\t", header=0)
        
    # Select only demographics from those participants for whom we have data
    df_select = df_participants[df_participants["sub"].isin([sub_id[-5:] for sub_id in sub_ids])]
    
    # Load datasets
    features, labels = load_data(base_dir, df_select)
    fig, axs = plt.subplots(2, 1, sharex=True)
    axs[0].plot(features['L']["left_ankle_ANGVEL_z"], c=(0, 0, 1), lw=1)
    axs[0].fill_between(np.arange(len(labels['L'][:,0])), labels['L'][:,0]*features['L']["left_ankle_ANGVEL_z"].min().min(), color=(0, 0, 1), alpha=0.1)
    plt.show()
    return

if __name__ == "__main__":
    
    # Get command line input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="The base data directory where the files are stored.")
    parser.add_argument("-pp", "--participants_file", help="The participants demographics info file.")
    args = parser.parse_args()
    
    # Call main function
    df_participants = main(base_dir=args.dir, participants_file=args.participants_file)