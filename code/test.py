import argparse
import os
import pandas as pd

def main(base_dir="", participants_file=""):
    # Get list of subject ids for which we have data
    sub_ids = [sub_id for sub_id in os.listdir(base_dir) if sub_id.startswith("sub-pp")]
    
    # Get demographics data from all participants
    df_participants = pd.read_csv(participants_file, sep="\t", header=0)
    print(f"DataFrame: {len(df_participants)}")
    
    # Select only demographics from those participants for whom we have data
    df_select = df_participants[df_participants["sub"].isin([sub_id[-5:] for sub_id in sub_ids])]
    print(f"DataFrame: {len(df_select)}")
    return df_select

if __name__ == "__main__":
    
    # Get command line input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir", help="The base data directory where the files are stored.")
    parser.add_argument("-t", "--task", help="The classification task, either `events` or `phases`.", choices=['events', 'phases'])
    parser.add_argument("-pp", "--participants_file", help="The participants demographics info file.")
    args = parser.parse_args()
    
    # Call main function
    df_participants = main(base_dir=args.dir, participants_file=args.participants_file)