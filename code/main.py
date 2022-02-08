import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
from utils import resamp1d, plot_omc_vs_imu

def main():
    # Set base directory
    base_dir = "/home/robbin/Projects/annotate_gait_events/data/rawdata" if (sys.platform=="linux") else "E:\\Data"

    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(base_dir) if sub_id.startswith("sub-pp")]

    # Loop over the subjects ids
    for (i_sub_id, sub_id) in enumerate(sub_ids[:5]):
        print(f"{i_sub_id:4d} : {sub_id:s}")

        # Get a list of OMC filenames (stereophotogrammetry)
        omc_filenames = [filename for filename in os.listdir(os.path.join(base_dir, sub_id, "motion")) if filename.endswith("_tracksys-omc_motion.tsv")]

        # Loop over the OMC filenames
        for (i_omc_filename, omc_filename) in enumerate(omc_filenames):
            print(f"{i_omc_filename:8d} : {omc_filename:s}")

            # Set the events filename (e.g. initial and final contacts)
            events_filename = omc_filename.replace("_tracksys-omc_motion.tsv", "_events.tsv")

            # Set the IMU filename
            imu_filename = omc_filename.replace("_tracksys-omc", "_tracksys-imu")

            # Load the OMC and IMU channels information
            df_omc_channels = pd.read_csv(os.path.join(base_dir, sub_id, "motion", omc_filename.replace("_motion.tsv", "_channels.tsv")), sep="\t", header=0)
            df_imu_channels = pd.read_csv(os.path.join(base_dir, sub_id, "motion", imu_filename.replace("_motion.tsv", "_channels.tsv")), sep="\t", header=0)

            # Read out sampling frequency
            fs_omc = df_omc_channels["sampling_frequency"].iloc[0]
            fs_imu = df_imu_channels["sampling_frequency"].iloc[0]

            # Load the OMC and IMU data
            df_omc = pd.read_csv(os.path.join(base_dir, sub_id, "motion", omc_filename), sep="\t", header=0)
            df_imu = pd.read_csv(os.path.join(base_dir, sub_id, "motion", imu_filename), sep="\t", header=0)
            
            # Load the annotated events
            df_events = pd.read_csv(os.path.join(base_dir, sub_id, "motion", events_filename), sep="\t", header=0)

            # If sampling frequency of IMU does not match sampling frequency of OMC
            if fs_imu != fs_omc:
                # Get numeric data from IMU pandas DataFrame
                X = df_imu.to_numpy()

                # Resample data
                Y = resamp1d(X, fs_imu, fs_omc)

                # Overwrite IMU pandas DataFrame
                df_imu = pd.DataFrame(data=Y, columns=df_imu.columns)
                if len(df_imu) > len(df_omc):
                    df_imu = df_imu.iloc[:len(df_omc)]
                elif len(df_omc) > len(df_imu):
                    df_omc = df_omc.iloc[:len(df_imu)]
                del X, Y

            # Plot before
            # plot_omc_vs_imu(df_omc, df_omc_channels, df_imu, df_imu_channels, df_events, fname=omc_filename.replace("_tracksys-omc_motion.tsv", ".pdf"))
            plot_omc_vs_imu(df_omc, df_omc_channels, df_imu, df_imu_channels, df_events, fname=omc_filename.replace("_tracksys-omc_motion.tsv", ".png"))
    return

if __name__ == "__main__":
    main()