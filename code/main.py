import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def main():
    # Set base directory
    base_dir = "/home/robbin/Projects/annotate_gait_events/data/rawdata" # /sub-pp106/motion"

    # Get list of subject ids
    sub_ids = [sub_id for sub_id in os.listdir(base_dir) if sub_id.startswith("sub-pp")]

    # Loop over the subjects ids
    for (i_sub_id, sub_id) in enumerate(sub_ids[:1]):
        print(f"{i_sub_id:4d} : {sub_id:s}")

        # Get a list of OMC filenames (stereophotogrammetry)
        omc_filenames = [filename for filename in os.listdir(os.path.join(base_dir, sub_id, "motion")) if filename.endswith("_tracksys-omc_motion.tsv")]

        # Loop over the OMC filenames
        for (i_omc_filename, omc_filename) in enumerate(omc_filenames[:1]):
            print(f"{i_omc_filename:8d} : {omc_filename:s}")

            # Set the events filename (e.g. initial and final contacts)
            events_filename = omc_filename.replace("_tracksys-omc_motion.tsv", "_events.tsv")

            # Set the IMU filename
            imu_filename = omc_filename.replace("_tracksys-omc", "_tracksys-imu")

            # Load the OMC data
            df_omc = pd.read_csv(os.path.join(base_dir, sub_id, "motion", omc_filename), sep="\t", header=0)
            df_events = pd.read_csv(os.path.join(base_dir, sub_id, "motion", events_filename), sep="\t", header=0)
            df_imu = pd.read_csv(os.path.join(base_dir, sub_id, "motion", imu_filename), sep="\t", header=0)

            # Plot figure
            fig, axs = plt.subplots(2, 1)
            axs[0].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
            axs[0].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
            axs[0].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
            axs[0].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
            axs[0].plot(np.arange(len(df_omc)), df_omc["l_heel_POS_z"], ls="-", c="b")
            axs[0].plot(df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values, df_omc["l_heel_POS_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec="b")
            axs[0].plot(np.arange(len(df_omc)), df_omc["l_toe_POS_z"], ls="-", c=(0.07, 0.62, 1.0))
            axs[0].plot(df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values, df_omc["l_toe_POS_z"].iloc[df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.07, 0.62, 1.0))
            axs[0].plot(np.arange(len(df_omc)), df_omc["r_heel_POS_z"], ls="-", c="r")
            axs[0].plot(df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values, df_omc["r_heel_POS_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec="r")
            axs[0].plot(np.arange(len(df_omc)), df_omc["r_toe_POS_z"], ls="-", c=(0.85, 0.325, 0.098))
            axs[0].plot(df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values, df_omc["r_toe_POS_z"].iloc[df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.85, 0.325, 0.098))
            axs[0].set_xlim((0, len(df_omc)))
            axs[0].set_xlabel("time, samples")
            axs[0].set_ylim((df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20))
            axs[0].set_ylabel("vertical position, mm")
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            axs[0].spines["bottom"].set_position(("data", 0))
            axs[0].spines["left"].set_position(("data", 0))
            axs[0].grid(visible=True, which="major", color=(0, 0, 0), alpha=0.1, ls=":")

            axs[1].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
            axs[1].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
            axs[1].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
            axs[1].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
            axs[1].plot(np.arange(len(df_imu)), df_imu["left_ankle_ANGVEL_z"], ls="-", c="b")
            axs[1].plot(df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values, df_imu["left_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec="b")
            axs[1].plot(df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values, df_imu["left_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values], ls="none", marker="x", mfc="none", mec="b")
            axs[1].plot(np.arange(len(df_imu)), df_imu["right_ankle_ANGVEL_z"], ls="-", c="r")
            axs[1].plot(df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values, df_imu["right_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec="r")
            axs[1].plot(df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values, df_imu["right_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values], ls="none", marker="x", mfc="none", mec="r")
            axs[1].set_xlim((0, len(df_imu)))
            axs[1].set_xlabel("time, samples")
            axs[1].set_ylim((df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20))
            axs[1].set_ylabel("angular velocity position, deg/s")
            axs[1].spines["top"].set_visible(False)
            axs[1].spines["right"].set_visible(False)
            axs[1].spines["bottom"].set_position(("data", 0))
            axs[1].spines["left"].set_position(("data", 0))
            axs[1].grid(visible=True, which="major", color=(0, 0, 0), alpha=0.1, ls=":")
            plt.show()

    return

if __name__ == "__main__":
    main()