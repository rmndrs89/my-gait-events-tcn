import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def resamp1d(X, fs_old, fs_new):
    try:
        # In case of (N, D) array
        N, D = X.shape
    except:
        # In case of (N,) array
        N = len(X)

    t  = np.arange(N)/fs_old                               # original time array
    ti = t[np.logical_not(np.any(np.isnan(X), axis=1))]    # time array without NaN
    Xi = X[np.logical_not(np.any(np.isnan(X), axis=1)),:]  # data array without NaN
    f = interp1d(ti, Xi, kind="linear", axis=0, fill_value="extrapolate")  # fit data
    tq = np.arange(N/fs_old*fs_new)/fs_new  # new time array
    return f(tq)


def plot_omc_vs_imu(df_omc, df_imu, df_events):
    # Plot figure
    fig, axs = plt.subplots(2, 1)

    # Grey out the time before and after the trial
    axs[0].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
    axs[0].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
    axs[0].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
    axs[0].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)

    # 
    axs[0].plot(np.arange(len(df_omc)), df_omc["l_heel_POS_z"], ls="-", c=(0.000, 0.314, 0.937), label="left heel")
    axs[0].plot(df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values, df_omc["l_heel_POS_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.000, 0.314, 0.937), ms=6)
    axs[0].plot(np.arange(len(df_omc)), df_omc["l_toe_POS_z"], ls="-", c=(0.106, 0.631, 0.937), label="left toe")
    axs[0].plot(df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values, df_omc["l_toe_POS_z"].iloc[df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.106, 0.631, 0.937))
    axs[0].plot(np.arange(len(df_omc)), df_omc["r_heel_POS_z"], ls="-", c=(0.980, 0.408, 0.000), label="right heel")
    axs[0].plot(df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values, df_omc["r_heel_POS_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.980, 0.408, 0.000))
    axs[0].plot(np.arange(len(df_omc)), df_omc["r_toe_POS_z"], ls="-", c=(0.941, 0.639, 0.039), label="right toe")
    axs[0].plot(df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values, df_omc["r_toe_POS_z"].iloc[df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.941, 0.639, 0.039))
    axs[0].set_xlim((np.max([1, df_events[(df_events["event_type"]=="start")]["onset"].values-50]), np.min([len(df_omc), df_events[(df_events["event_type"]=="stop")]["onset"].values+50])))
    axs[0].set_xlabel("time / samples")
    axs[0].set_ylim((df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20))
    axs[0].set_ylabel("vertical position / mm")
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[0].grid(visible=True, which="major", color=(0, 0, 0), alpha=0.1, ls=":")
    axs[0].legend()

    axs[1].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
    axs[1].fill_between(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20)*np.ones_like(np.arange(0, df_events[(df_events["event_type"]=="start")]["onset"].values)), color=(0, 0, 0), alpha=0.05)
    axs[1].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
    axs[1].fill_between(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc)), (df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20)*np.ones_like(np.arange(df_events[(df_events["event_type"]=="stop")]["onset"].values, len(df_omc))), color=(0, 0, 0), alpha=0.05)
    axs[1].plot(np.arange(len(df_imu)), df_imu["left_ankle_ANGVEL_z"], ls="-", c=(0.000, 0.314, 0.937), label="left ankle")
    axs[1].plot(df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values, df_imu["left_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_left")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.000, 0.314, 0.937))
    axs[1].plot(df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values, df_imu["left_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="final_contact_left")]["onset"].values], ls="none", marker="x", mfc="none", mec=(0.106, 0.631, 0.937))
    axs[1].plot(np.arange(len(df_imu)), df_imu["right_ankle_ANGVEL_z"], ls="-", c=(0.980, 0.408, 0.000), label="right ankle")
    axs[1].plot(df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values, df_imu["right_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="initial_contact_right")]["onset"].values], ls="none", marker="o", mfc="none", mec=(0.980, 0.408, 0.000))
    axs[1].plot(df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values, df_imu["right_ankle_ANGVEL_z"].iloc[df_events[(df_events["event_type"]=="final_contact_right")]["onset"].values], ls="none", marker="x", mfc="none", mec=(0.941, 0.639, 0.039))
    axs[1].set_xlim((np.max([1, df_events[(df_events["event_type"]=="start")]["onset"].values-50]), np.min([len(df_omc), df_events[(df_events["event_type"]=="stop")]["onset"].values+50])))
    axs[1].set_xlabel("time / samples")
    axs[1].set_ylim((df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+150))
    axs[1].set_ylabel("angular velocity position / deg/s")
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[1].grid(visible=True, which="major", color=(0, 0, 0), alpha=0.1, ls=":")
    axs[1].legend()

    # Plot a vertical line for each event
    for _, event in df_events[(df_events["event_type"]=="initial_contact_left")].iterrows():
        if event["onset"] > df_events[(df_events["event_type"]=="start")]["onset"].values:
            axs[0].plot([event["onset"], event["onset"]], [df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20], ls="--", lw=0.5, c=(0.000, 0.314, 0.937), alpha=0.8)
            axs[1].plot([event["onset"], event["onset"]], [df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20], ls="--", lw=0.5, c=(0.000, 0.314, 0.937), alpha=0.8)
            axs[1].text(event["onset"]-5, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+50, r"$\mathrm{IC_L}$", c=(0.000, 0.314, 0.937))

    for _, event in df_events[(df_events["event_type"]=="final_contact_left")].iterrows():
        if event["onset"] > df_events[(df_events["event_type"]=="start")]["onset"].values:
            axs[0].plot([event["onset"], event["onset"]], [df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20], ls="--", lw=0.5, c=(0.106, 0.631, 0.937), alpha=0.8)
            axs[1].plot([event["onset"], event["onset"]], [df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20], ls="--", lw=0.5, c=(0.106, 0.631, 0.937), alpha=0.8)
            axs[1].text(event["onset"]-5, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+50, r"$\mathrm{FC_L}$", c=(0.106, 0.631, 0.937))

    for _, event in df_events[(df_events["event_type"]=="initial_contact_right")].iterrows():
        if event["onset"] > df_events[(df_events["event_type"]=="start")]["onset"].values:
            axs[0].plot([event["onset"], event["onset"]], [df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20], ls="--", lw=0.5, c=(0.980, 0.408, 0.000), alpha=0.8)
            axs[1].plot([event["onset"], event["onset"]], [df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20], ls="--", lw=0.5, c=(0.980, 0.408, 0.000), alpha=0.8)
            axs[1].text(event["onset"]-5, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+50, r"$\mathrm{IC_R}$", c=(0.980, 0.408, 0.000))

    for _, event in df_events[(df_events["event_type"]=="final_contact_right")].iterrows():
        if event["onset"] > df_events[(df_events["event_type"]=="start")]["onset"].values:
            axs[0].plot([event["onset"], event["onset"]], [df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].min().min()-20, df_omc[["l_heel_POS_z", "l_toe_POS_z", "r_heel_POS_z", "r_toe_POS_z"]].max().max()+20], ls="--", lw=0.5, c=(0.941, 0.639, 0.039), alpha=0.8)
            axs[1].plot([event["onset"], event["onset"]], [df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].min().min()-20, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+20], ls="--", lw=0.5, c=(0.941, 0.639, 0.039), alpha=0.8)
            axs[1].text(event["onset"]-5, df_imu[["left_ankle_ANGVEL_z", "right_ankle_ANGVEL_z"]].max().max()+50, r"$\mathrm{FC_R}$", c=(0.941, 0.639, 0.039))
    plt.show()
    return