import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_predictions(path, data, targets, predictions, filename_prefix, tracked_point):
    # Get marker data
    df_omc = pd.read_csv(
        os.path.join(path, filename_prefix[:9], "motion", filename_prefix+"_tracksys-omc_motion.tsv"),
        sep = "\t", header = 0
    )
    
    # Get events
    df_events = pd.read_csv(
        os.path.join(path, filename_prefix[:9], "motion", filename_prefix+"_events.tsv"),
        sep = "\t", header = 0
    )
    
    # Get start and end index of trial
    ix_start = df_events[df_events["event_type"]=="start"]["onset"].values[0]
    ix_end = df_events[df_events["event_type"]=="stop"]["onset"].values[0]
    
    # Initialize and populate figure and axes
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(21, 8), gridspec_kw={"height_ratios": [2, 1, 1, 2]})
    if "left" in tracked_point:
        axs[0].plot(df_omc["l_toe_POS_z"], ls='-', lw=2)
        axs[0].plot(df_omc["l_heel_POS_z"], ls='-', lw=2)
    else:
        axs[0].plot(df_omc["r_toe_POS_z"], ls='-', lw=2)
        axs[0].plot(df_omc["r_heel_POS_z"], ls='-', lw=2)
    axs[0].xaxis.set_minor_locator(plt.MultipleLocator(20))
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[0].grid(which="both", c=(0, 0, 0), alpha=0.1, ls=":")
    axs[0].set_ylabel("vert. pos. / mm")
    
    axs[1].plot(np.arange(ix_start, ix_end), targets["initial_contact"], ls='-', lw=2)
    axs[1].plot(np.arange(ix_start, ix_end), predictions[0][0][:,0], ls='-', lw=2)
    axs[1].set_ylabel('Pr(IC)')
    axs[2].plot(np.arange(ix_start, ix_end), targets["final_contact"], ls='-', lw=2)
    axs[2].plot(np.arange(ix_start, ix_end), predictions[1][0][:,0], ls='-', lw=2)
    axs[2].set_ylabel('Pr(FC)')
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(20))
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[1].grid(which='both', c=(0, 0, 0), alpha=0.1, ls=':')
    axs[2].xaxis.set_minor_locator(plt.MultipleLocator(20))
    axs[2].yaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[2].grid(which='both', c=(0, 0, 0), alpha=0.1, ls=':')
    
    axs[3].plot(np.arange(ix_start, ix_end), data[:,0], '-', c='tab:blue', alpha=0.9, lw=2)
    axs[3].set_ylabel('vert. acc. / g', color='tab:blue')
    axs_ = axs[3].twinx()
    axs_.plot(np.arange(ix_start, ix_end), data[:,5], '-', c='tab:red', alpha=0.9, lw=2)
    axs_.set_ylabel('med.lat. ang. vel. / deg/s', color='tab:red')
    axs[3].set_xlim([ix_start, ix_end])
    axs[3].xaxis.set_minor_locator(plt.MultipleLocator(20))
    axs[3].grid(which='both', c=(0, 0, 0), alpha=0.1, ls=':')
    axs[3].set_xlabel('sample')
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    plt.show()

def plot_targets(df_omc, targets, ix_start=None, ix_end=None):
    # Dictionary to map gait event type to subplot/channel
    map_event_to_subplot = {"initial_contact_left": 0,
                            "final_contact_left": 0,
                            "initial_contact_right": 1,
                            "final_contact_right": 1}
    map_event_to_channel = {"initial_contact_left": "l_heel_POS_z",
                            "final_contact_left": "l_toe_POS_z",
                            "initial_contact_right": "r_heel_POS_z",
                            "final_contact_right": "r_toe_POS_z"}
    
    # Set start and end index if omitted by user
    ix_start = 0 if ix_start is None else ix_start
    ix_end = len(df_omc) if ix_end is None else ix_end
    
    # Define number of subplots    
    num_subplots = len(targets)//2
    
    # Instantiate figure
    fig, axs = plt.subplots(num_subplots, 1, sharex=True, figsize=(18, 6))
    
    # Loop over the event types (each event type is a key in the dictionary)
    for k in targets.keys():
        i_subplot = map_event_to_subplot[k]
        chan = map_event_to_channel[k]
        if "heel" in chan:
            axs[i_subplot].plot(np.arange(len(df_omc)), df_omc[chan].iloc[:], color="tab:blue", lw=2)
        else:
            axs[i_subplot].plot(np.arange(len(df_omc)), df_omc[chan].iloc[:], color="tab:blue", lw=2, alpha=0.4)
        axs[i_subplot].set_ylabel("vertical position / mm", color="tab:blue")
        axs[i_subplot].tick_params(axis="y", labelcolor="tab:blue")
        axs_ = axs[i_subplot].twinx()
        axs_.set_ylabel("probability / --", color="tab:red")
        if "heel" in chan:
            axs_.plot(np.arange(len(df_omc)), targets[k], color="tab:red", lw=1)
        else:
            axs_.plot(np.arange(len(df_omc)), targets[k], color="tab:red", lw=1, alpha=0.6)
        axs_.tick_params(axis="y", labelcolor="tab:red")
    for ax in axs:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(20))
        ax.xaxis.set_major_locator(plt.MultipleLocator(200))
        ax.grid(which="both", color=(0, 0, 0), alpha=0.05)
        ax.fill_between(np.arange(0, ix_start), np.ones_like(np.arange(0, ix_start))*df_omc[list(map_event_to_channel.values())].max().max(), facecolor=(0, 0, 0), alpha=0.12)
        ax.fill_between(np.arange(0, ix_start), np.ones_like(np.arange(0, ix_start))*df_omc[list(map_event_to_channel.values())].min().min(), facecolor=(0, 0, 0), alpha=0.12)
        ax.fill_between(np.arange(ix_end, len(df_omc)), np.ones_like(np.arange(ix_end, len(df_omc)))*df_omc[list(map_event_to_channel.values())].max().max(), facecolor=(0, 0, 0), alpha=0.12)
        ax.fill_between(np.arange(ix_end, len(df_omc)), np.ones_like(np.arange(ix_end, len(df_omc)))*df_omc[list(map_event_to_channel.values())].min().min(), facecolor=(0, 0, 0), alpha=0.12)
    axs[0].set_title("left")
    axs[1].set_title("right")
    axs[1].set_xlim((0, len(df_omc)))
    axs[1].set_xlabel("time / samples")
    plt.show()
    return