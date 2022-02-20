import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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