import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

def resamp1d(X, fs_old, fs_new):
    """Resample data to new sampling frequency.

    Parameters
    ----------
    X : (N, D) array-like
        Original data with N time steps across D channels.
    fs_old : int, float
        Original sampling frequency (in Hz).
    fs_new : int, float
        The new sampling frequency (in Hz).

    Returns
    -------
    _ : (N', D) array-like
        Interpolated data with N' time steps across D channels.
    """
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

def get_features(df_data, df_channels, tracked_points=[], incl_magn=False):
    """Gets the features (i.e. the numeric data) for the IMU sensor data DataFrame.

    Parameters
    ----------
    df_data : pandas DataFrame
        IMU sensor data.
    df_channels : pandas DataFrame
        IMU channels information.
    tracked_points : list, optional
        List of tracked points that we want to use for the analysis, by default [].
    incl_magn : bool, optional
        Whether to include the magnetometer readings, by default False.

    Returns
    -------
    _ : pandas DataFrame
        IMU sensor data from selected tracked points, with or without magnetometer readings.
    """
    # Select only features from tracked points
    col_tracked_pts = df_channels.loc[df_channels['tracked_point'].isin(tracked_points)]['name'].values
    df_init = df_data[col_tracked_pts]

    # Include magnetometer data or not
    if incl_magn is True:
        return df_init
    else:
        col_names = [col_name for col_name in df_init.columns if "MAGN" not in col_name]
        return df_init[col_names]

def get_annotated_events(df_events, event_type=""):
    """Gets the indices of specific gait events from the events DataFrame.

    Parameters
    ----------
    df_events : pandas DataFrame
        Gait events and the start and stop of the measurement.
    event_type : str, optional
        Type of gait events, e.g. `initial_contact_left`, by default ""

    Returns
    -------
    indx : array-like
        Indices corresponding to gait events of the given type.
    """
    indx = df_events[(df_events['event_type']==event_type)]['onset'].values - 1
    return indx

def plot_omc_vs_imu(df_omc, df_omc_channels, df_imu, df_imu_channels, df_events, fname=""):
    # Get sampling frequency
    fs_omc = df_omc_channels["sampling_frequency"].iloc[0]
    fs_imu = df_imu_channels["sampling_frequency"].iloc[0]
        
    # Index of start and stop
    indx_start = df_events[(df_events["event_type"]=="start")]["onset"].values[0]-1
    indx_stop  = df_events[(df_events["event_type"]=="stop")]["onset"].values[0]

    # Get features and indices of gait events    
    features = get_features(df_imu, df_imu_channels, tracked_points=['left_ankle', 'right_ankle'], incl_magn=False)
    indx_events = {'ICL': get_annotated_events(df_events, event_type='initial_contact_left'),
                   'FCL': get_annotated_events(df_events, event_type='final_contact_left'),
                   'ICR': get_annotated_events(df_events, event_type='initial_contact_right'),
                   'FCR': get_annotated_events(df_events, event_type='final_contact_right')}
    
    targets_vect = {'L': np.zeros((len(df_imu),)),
                'R': np.zeros((len(df_imu),))}

    if indx_events['FCL'][0] < indx_events['ICL'][0]:
        arr = indx_events['ICL'] - indx_events['FCL'][:len(indx_events['ICL'])]
        for i in range(len(arr)):
            targets_vect['L'][indx_events['FCL'][i]:indx_events['ICL'][i]] = 1.0
    else:
        arr = indx_events['ICL'][1:] - indx_events['FCL'][1:len(indx_events['ICL'][1:])]
        for i in range(len(arr)):
            targets_vect['L'][indx_events['FCL'][i]:indx_events['ICL'][1+i]] = 1.0
        targets_vect['L'][:indx_events['ICL'][0]] = 1.0

    if indx_events['FCR'][0] < indx_events['ICR'][0]:
        arr = indx_events['ICR'] - indx_events['FCR'][:len(indx_events['ICR'])]
        for i in range(len(arr)):
            targets_vect['R'][indx_events['FCR'][i]:indx_events['ICR'][i]] = 1.0
    else:
        arr = indx_events['ICR'][1:] - indx_events['FCR'][1:len(indx_events['ICR'][1:])]
        for i in range(len(arr)):
            targets_vect['R'][indx_events['FCR'][i]:indx_events['ICR'][1+i]] = 1.0
        targets_vect['R'][:indx_events['ICR'][0]] = 1.0
    
    # Instantiate figure object
    cm = 1/2.54
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(29.7*cm, 21*cm))
    
    # Optical motion capture data
    # Left heel
    axs[0].plot(np.arange(0, indx_start)/fs_omc, df_omc['l_heel_POS_z'].iloc[:indx_start], c=(0.000, 0.314, 0.937, 0.2))
    axs[0].plot(np.arange(indx_start, indx_stop)/fs_omc, df_omc['l_heel_POS_z'].iloc[indx_start:indx_stop], c=(0.000, 0.314, 0.937), label='left heel (marker)')
    axs[0].plot(np.arange(indx_stop, len(df_omc))/fs_omc, df_omc['l_heel_POS_z'].iloc[indx_stop:], c=(0.000, 0.314, 0.937, 0.2))
    axs[0].plot(indx_events['ICL']/fs_omc, df_omc['l_heel_POS_z'].iloc[indx_events['ICL']], ls='none', marker='v', mfc='none', mec=(0.000, 0.314, 0.937))

    # Left toe
    axs[0].plot(np.arange(0, indx_start)/fs_omc, df_omc['l_toe_POS_z'].iloc[:indx_start], c=(0.106, 0.631, 0.937, 0.2))
    axs[0].plot(np.arange(indx_start, indx_stop)/fs_omc, df_omc['l_toe_POS_z'].iloc[indx_start:indx_stop], c=(0.106, 0.631, 0.937), label='left toe (marker)')
    axs[0].plot(np.arange(indx_stop, len(df_omc))/fs_omc, df_omc['l_toe_POS_z'].iloc[indx_stop:], c=(0.106, 0.631, 0.937, 0.2))
    axs[0].plot(indx_events['FCL']/fs_omc, df_omc['l_toe_POS_z'].iloc[indx_events['FCL']], ls='none', marker='^', mfc='none', mec=(0.106, 0.631, 0.937))

    # Right heel
    axs[0].plot(np.arange(0, indx_start)/fs_omc, df_omc['r_heel_POS_z'].iloc[:indx_start], c=(0.980, 0.408, 0.000, 0.2))
    axs[0].plot(np.arange(indx_start, indx_stop)/fs_omc, df_omc['r_heel_POS_z'].iloc[indx_start:indx_stop], c=(0.980, 0.408, 0.000), label='right heel (marker)')
    axs[0].plot(np.arange(indx_stop, len(df_omc))/fs_omc, df_omc['r_heel_POS_z'].iloc[indx_stop:], c=(0.980, 0.408, 0.000, 0.2))
    axs[0].plot(indx_events['ICR']/fs_omc, df_omc['r_heel_POS_z'].iloc[indx_events['ICR']], ls='none', marker='v', mfc='none', mec=(0.980, 0.408, 0.000))

    # Right heel
    axs[0].plot(np.arange(0, indx_start)/fs_omc, df_omc['r_toe_POS_z'].iloc[:indx_start], c=(0.941, 0.639, 0.039, 0.2))
    axs[0].plot(np.arange(indx_start, indx_stop)/fs_omc, df_omc['r_toe_POS_z'].iloc[indx_start:indx_stop], c=(0.941, 0.639, 0.039), label='right toe (marker)')
    axs[0].plot(np.arange(indx_stop, len(df_omc))/fs_omc, df_omc['r_toe_POS_z'].iloc[indx_stop:], c=(0.941, 0.639, 0.039, 0.2))
    axs[0].plot(indx_events['FCR']/fs_omc, df_omc['r_toe_POS_z'].iloc[indx_events['FCR']], ls='none', marker='^', mfc='none', mec=(0.941, 0.639, 0.039))
    
    # Styling
    axs[0].spines['right'].set_color('none')
    axs[0].spines['top'].set_color('none')
    axs[0].set_ylim((df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].min().min()-20, df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].max().max()+20))
    axs[0].set_ylabel('vertical position / mm', fontsize=16)
    axs[0].yaxis.set_minor_locator(plt.MultipleLocator(10))
    axs[0].grid(which='both', ls=':', c=(0, 0, 0), alpha=0.1)
    axs[0].tick_params(axis='both', labelsize=16)
    axs[0].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=14)
    
    # Inertial measurement unit
    # Left 
    axs[1].plot(np.arange(0, indx_start)/fs_imu, features['left_ankle_ANGVEL_z'].iloc[:indx_start], c=(0.000, 0.314, 0.937, 0.2))
    axs[1].plot(np.arange(indx_start, indx_stop)/fs_imu, features['left_ankle_ANGVEL_z'].iloc[indx_start:indx_stop], c=(0.000, 0.314, 0.937), label='left ankle (IMU)')
    axs[1].plot(np.arange(indx_stop, len(df_imu))/fs_imu, features['left_ankle_ANGVEL_z'].iloc[indx_stop:], c=(0.000, 0.314, 0.937, 0.2))
    axs[1].plot(indx_events['ICL']/fs_imu, features['left_ankle_ANGVEL_z'].iloc[indx_events['ICL']], ls='none', marker='v', mfc='none', mec=(0.000, 0.314, 0.937))
    axs[1].plot(indx_events['FCL']/fs_imu, features['left_ankle_ANGVEL_z'].iloc[indx_events['FCL']], ls='none', marker='^', mfc='none', mec=(0.106, 0.631, 0.937))

    # Right 
    axs[1].plot(np.arange(0, indx_start)/fs_imu, features['right_ankle_ANGVEL_z'].iloc[:indx_start], c=(0.980, 0.408, 0.000, 0.2))
    axs[1].plot(np.arange(indx_start, indx_stop)/fs_imu, features['right_ankle_ANGVEL_z'].iloc[indx_start:indx_stop], c=(0.980, 0.408, 0.000), label='right ankle (IMU)')
    axs[1].plot(np.arange(indx_stop, len(df_imu))/fs_imu, features['right_ankle_ANGVEL_z'].iloc[indx_stop:], c=(0.980, 0.408, 0.000, 0.2))
    axs[1].plot(indx_events['ICR']/fs_omc, features['right_ankle_ANGVEL_z'].iloc[indx_events['ICR']], ls='none', marker='v', mfc='none', mec=(0.980, 0.408, 0.000))
    axs[1].plot(indx_events['FCR']/fs_omc, features['right_ankle_ANGVEL_z'].iloc[indx_events['FCR']], ls='none', marker='v', mfc='none', mec=(0.941, 0.639, 0.039))

    # Styling
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].set_ylim((features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].min().min()-200, features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].max().max()+200))
    axs[1].set_xlim((indx_start-20, indx_stop+20)/fs_imu)
    axs[1].set_xlabel('time / s', fontsize=16)
    axs[1].set_ylabel('angular velocity / deg/s', fontsize=16)
    axs[1].xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    axs[1].yaxis.set_minor_locator(plt.MultipleLocator(50))
    axs[1].grid(which='both', ls=':', c=(0, 0, 0), alpha=0.1)
    axs[1].tick_params(axis='both', labelsize=16)

    # Plot vertical lines for events
    for i in range(len(indx_events['ICL'])):
        axs[0].plot([indx_events['ICL'][i], indx_events['ICL'][i]]/fs_omc, [df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].min().min()-20, df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].max().max()+20], ls='--', c=(0.000, 0.314, 0.937, 0.5), lw=0.8)
        axs[1].plot([indx_events['ICL'][i], indx_events['ICL'][i]]/fs_imu, [features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].min().min()-200, features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].max().max()+200], ls='--', c=(0.000, 0.314, 0.937, 0.5), lw=0.8)

    for i in range(len(indx_events['FCL'])):
        axs[0].plot([indx_events['FCL'][i], indx_events['FCL'][i]]/fs_omc, [df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].min().min()-20, df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].max().max()+20], ls='--', c=(0.106, 0.631, 0.937, 0.5), lw=0.8)
        axs[1].plot([indx_events['FCL'][i], indx_events['FCL'][i]]/fs_imu, [features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].min().min()-200, features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].max().max()+200], ls='--', c=(0.106, 0.631, 0.937, 0.5), lw=0.8)
        
    for i in range(len(indx_events['ICR'])):
        axs[0].plot([indx_events['ICR'][i], indx_events['ICR'][i]]/fs_omc, [df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].min().min()-20, df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].max().max()+20], ls='--', c=(0.980, 0.408, 0.000, 0.5), lw=0.8)
        axs[1].plot([indx_events['ICR'][i], indx_events['ICR'][i]]/fs_imu, [features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].min().min()-200, features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].max().max()+200], ls='--', c=(0.980, 0.408, 0.000, 0.5), lw=0.8)
        
    for i in range(len(indx_events['FCR'])):
        axs[0].plot([indx_events['FCR'][i], indx_events['FCR'][i]]/fs_omc, [df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].min().min()-20, df_omc[['l_heel_POS_z', 'l_toe_POS_z', 'r_heel_POS_z', 'r_toe_POS_z']].max().max()+20], ls='--', c=(0.941, 0.639, 0.039, 0.5), lw=0.8)
        axs[1].plot([indx_events['FCR'][i], indx_events['FCR'][i]]/fs_imu, [features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].min().min()-200, features[['left_ankle_ANGVEL_z', 'left_ankle_ANGVEL_z']].max().max()+200], ls='--', c=(0.941, 0.639, 0.039, 0.5), lw=0.8)

    # Plot gait cycle swing phase
    axs[1].fill_between(np.arange(len(targets_vect['L']))/fs_imu, targets_vect['L']*features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].max().max(), color=(0.000, 0.314, 0.937), alpha=0.1, label='left swing phase')
    axs[1].fill_between(np.arange(len(targets_vect['L']))/fs_imu, targets_vect['L']*features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].min().min(), color=(0.000, 0.314, 0.937), alpha=0.1)
    axs[1].fill_between(np.arange(len(targets_vect['R']))/fs_imu, targets_vect['R']*features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].max().max(), color=(0.980, 0.408, 0.000), alpha=0.1, label='right swing phase')
    axs[1].fill_between(np.arange(len(targets_vect['R']))/fs_imu, targets_vect['R']*features[['left_ankle_ANGVEL_z', 'right_ankle_ANGVEL_z']].min().min(), color=(0.980, 0.408, 0.000), alpha=0.1)
    axs[1].legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=14)

    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    if len(fname) > 0:
        if fname.endswith('.pdf'):
            plt.savefig(os.path.join("/home/robbin/Desktop/fig", fname), dpi='figure', format='pdf')
        elif fname.endswith('.png'):
            plt.savefig(os.path.join("/home/robbin/Desktop/fig", fname), dpi=300, format='png')
    # plt.show()
    return plt.gcf()