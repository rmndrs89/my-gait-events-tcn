import numpy as np
from scipy.interpolate import interp1d

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

def normalize_data(X, axis=0):
    """Normalize data by subtracting the mean and dividing by the standard deviation.

    Parameters
    ----------
    X : (N, D) numpy array_like
        Original data with N time steps across D channels.
    axis : int, optional
        Axis along which the mean and standard deviation are calculated, by default 0

    Returns
    -------
    _ : (N, D) numpy array_like
        Normalized data.
    """
    # Compute channel-wise mean and standard deviation
    mn = np.mean(X, axis=axis)
    sd = np.std(X, axis=axis)
    if any(sd == 0):
        # Cannot divide by zero
        return 
    else:
        # Subtract the mean, and divide by standard deviation
        return (X - np.tile(np.expand_dims(mn, axis=axis), (X.shape[0], 1))) / np.tile(np.expand_dims(sd, axis=axis), (X.shape[0], 1))

def create_sequences(values, win_len, step_len):
    """Creates sequences of equal length for batch input to the Keras model.
    
    Parameters
    ----------
    values : (N, D) array_like
        The input data with N time steps across D channels.
    win_len : int
        The window length, or length of the sequence.
    step_len : int
        The stride, or number of samples that the windows slides forward.
        
    Returns
    -------
    output : (batch_size, win_len, num_channels) array_like
        The output data with batches of data, each with shape (win_len, num_channels).
        Changed 21 Feb 2022: output is a list of equal-length numpy arrays.
    """
    # Initialize output
    output = []
    
    # Loop over data
    for i in range(0, values.shape[0]-win_len+1, step_len):
        output.append(values[i:(i+win_len),:])
    return output # np.stack(output)

def create_batch_sequences(ds, win_len, step_len):
    if len(ds[0]["targets"].keys()) == 2:
        map_class_to_chan = {"initial_contact": 0, "final_contact": 1}
    else:
        map_class_to_chan = {"initial_contact_left": 0, "final_contact_left": 1, "initial_contact_right": 2, "final_contact_right": 3}
    map_chan_to_class = {val: k for (k, val) in map_class_to_chan.items()}

    # For each example, keep track of the filename (and side) it was derived from
    list_examples = []
    list_filenames, list_filenames_examples = [], []

    # Initialize list to store sequences
    list_data = []

    # Loop over the dataset
    for d in range(len(ds)):

        # Stack data and targets
        X = ds[d]["data"]
        Y = np.zeros((X.shape[0], len(ds[d]["targets"].keys())))
        for i, k in enumerate(ds[d]["targets"].keys()):
            Y[:,map_class_to_chan[k]] = ds[d]["targets"][k].reshape(-1,)
        XY = np.hstack((X, Y))

        # Create sequences of shape (win_len, num_channels+num_classes)
        sequences = create_sequences(XY, win_len=win_len, step_len=step_len)

        # Add to existing lists
        for s in range(len(sequences)):
            list_examples.append(
                {
                    "filename_prefix": ds[d]["filename_prefix"],
                    "ix_start": s * step_len
                }
            )
            if "left_or_right" in ds[d].keys():
                list_examples[-1]["left_or_right"] = ds[d]["left_or_right"]
        # list_filenames += [ds[d]["filename_prefix"] for _ in range(len(sequences))]
        # list_filenames_examples += [_ for _ in range(len(sequences))]
        list_data += sequences
    
    # Stack list of sequences in numpy array
    arr_data = np.stack(list_data)

    # Split data and targets
    data = arr_data[:,:,:-len(ds[0]["targets"].keys())]
    targets = {}
    for i, k in enumerate(ds[0]["targets"].keys()):
        targets[map_chan_to_class[i]] = np.expand_dims(arr_data[:,:,-len(ds[0]["targets"].keys())+i], axis=-1)

    return data, targets, list_examples # list_filenames, list_filenames_examples