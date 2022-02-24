import numpy as np
from scipy.interpolate import interp1d
from sklearn.model_selection import LeavePGroupsOut

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
    # Map class name to channel index, and vice versa    
    map_class_to_chan = {"initial_contact": 0, "final_contact": 1}
    map_chan_to_class = {val: k for (k, val) in map_class_to_chan.items()}

    # Initialize empty list to store output dataset
    ds_out = []
    sequences = []

    # Loop over the dataset
    for ix_instance in range(len(ds)):
        
        # Get the data and targets as a single array
        X = ds[ix_instance]["data"]
        Y = np.zeros((X.shape[0], len(ds[ix_instance]["targets"].keys())))
        for i, k in enumerate(list(ds[ix_instance]["targets"].keys())):
            Y[:,map_class_to_chan[k]] = ds[ix_instance]["targets"][k].reshape(-1,)
        XY = np.hstack((X, Y))
        
        # Create batches of sequences of equal length (bs, win_len, num_channels+num_classes)
        sequences_ = create_sequences(XY, win_len=win_len, step_len=step_len)
        
        # Add to output dataset
        for seq in sequences_:
            ds_out.append({
                "filename_prefix": ds[ix_instance]["filename_prefix"],
                "tracked_point": ds[ix_instance]["tracked_point"],
                "data": seq[:,:-2],
                "targets": {
                    "initial_contact": seq[:,-2],
                    "final_contact": seq[:,-1]
                }
            })
        
        # Accumulate in overall data array
        sequences += sequences_
    
    # Convert to numpy array
    sequences = np.stack(sequences)
    
    # Split data and targets
    data = sequences[:,:,:-2]
    targets = {}
    for i, k in enumerate(list(ds[0]["targets"].keys())):
        targets[map_chan_to_class[i]] = np.expand_dims(sequences[:,:,-2+i], axis=-1)
    return data, targets, ds_out