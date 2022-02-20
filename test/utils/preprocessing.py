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

def create_sequences(values, win_len, step_len):
    """Creates sequences of equal length for batch input to the Keras model.
    
    Parameters
    ----------
    values : (N, D) array_like
        The input data with N time steps across D channels.
    win_len : int
        The window length, or length of the sequence.
    step_len : int
        The step length, or number of samples that the windows slides forward.
        
    Returns
    -------
    output : (batch_size, win_len, num_channels) array_like
        The output data with batches of data, each with shape (win_len, num_channels).
    """
    # Initialize output
    output = []
    
    # Loop over data
    for i in range(0, values.shape[0]-win_len+1, step_len):
        output.append(values[i:(i+win_len),:])
    return np.stack(output)