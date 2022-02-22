import numpy as np

def compare_events(annotated, predicted, thr=50):
    """
    Compares the timings of the annotated (true) events and the predicted events.
    Parameters
    ----------
    annotated : array
        A numpy array with the indexes of annotated gait events.
    predicted : array
        A numpy array with the indexes of predicted gait events.
    thr : int
        Threshold (in samples) that determines which time difference is allowed,
        between the annotated and predicted event for it to be considered a match.
    """

    if len(annotated)==0 and len(predicted)==0:
        print("No gait events annotated, no gait events detected!")
        return np.array([]), np.array([]), np.array([])
    if len(annotated)!=0 and len(predicted)==0:
        print(f"{len(annotated)} gait events annotated, but none were detected")
        return np.array([-999 for _ in range(len(annotated))]), np.array([]), np.array([-999 for _ in range(len(annotated))])
    if len(annotated)==0 and len(predicted)!=0:
        print(f"No gait events annotated, but {len(predicted)} events were detected.")
        return np.array([]), np.array([-999 for _ in range(len(predicted))]), np.array([-999 for _ in range(len(predicted))])
    
    # Map every item in the list of annotated events to an item in the list of predicted events ...
    a2b = np.empty_like(annotated)
    for i in range(len(annotated)):
        imin = np.argmin(np.abs(predicted - annotated[i]))
        a2b[i] = imin

    # ... and vice versa
    b2a = np.empty_like(predicted)
    for i in range(len(predicted)):
        imin = np.argmin(np.abs(annotated - predicted[i]))
        b2a[i] = imin
    
    # If multiple items from the list of annotated events point to the same item
    # in the list of predicted events ...
    a2b_unique = np.unique(a2b)
    for i in range(len(a2b_unique)):
        indices = np.argwhere(a2b == a2b_unique[i])[:,0]
        if len(indices) > 1:
            # ... determine which one is closest to the predicted item, and
            # set the other to -999
            a2b[np.setdiff1d(indices, b2a[a2b_unique[i]])] = -999
    
    b2a_unique = np.unique(b2a)
    for i in range(len(b2a_unique)):
        indices = np.argwhere(b2a == b2a_unique[i])[:,0]
        if len(indices) > 1:
            b2a[np.setdiff1d(indices, a2b[b2a_unique[i]])] = -999
    
    # All valid pointers in the annotated list should have a pointer in the 
    # predicted list
    indices_a2b = np.argwhere(a2b > -999)[:,0]
    a2b[indices_a2b[np.argwhere(b2a[a2b[indices_a2b]] == -999)[:,0]]] = -999

    # ... and vice versa
    indices_b2a = np.argwhere(b2a > -999)[:,0]
    b2a[indices_b2a[np.argwhere(a2b[b2a[indices_b2a]] == -999)[:,0]]] = -999

    # Initial estimate of the time difference
    time_diff = predicted[b2a > -999] - annotated[a2b > -999]

    # Create local copies
    indices_a2b = a2b[a2b > -999]
    indices_b2a = b2a[b2a > -999]
    for ti in range(len(time_diff)-1, -1, -1):
        if time_diff[ti] > thr:
            a2b[indices_b2a[ti]] = -999
            b2a[indices_a2b[ti]] = -999

    # Final estimate of the time difference
    time_diff = predicted[b2a > -999] - annotated[a2b > -999] 
    return a2b, b2a, time_diff