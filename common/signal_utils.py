"""
Utility functions for processing MediaPipe signals

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d

import common.config as config

def single_feature_signal_processing(signal, resample_signal = True, scaler = "standard"):
    """
    Process a single MediaPipe Facemesh signal feature 

    Args:
        signal (list or numpy array): The signal to be processed.
        resample_signal (bool): Whether to resample the signal to a fixed length.
        scaler (str): Type of scaler to use ("standard" for StandardScaler, "minmax" for MinMaxScaler).
    
    Returns:
        list: The processed signal.
    """
    try:
        proc_signal = interp_fill_nans(np.array(signal))     
    except Exception as e:
        return [0 for i in range(config.single_dynamic_signal_len)]
    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("Unsupported scaler type. Use 'standard' for StandardScaler or 'minmax' for MinMaxScaler.")
    proc_signal = scaler.fit_transform(proc_signal.reshape(-1, 1)).reshape(-1) # important so that concat signal isn't just dominated by scale differences that inflate pearson score
    proc_signal = rolling_average(proc_signal, n = 2)
    if resample_signal:
        proc_signal = ResampleLinear1D(proc_signal, config.single_dynamic_signal_len) # downsample  
    else:
        proc_signal = proc_signal.tolist()  
    return proc_signal


def rolling_average(a, n = 3):
    """
    Apply a rolling average to a 1D numpy array.
    https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy

    Args:
        a (numpy array): The input array.
        n (int): The window size for the rolling average.
    
    Returns:
        numpy array: The smoothed array.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    # BELOW LINES CHANGED AS OF DEC 2024
    ret[n - 2] = ret[n - 1]
    return ret[n - 2:] / n
    # return ret[n - 1:] / n OG 


def ResampleLinear1D(x, targetLen):
    """
    Resample a 1D numpy array to a target length using linear interpolation.
    https://stackoverflow.com/questions/29085268/resample-a-numpy-array

    Args:
        x (numpy array): The input array.
        targetLen (int): The desired length of the output array.
    
    Returns:
        list: The resampled array.
    """
    factor = len(x) / targetLen
    n = np.ceil(x.size / factor).astype(int)
    f = interp1d(np.linspace(0, 1, x.size), x, 'linear')
    return f(np.linspace(0, 1, n)).tolist()


def interp_fill_nans(signal):
    """
    Interpolate to fill NaN values in a 1D numpy array.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Args:
        signal (numpy array): The input array with potential NaN values.
    
    Returns:
        numpy array: The array with NaNs filled via interpolation.
    """
    ok = ~np.isnan(signal)
    xp = ok.ravel().nonzero()[0]
    fp = signal[~np.isnan(signal)]
    x  = np.isnan(signal).ravel().nonzero()[0]
    signal[np.isnan(signal)] = np.interp(x, xp, fp)
    return signal
