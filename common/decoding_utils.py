"""
Utils for decoding pixel-level signals to extract signatures

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import cv2
import numpy as np
from scipy.signal import butter, filtfilt

import common.config as config

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Butterworth bandpass filter design

    Args:
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Order of the filter
    
    Returns:
        b, a (ndarray, ndarray): Numerator (b) and denominator (a) polynomials of the IIR filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to the data

    Args:
        data (ndarray): Input signal
        lowcut (float): Low cutoff frequency in Hz
        highcut (float): High cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Order of the filter
    
    Returns:
        y (ndarray): Filtered signal
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data) # Use filtfilt instead of lfilter to avoid any shifting of signal: https://dsp.stackexchange.com/questions/19084/applying-filter-in-scipy-signal-use-lfilter-or-filtfilt
    return y


def loadVideo(video_path, colorspace = 'bgr', downsamples = 0, crop_coords = None, Hom = None):
    """
    Given a video path, load it as a list of np arrays of shape (H, W, 3)
    From following Eulerian Video Magnification implementation found online: 
    https://github.com/hbenbel/Eulerian-Video-Magnification/, with several modifications

    Args:
        video_path (str): path to video file
        colorspace (str): colorspace to convert to. Options are 'bgr', 'yuv', 'ycrcb'
        downsamples (int): number of times to downsample the video frames by factor of 2
        crop_coords (list): coordinates to crop the video to. Format is x1, y1, x2, y2
        Hom (ndarray): 3x3 homography matrix to apply to each frame. If None, no homography is applied.

    Returns:
        image_sequence (list of ndarray): list of np arrays of shape (H, W, 3) containing the video frames
        fps (float): frames per second of the video
    """
    if crop_coords is not None and downsamples > 0:
        print("CAREFUL, you are both cropping and downsampling. Do you really want to do this?")

    image_sequence = []
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)

    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
       
        if crop_coords is not None: 
            x1, y1, x2, y2 = crop_coords
            frame = frame[y1:y2, x1:x2, :]
        
        if downsamples > 0:
            for i in range(downsamples):
                frame = cv2.pyrDown(frame)
        
        if colorspace == 'yuv':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        elif colorspace == 'ycrcb':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

        if Hom is not None:
            frame = cv2.warpPerspective(frame, Hom, (640, 360))

        image_sequence.append(frame[:, :, :])

    video.release()

    return np.asarray(image_sequence), fps


def get_homography(sorted_contour_centers, reference_centers, heatmap, reference_img, display = True):
    """
    Use provided contour centers and reference centers as corresponences compute homography matrix
    between video view and canonical bitmap view

    Args:
        sorted_contour_centers (list): list of (x, y) tuples of centers of localization cell contours, in order from top-left to bottom-right
        reference_centers (list): list of (x, y) tuples of reference centers of localization cells in canonical bitmap view, 
                                also in order from top-left to bottom-right
        heatmatmap (ndarray): heatmap image to visualize homography on
        reference_img (ndarray): reference image to visualize homography on
        display (bool): whether to display the homography result on the heatmap and reference image
    
    Returns:
        H (ndarray): 3x3 homography matrix
    """
    sorted_contour_centers = np.array(sorted_contour_centers)
    reference_centers = np.array(reference_centers)
    H, status = cv2.findHomography(sorted_contour_centers, reference_centers)

    # Warp source image to destination based on homography to visualize success
    if display:
        if len(heatmap.shape) < 3:
            vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        else:
            vis_heatmap = heatmap
        img_out = cv2.warpPerspective(vis_heatmap, H, (config.slm_W, config.slm_H))
        cv2.imshow("Src image", vis_heatmap)
        cv2.imshow("Dst image", reference_img)
        cv2.imshow("Result", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # weird workadound to make destroyAllWindows work
        # https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
        for i in range (1,5):
            cv2.waitKey(1)     
    return H


def get_normalized_mag_at_freq(signal, fps):
    """
    Get power of signal at localization frequency and normalize it by the mean power in the
    frequency bands specified as lower range, upper range, etc. in config

    Args:
        signal (ndarray): 1D array of pixel signal over time
        fps (float): frames per second of the video
    
    Returns:
        normalized_mag (float): normalized magnitude at localization frequency
    """
    signal = signal - np.mean(signal)
    ps =  np.abs(np.fft.fft(signal))**2
    freqs = np.fft.fftfreq(len(signal), 1/fps)

    # get target freq power
    lower_freq_index = (np.abs(freqs - config.localization_frequency)).argmin()
    upper_freq_index = None
    for i, f in enumerate(freqs):
        if i < lower_freq_index and np.abs(config.localization_frequency - f) < config.target_lower_epsilon:
            lower_freq_index = i
        if i > lower_freq_index and np.abs(config.localization_frequency - f) < config.target_upper_epsilon:
            upper_freq_index = i + 1 #add 1 bc slicing would cut here
    if upper_freq_index is None:
        upper_freq_index = lower_freq_index + 1
    target_freq_power = np.sum(ps[lower_freq_index:upper_freq_index])
    
    # get "noise" power
    lower_range_start_index = (np.abs(freqs - config.lower_range_start)).argmin()
    lower_range_end_index = (np.abs(freqs - config.lower_range_end)).argmin()
    upper_range_start_index = (np.abs(freqs - config.upper_range_start)).argmin()
    upper_range_end_index = (np.abs(freqs - config.upper_range_end)).argmin()
    normalization_power =  np.mean(ps[[i for i in range(lower_range_start_index, lower_range_end_index)] + [i for i in range(upper_range_start_index, upper_range_end_index)]])
    
    normalized_mag = target_freq_power / normalization_power
    return normalized_mag


def valid_r_c(r, c, max_cells_W,  max_cells_H, localization_N):
    """
    If localization corners being used, determine if cell at index specified by r, c 
    is a valid sync or data cell, or if it would be at a position occupied by a localization marker

    Args:
        r (int): row index of cell
        c (int): column index of cell
        max_cells_W (int): maximum number of cells that can fit in width of SLM bitmaps
        max_cells_H (int): maximum number of cells that can fit in height of SLM bitmaps
        localization_N (int): width/height of localization markers in number of cells + buffer space. 
                            If None, no localization markers are used.

    Returns:
        True if cell is a valid sync or data cell, False if it is in a localization marker region
    """
    if localization_N is None:
        return True
    if (r < localization_N and c < localization_N) or \
    (r < localization_N and c >= max_cells_W - localization_N) or \
    (r >= max_cells_H - localization_N and  c < localization_N) or\
    (r >= max_cells_H - localization_N and c >= max_cells_W - localization_N):
        return False
    else:
        return True


def get_loc_marker_center(loc_marker_num, slm_W, slm_H, N, buffer_space, localization_N):
    """
    Get center coordinates of localization marker cell specified by loc_marker_num

    Args:
        loc_marker_num (int): which localization marker to get center of. 0 is top
                                left, 1 is top-right, 2 is bottom-left, 3 is bottom-right
        slm_W (int): width of SLM bitmaps in pixels
        slm_H (int): height of SLM bitmaps in pixels
        N (int): width/height of each cell in pixels
        buffer_space (int): space between cells in pixels
        localization_N (int): width/height of localization markers in number of cells + buffer space
    
    Returns:
        x, y coordinates of center of localization marker cell
    """
    max_cells_W = int((slm_W ) / (N + buffer_space))
    max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))

    if loc_marker_num == 0:
        cell_top = 0
        cell_left = 0
    elif loc_marker_num == 1:
        cell_top = 0
        cell_left =  (max_cells_W - localization_N) * (N + buffer_space)
    elif loc_marker_num == 2:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) 
        cell_left = 0
    else:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space)
        cell_left = (max_cells_W - localization_N) * (N + buffer_space)

    #loc_cell_dim =  (localization_N * N) + (localization_N - 1) * buffer_space - buffer_space
    loc_cell_dim = localization_N * (N + buffer_space) - buffer_space
    return cell_left + loc_cell_dim//2, cell_top + loc_cell_dim//2


def get_localization_signal_from_imgseq(image_sequence, loc_marker_num, target_channel, slm_W, slm_H, N, buffer_space, localization_N, max_cells_W, max_cells_H, padding = 0, display = False):
    """
    Get signal of localization marker. Also return the cell boundaries because they are useful
    in next step of adaptation 

    Args:
        image_sequence (list of ndarray): list of np arrays of shape (H, W, 3) containing the video frames
        loc_marker_num (int): which localization marker to get signal from. 0 is top-left, 
                                1 is top-right, 2 is bottom-left, 3 is bottom-right
        target_channel (int or str): which channel to use for signal extraction. 
                                        0, 1, 2 for B, G, R channels respectively, or "sum" for sum of all channels
        slm_W (int): width of SLM bitmaps in pixels
        slm_H (int): height of SLM bitmaps in pixels
        N (int): width/height of each cell in pixels
        buffer_space (int): space between cells in pixels
        localization_N (int): width/height of localization markers in number of cells + buffer space
        max_cells_W (int): maximum number of cells that can fit in width of SLM
        max_cells_H (int): maximum number of cells that can fit in height of SLM
        padding (int): number of pixels to pad inside the cell boundaries when extracting signal
        display (bool): whether to display the localization marker cell on each frame as the signal is extracted
    
    Returns:
        signal (list): list of intensity values of the localization marker cell in each frame
    """
    if loc_marker_num == 0:
        cell_top = padding
        cell_left = padding
    elif loc_marker_num == 1:
        cell_top = padding
        cell_left =  (max_cells_W - localization_N) * (N + buffer_space) + padding
    elif loc_marker_num == 2:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) + padding
        cell_left = padding
    else:
        cell_top = (max_cells_H - localization_N) * (N + buffer_space) + padding
        cell_left = (max_cells_W - localization_N) * (N + buffer_space) + padding
 
    # loc_cell_dim =  (localization_N * N) + (localization_N - 1) * buffer_space
    loc_cell_dim = localization_N * (N + buffer_space) - buffer_space
    cell_bottom = cell_top + loc_cell_dim - padding
    cell_right = cell_left + loc_cell_dim - padding

    mask = np.zeros((slm_H, slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   
    
    signal = []
    for i in range(image_sequence.shape[0]):
        img = image_sequence[i, :,:, :]
        if target_channel == "sum":
            mean_cell_brightness = cv2.mean(img, mask=mask)[0]
            mean_cell_brightness += cv2.mean(img, mask=mask)[1]
            mean_cell_brightness += cv2.mean(img, mask=mask)[2]
        else:
            mean_cell_brightness = cv2.mean(img, mask=mask)[target_channel]
        
        if display:
            cv2.rectangle(img, (cell_left, cell_top), (cell_right, cell_bottom), 255, 1)   
            cv2.imshow("localization marker", img)
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

        signal.append(mean_cell_brightness)
        
    return signal, [cell_top, cell_left, cell_bottom, cell_right]


def get_nonloc_cell_signal_from_imgseq(image_sequence, cell_row, cell_col, display = False):
    """
    Get signal of non-localization marker cell specified by cell_row, cell_col

    Args:
        image_sequence (list of ndarray): list of np arrays of shape (H, W, 3) containing the video frames
        cell_row (int): row index of cell to get signal from
        cell_col (int): column index of cell to get signal from
        display (bool): whether to display the cell on each frame as the signal is extracted
    
    Returns:
        signal (list): list of intensity values of the cell in each frame
    """
    cell_top = cell_row * (config.N + config.buffer_space) 
    cell_left = cell_col * (config.N + config.buffer_space) 
    cell_bottom = cell_top + config.N
    cell_right = cell_left + config.N

    mask = np.zeros((config.slm_H, config.slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   

    signal = []
    for i in range(image_sequence.shape[0]):
        img = image_sequence[i, :,:, :]
        if display:
            cv2.rectangle(img, (cell_left, cell_top), (cell_right, cell_bottom), 255, 1)
            cv2.imshow(f"Cell {i}", img)
            cv2.waitKey(0)
        if config.target_channel == "sum":
            mean_cell_brightness = cv2.mean(img, mask = mask)[0]
            mean_cell_brightness += cv2.mean(img, mask = mask)[1]
            mean_cell_brightness += cv2.mean(img, mask = mask)[2]
        else:
            mean_cell_brightness = cv2.mean(img, mask=mask)[config.target_channel]
        signal.append(mean_cell_brightness)
        
    return signal


def maxSum(arr, n, k):
    """"
    Return maximum sum of subarray of size k in arr of size n, along with the indices of that subarray
    Modified from code found at https://www.geeksforgeeks.org/find-maximum-minimum-sum-subarray-size-k/

    Args:
        arr (list): list of numbers
        n (int): length of arr
        k (int): size of subarray to consider
    
    Returns:
        res (int): maximum sum of subarray of size k
        indices (list): list of indices of the subarray with maximum sum
    """
    # k must be smaller than n
    if (n < k):
        return -1, [i for i in range(len(arr))]
    
    indices = [i for i in range(k)]
     
    # Compute sum of first
    # window of size k
    res = 0
    for i in range(k):
        res += arr[i]
 
    # Compute sums of remaining windows by
    # removing first element of previous
    # window and adding last element of 
    # current window.
    curr_sum = res
    for i in range(k, n):
        curr_sum += arr[i] - arr[i-k]
        if res < curr_sum:
            indices = [r for r in range(i-k+1, i+1)]
            res = curr_sum
    return res, indices