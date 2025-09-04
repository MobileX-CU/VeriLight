"""
Decode the data embedded into a speech video.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""

from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import matplotlib.pyplot as plt
from colorama import Style, Fore, Back

import common.config as config
from common.decoding_utils import valid_r_c, get_nonloc_cell_signal_from_imgseq, get_normalized_mag_at_freq, maxSum
from embedding.perceptibility_utils import get_on_off_cell_val_from_cell_signal, CIEDE2000

def decode_sequence(image_sequence, fps, get_perceptibilities = False, gts = None, pkl_path_prefix = None, signal_pkls_path = None, display = False):
    """
    Args:
        image_sequence (list of np.array): List of numpy arrays of shape (H, W, 3) representing the video frames, 
                        with the homography already applied (i.e., only contains the projection region)
        fps (int): Frame rate of the video, FPS
        get_perceptibilities (bool): If True, will return perceptibility scores for each cell (only needed for adaptation)
        gts (list of str): List of ground truth bitstrings for each cell, if available. If None, no ground truth comparisons
                            will be performed.
        pkl_path_prefix (str): If provided, will save the signals for each cell to a pkl file with this prefix
        signal_pkls_path (str): If provided, will load the signals for each cell from a pkl file with this prefix
                                rather than extracting from the image sequence
        display (bool): If True, will display plots of the signals and decoding process.    

    Returns:
        tot_hard_pred (str): The concatenated hard predictions for all cells, where '0' and '1' represent the decoded bits
        tot_probs (list of float): The concatenated probabilities for all cells, where each element is a float representing
                                the probability of the bit being '1'. Used for Soft Decision Decoding
        tot_gt (str): The concatenated ground truth bitstrings for all cells, if gts is provided. Otherwise, an empty string.
        sync_mags (list of float): The normalized magnitudes of the sync cells, used for debugging and analysis
        all_perceptibilities (list of float): The perceptibility scores for each cell, if get_perceptibilities is True.
                                              Each element is a float representing the perceptibility score of the cell.
                                              Used for adaptation.
        all_off_cell_bgrs (list of list): The BGR values of the off state for each cell, if get_perceptibilities is True.
                                           Each element is a list of three floats representing the BGR values of the off state of the cell.
                                           Used for adaptation.
    """
    # get sync cell signals 
    sync_mags = []
    sync_signals = []
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N) or f"{r}-{c}" in config.reserved_localization_cells:
                continue
            
            if (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): # this a border cell, and not in reserved localization cells, so must be a sync cell
                
                if not signal_pkls_path:
                    cell_signal = get_nonloc_cell_signal_from_imgseq(image_sequence, r, c)
                else: #load this sync from the pkl at the path instead of extracting from img sequence
                    with open(signal_pkls_path + f"_sync_r{r}_c{c}.pkl", "rb") as pklfile:
                        cell_signal = pickle.load(pklfile)

                norm_power = get_normalized_mag_at_freq(cell_signal, fps)
                sync_mags.append(norm_power)

                if pkl_path_prefix: # for saving for later analysis, if pkl_path_prefix is provided
                    with open(pkl_path_prefix + f"_sync_r{r}_c{c}.pkl", "wb") as pklfile:
                        pickle.dump(cell_signal, pklfile)   

                smoothed_cell_signal, fully_processed_cell_signal,  _ = process_signal(cell_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n) 
                sync_signals.append(cell_signal)
                
                if display:
                    fig, axes = plt.subplots(3)
                    axes[0].set_title("Raw Cell Signal")
                    axes[0].plot(cell_signal)
                    axes[1].set_title("Smoothed Cell Signal")   
                    axes[1].plot(smoothed_cell_signal)
                    axes[2].set_title("Fully Processed Cell Signal")
                    axes[2].plot(fully_processed_cell_signal)
                    plt.suptitle(f"Sync Cell Cell {r}-{c}")
                    plt.show()

    # get blinks based on blinks of averaged sync cells
    sync_signals = np.array(sync_signals)
    avgd_sync_signal = np.mean(np.array(sync_signals), axis = 0)
    smoothed_avgd_sync_signal, fully_processed_avgd_sync_signal, _ = process_signal(avgd_sync_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n)
    sync_blinks = get_overall_sync_blinks(fully_processed_avgd_sync_signal, fps)
    if display:
        fig, axes = plt.subplots(2)
        plt.suptitle("Avgd Sync Signal and Final Sync Signal Blinks")
        axes[0].plot(avgd_sync_signal)
        axes[1].plot(fully_processed_avgd_sync_signal)
        axes[1].vlines(sync_blinks, min(fully_processed_avgd_sync_signal), max(fully_processed_avgd_sync_signal), color = 'r', linestyle = 'dashed')
        plt.show()
    
    # decode non-channel or localization cells
    # optionally, get perceptibility score of each
    tot_probs = [] # probs for all cells
    tot_hard_pred = ""
    tot_gt = ""
    info_cell_num = 0
    all_perceptibilities = [] #perceptibilities for all cells. will be filled if get_perceptibilities = True
    all_off_cell_bgrs = []
    num_bits_per_cell = config.frequency * config.embedding_window_duration 
    for r in range(config.max_cells_H):
        for c in range(config.max_cells_W): 
            if not valid_r_c(r, c, config.max_cells_W,  config.max_cells_H, config.localization_N): #skip any non-border cells covered by a localization corner, if in use
                continue
            
            #this not a border cell or covered by localization corner, so decode it to recover info 
            if not  (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): 
                if not signal_pkls_path:
                    cell_signal = get_nonloc_cell_signal_from_imgseq(image_sequence, r, c)
                else:
                    with open(signal_pkls_path + f"_info_r{r}_c{c}.pkl", "rb") as pklfile:
                        cell_signal = pickle.load(pklfile)

                if gts is not None:
                    gt = gts[info_cell_num]
                    if len(gt) != num_bits_per_cell:
                        print(Fore.YELLOW + Back.RED + f"WARNING: Cell {r}-{c} gt is too short ({len(gt)} bits intead of {num_bits_per_cell}). Terminating decoding." + Style.RESET_ALL)
                        break
                else:
                    gt = None

                if pkl_path_prefix:
                    with open(pkl_path_prefix + f"_info_r{r}_c{c}.pkl", "wb") as pklfile:
                        pickle.dump(cell_signal, pklfile)
                        pickle.dump(gt, pklfile)

                # if no blinks recovered, assume all 0s
                if sync_blinks is not None:
                    if display:
                        cell_hard_pred, cell_probs =  decode_data_cell_signal(sync_blinks, cell_signal, gt = gt, plot_title = f"Cell {r}-{c}")
                    else:
                        cell_hard_pred, cell_probs =  decode_data_cell_signal(sync_blinks, cell_signal)
                else:
                    cell_hard_pred = ''.join(['0' for i in range(int(config.embedding_window_duration * config.frequency))])
                    cell_probs = [0 for i in range(int(config.embedding_window_duration * config.frequency))]

                tot_probs += cell_probs

                if gt is not None:
                    tot_gt += gt
                tot_hard_pred += cell_hard_pred
                info_cell_num += 1
            
            if get_perceptibilities:
                # get perceptibility + BGR for all cells, not just info ones
                cell_top = r * (config.N + config.buffer_space) 
                cell_left = c * (config.N + config.buffer_space) 
                cell_bottom = cell_top + config.N
                cell_right = cell_left + config.N

                off_cell_bgr, on_cell_bgr = get_on_off_cell_val_from_cell_signal(cell_signal, image_sequence, [cell_top, cell_left, cell_bottom, cell_right])
                perceptibility = CIEDE2000(off_cell_bgr[::-1], on_cell_bgr[::-1])
                all_perceptibilities.append(perceptibility)
                all_off_cell_bgrs.append(off_cell_bgr)
    
    return tot_hard_pred, tot_probs, tot_gt, sync_mags, all_perceptibilities, all_off_cell_bgrs


def process_signal(raw_cell_signal, rollingavg_n = 2, detrending_rollingmin_n = None, detrending_rollingavg_n = None):
    """
    Process the raw cell signal by smoothing it with a rolling average and detrending it.

    Args:
        raw_cell_signal (list of float): The raw cell signal to process.
        rollingavg_n (int): The size of the rolling average window for smoothing.
        detrending_rollingmin_n (int): The size of the rolling minimum window for detrending. If None, detrending will not be performed.
        detrending_rollingavg_n (int): The size of the rolling average window for detrending. If None, detrending will not be performed.
    
    Returns:
        smoothed_cell_signal (list of float): The smoothed cell signal.
        fully_processed_cell_signal (list of float): The fully processed cell signal after both smoothing and detrending.
        neg_env (list of float): The negative envelope of the cell signal, used for detrending (for debugging)
    """
    # smooth the cell signal with a rolling average
    smoothed_cell_signal = pd.Series(raw_cell_signal).rolling(rollingavg_n, center=True).mean().tolist()
    for i in range(int((rollingavg_n- 1)/2) + 1):
        smoothed_cell_signal[i] = smoothed_cell_signal[int((rollingavg_n- 1)/2) + 1]
    for i in range(int((rollingavg_n - 1)/2)):
        smoothed_cell_signal[-(i+1)] = smoothed_cell_signal[-(int((rollingavg_n- 1)/2) + 1)]

    # apply detrending to the smoothed signal
    fully_processed_cell_signal = None
    neg_env = None
    if detrending_rollingmin_n is not None:
        rolling_min = pd.Series(smoothed_cell_signal).rolling(detrending_rollingmin_n, center = True).min().tolist()
        for i in range(int((detrending_rollingmin_n - 1)/2)):
            if i >= len(rolling_min):
                break
            rolling_min[i] = rolling_min[min(int((detrending_rollingmin_n- 1)/2), len(rolling_min) - 1)]
        for i in range(int((detrending_rollingmin_n - 1)/2)):
            if i < 0:
                break
            rolling_min[-(i+1)] = rolling_min[max(int((detrending_rollingmin_n- 1)/2) * -1 + 1, 0)]

        neg_env = pd.Series(rolling_min).rolling(detrending_rollingavg_n, center=True).mean().tolist()
        left_fill = np.mean(smoothed_cell_signal[:int((detrending_rollingavg_n - 1)/2)])
        right_fill = np.mean(smoothed_cell_signal[-int((detrending_rollingavg_n - 1)/2):])
        for i in range(int((detrending_rollingavg_n - 1)/2)):
            if i >= len(neg_env):
                break
            neg_env[i] = left_fill
        for i in range(int((detrending_rollingavg_n - 1)/2)):
            if i < 0:
                break
            neg_env[-(i+1)] = right_fill
        smoothed_cell_signal = np.array(smoothed_cell_signal)
        neg_env = np.array(neg_env)
        fully_processed_cell_signal = smoothed_cell_signal - neg_env
        fully_processed_cell_signal = fully_processed_cell_signal.tolist()
    else:
        fully_processed_cell_signal = smoothed_cell_signal
        
    return smoothed_cell_signal, fully_processed_cell_signal, neg_env


def get_overall_sync_blinks(sync_signal, fps):
    """
    Given a sync cell signal, recorded at specified FPS return the indices corresponding 
    to frames in the signal at which a blink (either cell on or cell off) occurred

    Args:
        sync_signal (list of float): The sync cell signal to process.
        fps (int): Frame rate of the video, FPS

    Returns:
        target_sync_cell_blinks (list of int): The indices in the sync signal corresponding to frames at which a blink occurred.
                                                If there aren't the expected number of peaks and troughs, 
                                                they are added to ensure the correct number is present for decoding.
    """
    sync_signal = np.array(sync_signal) # ensure sync_signal is a numpy array
    min_dist = int((1/(config.frequency))*fps) * .4
    min_prominence = 0.05 #0.1
    sync_peaks, sync_peak_properties  = find_peaks(sync_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))
    sync_troughs, sync_trough_properties = find_peaks(-1*sync_signal, distance = min_dist, prominence = min_prominence, plateau_size=(None,None), height=(None, None), width = (None, None))

    if len(sync_peaks) == 0:
        #return dummy template
        template = [i* int((1/(config.frequency) / 2)*fps) for i in range(int(config.embedding_window_duration * config.frequency) * 2)]
        template = np.array(template)
        template += 3 #start???
        return template

    # filter peaks, considering only those in the largest continuous subsequence of detected peaks by prominence
    sync_peak_prominences = sync_peak_properties['prominences']
    _, target_sync_peak_is = maxSum(sync_peak_prominences, len(sync_peak_prominences), int(config.embedding_window_duration * config.frequency))

    # if short peaks, assess whether they should be at start or end and add as many as necessary
    num_missing_peaks = int(config.embedding_window_duration * config.frequency) - len(target_sync_peak_is)
    if num_missing_peaks > 0:
        num_added_peaks = 0
        added_peaks = []
        step = int((1/(config.frequency))*fps) # interpeak/intertrough distance
        if sync_peaks[target_sync_peak_is][0] < 10: # add to beginning if there appears to be space. Use 10 as a heuristic here.
            # add equally spaces peaks to start until correct number of peaks present or no space to add
            next_front_peak = sync_peaks[target_sync_peak_is][0] - step
            while num_added_peaks < num_missing_peaks and next_front_peak > 0:
                added_peaks.append(next_front_peak)
                next_front_peak -= step
                num_added_peaks += 1
        next_end_peak = sync_peaks[target_sync_peak_is][-1] + step # otherwise add to end
        while num_added_peaks < num_missing_peaks and next_end_peak < len(sync_signal):
            # add equally spaces peaks to end until correct number of peaks present or no space to add
            added_peaks.append(next_end_peak)
            next_end_peak -= step
            num_added_peaks += 1
        # in worst case, there are simply not enough peaks and the above attempts to reconcile this fail, 
        # so just tack on from end one after another so decoding can proceed
        if num_added_peaks < num_missing_peaks:
            next_end_peak = len(sync_signal) - 1
            while num_added_peaks < num_missing_peaks:
                added_peaks.append(next_end_peak)
                next_end_peak -= 1
                num_added_peaks += 1
        added_peaks = np.array(added_peaks)
        target_sync_peaks = np.concatenate((sync_peaks[target_sync_peak_is], added_peaks))
    else:
        # too many/just the right amount of peaks detected
        target_sync_peaks = sync_peaks[target_sync_peak_is]

    # filter troughs, only considering those occuring bewteen peaks
    target_sync_trough_is = []
    for i in target_sync_peak_is:
        for j, t in enumerate(sync_troughs):
            if i < len(sync_peaks) - 1:
                if t > sync_peaks[i] and t < sync_peaks[i + 1]:
                    target_sync_trough_is.append(j)
                    break 
    
    # if short or over troughs, assess whether they should be at start or end and add as many as necessary
    num_missing_troughs = int(config.embedding_window_duration * config.frequency) - len(target_sync_trough_is)
    if num_missing_troughs == 1:
        # most common case - just last trough missing. Using last peak as reference has worked well in practice.
        target_sync_troughs = np.concatenate((sync_troughs[target_sync_trough_is], np.array([target_sync_peaks[-1] + int((1/(config.frequency))*fps/2)])))
    elif len(target_sync_trough_is) != int(config.embedding_window_duration * config.frequency):
        # more than one trough missing, we will add multiple. Start by adding to the end with space from the last trough
        added_troughs = []
        step = int((1/(config.frequency))*fps) # interpeak/intertrough distance
        next_end_trough = target_sync_peaks[-1] + step
        num_added_troughs = 0
        while num_added_troughs < num_missing_troughs and next_end_trough < len(sync_signal):
            added_troughs.append(next_end_trough)
            next_end_trough += step
            num_added_troughs += 1
        # in worst case, there are simply not enough troughs and the above attempts to reconcile this fail, 
        # so just tack on from end one after another so decoding can proceed
        if num_added_troughs < num_missing_troughs:
            next_end_trough = len(sync_signal) - 1
            while num_added_troughs < num_missing_troughs:
                added_troughs.append(next_end_trough)
                next_end_trough -= 1
                num_added_troughs += 1
        added_troughs = np.array(added_troughs)
        target_sync_troughs = np.concatenate((sync_troughs[target_sync_trough_is], added_troughs))
        # target_sync_troughs = np.concatenate((sync_troughs[target_sync_trough_is], np.array([target_sync_peaks[-1] + int((1/(config.frequency))*fps/2)])))
    else:
        # too many/just the right amount of troughs detected
        target_sync_troughs = sync_troughs[target_sync_trough_is]

    target_sync_cell_blinks = np.concatenate((target_sync_peaks, target_sync_troughs)) 
    
    # sort all the blinks
    target_sync_cell_blinks = np.sort(target_sync_cell_blinks)
    return target_sync_cell_blinks


def decode_data_cell_signal(sync_cell_blinks, raw_cell_signal, plot_title = None, d = 0, gt = None, extra_vlines = None):
    """
    Decode a data cell signal using Manchester decoding and sync_cell_blinks as indicator of clock.
    Return both the hard decisions (0 or 1) and soft decisions (float representing nearness to 1 or 0).

    Args:
        sync_cell_blinks (list of int): The indices in the sync signal corresponding to frames at which a blink occurred.
                                        Should be even in number, with each pair representing the start of a bit period.
        raw_cell_signal (list of float): The raw cell signal to decode.
        plot_title (str): Title for plot, if desired. If no plot_title is provided, a plot will not be produced.
        d (int): Number of frames to average over when calculating the average value of the cell signal during each half-bit period.
                 Used to reduce noise when making decisions. Default is 0, meaning no averaging.
        gt (str): Ground truth bitstring for the cell, if available. If None, no ground truth comparisons will be performed.
        extra_vlines (list of int): Additional vertical lines to plot on the signal plot, if desired. Useful for debugging.


    Returns:
        pred_string (str): The hard predictions for the cell, where '0' and '1' represent the decoded bits.
        probs (list of float): The probabilities for the cell, where each element is a float representing the probability of the bit being '1'.
    """
    smoothed_cell_signal, fully_processed_cell_signal, neg_env = process_signal(raw_cell_signal, rollingavg_n = config.signal_rollingavg_n, detrending_rollingmin_n = config.signal_detrending_rollingmin_n, detrending_rollingavg_n = config.signal_detrending_rollingavg_n)

    # decode using Manchester decoding
    pred_string = ['0' for i in range(int(config.frequency * config.embedding_window_duration))]
    probs = [0 for i in range(int(config.frequency * config.embedding_window_duration))]
    diffs = []
    for i in range(0, len(sync_cell_blinks), 2):
        if i + 1 >= len(sync_cell_blinks):
            break
        t1 = int(sync_cell_blinks[i])
        t2 = int(sync_cell_blinks[i + 1])
        p1_avg = np.mean(fully_processed_cell_signal[t1:t1+d+1])
        p2_avg = np.mean(fully_processed_cell_signal[t2:t2+d+1])
        diffs.append(p1_avg - p2_avg)
        if p1_avg < p2_avg:
            pred_string[int(i/2)] = '1'
        probs[int(i/2)] = p1_avg - p2_avg
    pred_string = "".join(pred_string)

    # optionally plot
    if plot_title is not None:
        ymin = min(fully_processed_cell_signal)
        ymax = max(fully_processed_cell_signal)

        if config.signal_detrending_rollingmin_n:
            fig, axes = plt.subplots(2, tight_layout = True)
            if gt is not None:
                title = plot_title + f"\nGT:    {gt}"
            else:
                title = plot_title
            axes[0].plot(smoothed_cell_signal)
            axes[0].plot(neg_env, alpha=0.6)
            axes[1].plot(fully_processed_cell_signal)
            axes[1].vlines(sync_cell_blinks, ymin, ymax, linestyles = "dashed", colors = "grey")
            if extra_vlines is not None:
                axes[1].vlines(extra_vlines, ymin, ymax, linestyles = "dashed", colors = "red", alpha=0.5)
            for i in range(0, len(sync_cell_blinks), 2):
                axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/3, f"{probs[int(i/2)]:.2f}")
                if gt is not None:
                    if pred_string[int(i/2)] != gt[int(i/2)]:
                        axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)], c = 'r')
                        axes[0].set_title(f"{title}\nPred: {pred_string}", fontdict ={'color':'red','size':10})
                    else:
                        axes[0].set_title(f"{title}\nPred: {pred_string}", fontsize = 10)
                else:
                    axes[1].text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)])   
                    axes[0].set_title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.show()
        else:
            if gt is not None:
                title = plot_title + f"\nGT:   {gt}"
            else:
                title = plot_title
            plt.title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.plot(smoothed_cell_signal)
            plt.vlines(sync_cell_blinks, ymin, ymax, linestyles = "dashed", colors = "grey")
            if extra_vlines is not None:
                plt.vlines(extra_vlines, ymin, ymax, linestyles = "dashed", colors = "red", alpha=0.5)
            for i in range(0, len(sync_cell_blinks), 2):
                plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/3, f"{probs[int(i/2)]:.2f}")
                if gt is not None:
                    if pred_string[int(i/2)] != gt[int(i/2)]:
                        plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)], c = 'r') 
                else:
                    plt.text(sync_cell_blinks[i], ymin + (ymax - ymin)/2, pred_string[int(i/2)])
                    plt.title(f"{title}\nPred: {pred_string}", fontsize = 10)
            plt.show()
            plt.clf()
            
    return pred_string, probs

