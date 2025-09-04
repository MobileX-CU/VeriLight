"""
Verify a video
Running with command-line arguments will use default verification settings. 
See verify() function for details on alternative settings.

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import cv2
from scipy.signal import find_peaks
from colorama import Fore, Style
from argparse import ArgumentParser

import sys
import hmac
import hashlib

import common.config as config
from common.digest_extraction import VideoFeatureExtractor
from common.decoding_utils import loadVideo, get_homography, get_loc_marker_center, valid_r_c, get_nonloc_cell_signal_from_imgseq, butter_bandpass_filter
from common.decode_sequence import decode_sequence
from common.bitstring_utils import bytes_to_bitstring, bitstring_to_bytes
from common.rp_lsh import hamming

from embedding.calibration_utils import  get_user_points, detect_heatmap_cells, order_calibration_code_corners
from embedding.psk_encode_minimal import create_sample_frame

from verification.create_heatmap import create_localization_heatmap
from verification.dynamic_hash_check import get_dynamic_hash_dist
from verification.visualization import visualize_ver_results, print_results_summary

# decision thresholds. same as those used for paper evaluation
ID_THRESH = 42
DYN_THRESH = 56

def generate_localization_reference_corners(display = False):
    """
    Generate coordinates of center of localization corners in a bitmap,
    for use as the correspondences for homography calculation.

    Args:
        display (bool): Whether to display the reference image with localization corners (debugging)

    Returns:
        ref_img (numpy array): Reference image with localization corners drawn on it
        ref_corners (list): List of tuples containing the coordinates of the localization corners
    """
    ref_img = np.zeros((config.slm_H, config.slm_W)).astype(np.float32)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)

    assert config.localization_N is not None, "Localization cell type not supported. Please set config.localization_N to a valid value."
    ref_corners = []
    for i in range(4):
        c = get_loc_marker_center(i, config.slm_W, config.slm_H, config.N, config.buffer_space, config.localization_N)
        ref_corners.append(c)
        cv2.circle(ref_img, c, 2, (0, 0, 255), -1) 

    if display:
        cv2.imshow("Localization reference", ref_img)
        cv2.waitKey(0)
    return ref_img, ref_corners


def localize(video_path, heatmap_settings = None, output_path = "", force_recalculate_homography = False, manually_approve = False, display_localization_progress = False):
    """
    Search video for localization corners and calculate homography to apply to 
    the video frames to rectify it to canonical bitmap view.

    Args:
        video_path (str): Path to the video file
        heatmap_settings (dict or None): Settings for localization heatmap. If None, default settings stored in create_heatmap.py will be used.
                                        See create_heatmap(), detect_heatmap_cells(), and order_calibration_code_corners() for details on each setting.
        output_path (str): Path to save the output files
        force_recalculate_homography (bool): Whether to force recalculation of homography even if it already exists in the output folder
        manually_approve (bool): Whether to manually approve the inferred localization corners via command line prompt (useful for debugging/experiments).
                                Otherwise, the corners are automatically accepted and used for homography calculation.
        display_localization_progress (bool): Whether to display the localization heatmap and inferred corners 
    
    Returns:
        Hom (numpy array): Homography matrix if successful, None otherwise  
        sorted_corner_centers (list): List of tuples containing the coordinates of the detected localization corners in the video frame
    """
    
    assert os.path.exists(video_path), f"Video file not found at {video_path}"

    if heatmap_settings is None:
        from verification.create_heatmap import default_heatmap_settings
        heatmap_settings = default_heatmap_settings
    
    # if homography.pkl already exists at output_path, load it, otherwise calculate it from the heatmap
    heatmap_path = f"{output_path}/heatmap.png"
    hom_path = f"{output_path}/homography.pkl"
    if not os.path.exists(hom_path) or force_recalculate_homography:
        if os.path.exists(heatmap_path):
            heatmap = cv2.imread(heatmap_path, cv2.IMREAD_GRAYSCALE)
        else:
            heatmap, _, _ = create_localization_heatmap(video_path,
                                                        frame_range = heatmap_settings["frame_range"],
                                                        denoise = True, display = False)
            cv2.imwrite(heatmap_path, heatmap)
        
        corner_centers, corner_bboxes = detect_heatmap_cells(heatmap, density_diameter = heatmap_settings["density_diameter"], density_threshold  = heatmap_settings["density_threshold"], otsu_inc = heatmap_settings["otsu_inc"],  erode = heatmap_settings["erode"], area_threshold = heatmap_settings["area_threshold"], blurthensharp = heatmap_settings["blurthensharp"], kernel_dim = heatmap_settings["kernel_dim"], min_squareness = heatmap_settings["min_squareness"], display = display_localization_progress)
        
        # get source corners to use for homography
        localization_result = order_calibration_code_corners(corner_centers, heatmap, slope_epsilon = heatmap_settings["slope_epsilon"], display = False)
        if manually_approve:
            if localization_result is not None:
                sorted_corner_centers, labeled_heatmap = localization_result
                cv2.imshow("Inferred calibration corners", labeled_heatmap)
                cv2.waitKey(0)
                accept_inferred_corners = input("Are the inferred calibration corners ok? y/n")
            if localization_result is None or accept_inferred_corners == "n":
                vis_corners = heatmap.copy()
                vis_corners = cv2.cvtColor(vis_corners, cv2.COLOR_GRAY2BGR)
                for c in corner_centers:
                    cv2.circle(vis_corners, c, 2, (0, 0, 255), -1)

                sorted_corner_centers = get_user_points(vis_corners) #corners MUST be in order topleft, topright, bottom left, bottom right
                if len(sorted_corner_centers) == 0:
                    sys.exit()
        else:
            if localization_result is None:
                print("No localization corners found. Cannot proceed with homography calculation.")
                return None
            sorted_corner_centers, _ = localization_result
                    
        # perform homography between heatmap and a reference for visualization
        try:
            _, reference_corner_centers = generate_localization_reference_corners(display = False)
            sample_frame = create_sample_frame()
            Hom = get_homography(sorted_corner_centers, reference_corner_centers, heatmap, sample_frame, display = display_localization_progress) 
        except Exception as e:
            print(f"Error computing homography: {e}")
            return None

        if Hom is None:
            print("Invalid homography (cv2.findHomography returned None).")
            return None
    
        # save homography to file for future re-use
        with open(f"{output_path}/homography.pkl", "wb") as pklfile:
            pickle.dump(Hom, pklfile)
            pickle.dump(sorted_corner_centers, pklfile)
    else:
        print(f"Loading existing homography from {hom_path}...")
        Hom = pickle.load(open(hom_path, "rb"))
        sorted_corner_centers = pickle.load(open(hom_path, "rb"))
    
    return Hom, sorted_corner_centers
    

def detect_window_boundaries(img_seq, fps, display = False, output_path = "", force_repredict_interwin = False,  plot_save_path = None):
    """
    Determine the start/end frame of each window by using the synchronization cell signals

    Args:
        img_seq (list of numpy array): List of numpy array of shape (H, W, 3) containing the video frames
                                with the homography already applied (i.e., only contains the projection region)
        fps (int): Frame rate of the video, FPS
        display (bool): Whether to display plot with the interwindow detection results and sync signals
        output_path (str): Path to save the output files
        force_repredict_interwin (bool): Whether to force recalculation of interwindow boundaries even if they already exist in the output folder
        plot_save_path (str or None): If not None, path to create and save plot with the interwindow detection results and sync signals
    
    Returns:
        pred_window_boundaries (list): List of frame indices indicating the start of each window
        main_sync_signal (numpy array): The main sync signal (i.e., average of all sync cell signals) used for interwindow detection
    """
    # if pred_window_boundaries.pkl already exists, load it, otherwise determine from img_seq
    pred_window_boundaries_path = f"{output_path}/pred_window_boundaries.pkl"
    if not os.path.exists(pred_window_boundaries_path) or force_repredict_interwin:
        # get all sync cell signals
        all_sync_signals = []
        for r in range(config.max_cells_H):
            for c in range(config.max_cells_W): 
                if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N) or f"{r}-{c}" in config.reserved_localization_cells:
                    continue
                if (r == 0 or r == config.max_cells_H - 1 or c == 0 or c == config.max_cells_W - 1): # this a border cell, and not in reserved localization cells, so must be a sync cell
                    cell_signal = get_nonloc_cell_signal_from_imgseq(img_seq, r, c)
                    all_sync_signals.append(cell_signal)
            
        # use average of all separate sync cell signals as one main sync signal
        all_sync_signals = np.array(all_sync_signals)
        main_sync_signal = np.mean(np.array(all_sync_signals), axis = 0)

        # bandpass filter the main sync signal around the expected frequency
        bp_main_sync_signal = main_sync_signal - main_sync_signal.mean()
        bp_main_sync_signal = butter_bandpass_filter(bp_main_sync_signal, config.frequency - config.interwin_detection_bandpass_tolerance, config.frequency + config.interwin_detection_bandpass_tolerance, fps, order=5)
        bp_main_sync_signal = pd.Series(bp_main_sync_signal)

        # get upper envelope of bandpassed main sync signal
        rolling_max = bp_main_sync_signal.rolling(config.interwin_upper_env_rollingmax_n, center = True).max().tolist()
        for i in range(int((config.interwin_upper_env_rollingmax_n - 1)/2)):
            rolling_max[i] = rolling_max[int((config.interwin_upper_env_rollingmax_n - 1)/2)]
        for i in range(int((config.interwin_upper_env_rollingmax_n - 1)/2)):
            rolling_max[-(i+1)] = rolling_max[-(int((config.interwin_upper_env_rollingmax_n - 1)/2) + 1)]
        pos_env = pd.Series(rolling_max).rolling(config.interwin_upper_env_rollingavg_n, center = True).mean().tolist()
        left_fill = np.mean(bp_main_sync_signal[:int((config.interwin_upper_env_rollingavg_n - 1)/2)])
        right_fill = np.mean(bp_main_sync_signal[-int((config.interwin_upper_env_rollingavg_n - 1)/2):])
        for i in range(int((config.interwin_upper_env_rollingavg_n - 1)/2)):
            pos_env[i] = left_fill
        for i in range(int((config.interwin_upper_env_rollingavg_n - 1)/2)):
            pos_env[-(i+1)] = right_fill
        pos_env = np.array(pos_env)

        # use troughs in upper envelope of bandpassed loc signal as indicators of an interwindow period
        pred_window_boundaries, _ = find_peaks(-1 * pos_env, distance = config.min_interwin_time * fps)

        # optional visualization of interwindow detection results
        if display or plot_save_path is not None:
            print("Plotting interwindow detection results...")
            fig, axes = plt.subplots(2, tight_layout=True, figsize=(12, 6))
            plt.suptitle(f"Interwindow Boundaries\nAveraged Sync Cell")
            axes[0].set_title("Raw Signal")
            axes[0].plot(main_sync_signal)
            axes[1].set_title(f"Bandpassed w/ Envelope (rolling max n = {config.interwin_upper_env_rollingmax_n}, rolling avg n = {config.interwin_upper_env_rollingavg_n})")
            axes[1].plot(bp_main_sync_signal)
            axes[1].vlines(pred_window_boundaries, min(bp_main_sync_signal), max(bp_main_sync_signal), color = 'r', linestyle = 'dashed')
            axes[1].plot(pos_env, color = "purple", alpha = 0.5)
            if display:
                plt.show()
            if plot_save_path:
                fig.savefig(plot_save_path)
            plt.close()

            # detailed per-window sync signal visualization, uncomment if desired for debugging
            # for i in range(len(pred_window_boundaries)):
            #     vis_window_sync_signal(i, main_sync_signal, pred_window_boundaries, display = display)

        with open(f"{output_path}/pred_window_boundaries.pkl", "wb") as f:
            pickle.dump(pred_window_boundaries, f)
    else:
        with open(pred_window_boundaries_path, "rb") as f:
            pred_window_boundaries = pickle.load(f)
    
    return pred_window_boundaries


def get_digest_from_signature(payload):
    """
    Given signature (i.e., digest + HMAC tag), extract digest and return them along with decision whether it passed HMAC validation
    
    Args:
        payload (str): Bitstring of length config.signature_size containing the raw data embedded in the video

    Returns:
        digest (str): Bitstring of length config.digest_size containing the digest extracted from the payload
        bin_seq_num (str): Bitstring of length config.bin_seq_num_size containing the binary sequence number extracted from the digest
        id_feat_hash_half (str or None): Bitstring of length config.identity_hash_k // 2 containing the identity feature hash half extracted from the digest, or None if the ID feature couldn't be extracted during embedding
        dynamic_feat_hash (str): Bitstring of length config.dynamic_hash_k containing the dynamic feature hash extracted from the digest
        pass_checksum (bool): Whether the digest passed HMAC validation
    """
    digest = payload[:config.digest_size] 
    tag = payload[config.digest_size:config.digest_size + config.tag_size * 8]
  
    digest_bytes = bitstring_to_bytes(digest)
    h = hmac.new(config.key, digest_bytes, hashlib.sha256)
    comp_tag = h.digest()[:config.tag_size]
    comp_tag_bits = bytes_to_bitstring(comp_tag)

    if comp_tag_bits != tag:
        pass_checksum = False
    else:
        pass_checksum = True

    bin_seq_num = digest[:config.bin_seq_num_size]

    id_feat_hash_half = digest[config.bin_seq_num_size:config.bin_seq_num_size + config.identity_hash_k // 2]
    # if id_feat_hash_half is all zeros, that means the ID feature couldn't be extracted on core unit during embedding.
    # this is the convention established in digest_extraction.py's create_digest_from_features() 
    if id_feat_hash_half.count("0") == config.identity_hash_k // 2:
        id_feat_hash_half = None

    dynamic_feat_hash = digest[config.bin_seq_num_size + config.identity_hash_k // 2:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k] 

    # extra metadata not needed for current verification implementation but generally helpful for provenance/forensics
    unit_id = digest[config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k:config.bin_seq_num_size + config.identity_hash_k // 2 + config.dynamic_hash_k + config.unit_id_size]
    date_ordinal = digest[-config.date_ordinal_size:]
   
    return digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, pass_checksum

def recover_digests(img_seq, fps, pred_window_boundaries, force_rerecover_digests = False,  output_path = "", display_demodulation_progress = False):
    """
    Recover digests from each window of video.

    Args:
        img_seq (list of numpy array): List of numpy arrays of shape (H, W, 3) containing the video frames
                                with the homography already applied (i.e., only contains the projection region)
        fps (int): Frame rate of the video, FPS
        pred_window_boundaries (list): List of frame indices indicating the start of each window
        force_rerecover_digests (bool): Whether to force recalculation of digests even if they already exist in the output folder
        output_path (str): Path to save the output files
        display_demodulation_progress (bool): Whether to display the decoded sequences during recovery
    
    Returns:
        digest_components (list): List of recovered digest components for each window
        boundaries_by_seq (dict): Dictionary mapping sequence numbers to their frame boundaries
    """
     # if rec_digests.pkl already exists at output_path, load it, otherwise calculate it from the img_seq
    rec_digests_path = f"{output_path}/recovered_digest_components.pkl"
    if os.path.exists(rec_digests_path) and not force_rerecover_digests:
        with open(rec_digests_path, "rb") as pkfile:
            digest_components = pickle.load(pkfile)
            window_boundaries = pickle.load(pkfile)
    else:
        
        # decode each window
        curr_boundary_index = 0
        seq_num = 0
        last_decodable_seq = False
        curr_id_hash = None
        last_full_id_hash = None
        digest_components = []
        window_boundaries = []
    
        while not last_decodable_seq:
            print(f"Decoding sequence {seq_num}...")

            if curr_boundary_index == len(pred_window_boundaries) - 1:
                this_img_seq = img_seq[pred_window_boundaries[curr_boundary_index]:, : , :, :]
                last_decodable_seq = True
                window_boundaries.append((pred_window_boundaries[curr_boundary_index], len(img_seq)))
            else:
                this_img_seq = img_seq[pred_window_boundaries[curr_boundary_index]:pred_window_boundaries[curr_boundary_index + 1], : , :, :]
                window_boundaries.append((pred_window_boundaries[curr_boundary_index], pred_window_boundaries[curr_boundary_index + 1]))

            tot_hard_pred, tot_probs, _, _ ,  _, _ = decode_sequence(this_img_seq, fps, display = display_demodulation_progress)
            
            # error correct
            correctable_payload = True
            try:
                extracted_signature , _, _ = config.error_corrector.decode_payload(tot_probs[:config.viterbi_signature_size])
                extracted_signature = extracted_signature[:config.signature_size]
            except Exception as e:
                print(f"Unrecognizable error recovering encountered Seq {seq_num}. Reported error: {e}.")
                correctable_payload = False
            
            if correctable_payload:
                digest, bin_seq_num, id_feat_hash_half, dynamic_feat_hash, pass_checksum = get_digest_from_signature(extracted_signature)
                rec_seq_num  = int(bin_seq_num, 2)

                if id_feat_hash_half is not None:
                    if rec_seq_num % 2 == 0:
                        curr_id_hash = id_feat_hash_half
                    else:
                        if curr_id_hash is not None: 
                            curr_id_hash += id_feat_hash_half
                            if len(curr_id_hash) == config.identity_hash_k:
                                # only update last full id hash when we have the full hash. If we don't, we keep the previous one.
                                # this protects against case where previous half was not recovered due to corruption
                                last_full_id_hash = curr_id_hash 
                if pass_checksum:
                    digest_components.append([1, rec_seq_num, last_full_id_hash, dynamic_feat_hash])
                    #print(Fore.GREEN + f"Recovered digest from encountered Seq {seq_num} (actual Seq num: {rec_seq_num})." + Style.RESET_ALL)
                else:
                    digest_components.append([0, rec_seq_num, last_full_id_hash, dynamic_feat_hash])
                    #print(Fore.RED + f"Failed to recover digest from encountered Seq {seq_num}  because of checksum failure." + Style.RESET_ALL)
            else:
                digest_components.append([0, None, None, None])
                #print(Fore.RED + f"Failed to recover digest for encountered Seq {seq_num} because of error correction failure." + Style.RESET_ALL)

            curr_boundary_index += 1
            seq_num += 1
              
        with open(rec_digests_path, "wb") as pkfile:
            pickle.dump(digest_components, pkfile)
            pickle.dump(window_boundaries, pkfile)

    return digest_components, window_boundaries


def verify(video_path, output_path, heatmap_settings = None, 
            force_recalculate_homography = False, force_repredict_interwin = False, force_rerecover_digests = False, force_reverify = False,
            display_localization_progress = False, display_interwin_boundary_predictions = False, display_demodulation_progress = False):
    """
    Verify a video containing Verilight signatures.
    This function will localize the video, detect interwindow boundaries, recover digests, and valid the video content against the recovered digests.
    Outputs from each of these stages will be saved as followsllowing paths so that they can be reused in future runs or examined:
    - {output_path}/localization.pkl: contains the homography matrix and sorted contour centers
    - {output_path}/heatmap.png: the localization heatmap used for localization
    - {output_path}/interwindow_boundaries.pkl: contains the detected interwindow boundaries
    - {output_path}/recovered_digests.pkl: contains the recovered digests
    - {output_path}/final_results.pkl: contains the final verification results

    Args:
        video_path (str): Path to the video file
        output_path (str): Path to save the output files. Will be created if it doesn't exist.
        heatmap_settings (dict or None): Settings for extracting and detecting contours in localization heatmap. If None, default settings stored in create_heatmap.py will be used.
                                         See create_heatmap(), detect_heatmap_cells(), and order_calibration_code_corners() for details on each setting.
        force_recalculate_homography (bool): Whether to force recalculation of homography even if it already exists in the output folder at {output_path}/homography.pkl
        force_repredict_interwin (bool): Whether to force recalculation of interwindow boundaries even if they already exist in the output folder at {output_path}/interwindow_boundaries.pkl
        force_rerecover_digests (bool): Whether to force recalculation of digests even if they already exist in the output folder at {output_path}/recovered_digests.pkl
        force_reverify (bool): Whether to force re-verification even if final results already exist in the output folder at {output_path}/final_results.pkl
        display_localization_progress (bool): Whether to display the localization heatmap and inferred corners at different stages of localization
        display_interwin_boundary_predictions (bool): Whether to display the interwindow boundary predictions and sync signals
        display_demodulation_progress (bool): Whether to display the decoded sequences during recovery
    """

    print(Fore.BLUE + f"------------------------------- VERIFYING {video_path} -------------------------------" + Style.RESET_ALL)
    
    # make sure video exists at video_path
    if not os.path.exists(video_path):
        print(Fore.RED + f"Can't find video file at {video_path}" + Style.RESET_ALL)
        return
    try:
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if frame_rate == 0:
            print("ERROR: Corrupt video.") # this is also a reliable way to check if the video is corrupt, as it should never occur for a valid video
            return
    except:
        print(f"ERROR: Can't open video file at {video_path}")
        return
    
    # create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    if os.path.exists(f"{output_path}/final_results.pkl") and not force_reverify:
        final_results = pickle.load(open(f"{output_path}/final_results.pkl", "rb"))
    else:
        # localize
        print("Localizing signature and getting homography...")
        Hom, sorted_contour_centers = localize(video_path, heatmap_settings = heatmap_settings, force_recalculate_homography = force_recalculate_homography, output_path = output_path, display_localization_progress = display_localization_progress)
        if Hom is None:
            print("Couldn't localize signature and obtain homography.")
            return

        # load video for interwindow prediction and digest recovery
        print("Loading video frames...")
        img_seq, fps = loadVideo(video_path, colorspace = config.colorspace, Hom = Hom)
        
        # get interwin predictions
        print("Predicting interwindow frames...")
        pred_window_boundaries = detect_window_boundaries(img_seq, fps, force_repredict_interwin = force_repredict_interwin, display = display_interwin_boundary_predictions, output_path = output_path, plot_save_path = f"{output_path}/pred_interwindow_boundaries_plot.png")

        # recover digests and clean up window boundaries to ensure valid
        print("Recovering digests...")
        digest_components, window_boundaries = recover_digests(img_seq, fps, pred_window_boundaries, force_rerecover_digests = force_rerecover_digests, output_path = output_path, display_demodulation_progress = display_demodulation_progress)
        
        #initialize video digest extractor and extract MP features
        print("Extracting MP features...")
        vid_feature_extractor = VideoFeatureExtractor(video_path, output_path)
        dynamic_features, pose, face_bbox, raw_detection_results = vid_feature_extractor.extract_mp_features()

        print("Verifying sequences...")
        num_verified_seqs = 0
        final_results = []
        for i in range(len(window_boundaries) - 1):
            reference_digest = digest_components[i + 1] 
            if i == len(digest_components) - 2 and \
                (window_boundaries[i + 1][1] - window_boundaries[i + 1][0] < fps * config.video_window_duration * 0.9) and \
                reference_digest[0] == 0:
                # if the last window is too short to contain an intact signature, we can't verify the second-to-last window's video, 
                # this is expected behavior -- not due to corruption -- so we skip it without printing an error message
                continue

            start_frame, end_frame = window_boundaries[i]

            # get components of the digest embedded in the next window, whose contents we will use to verify this window's video
            if reference_digest[0] == 0:
                print("Window signature cannot be recovered. Skipping verification.")
                print("------------------------------------------------")
                final_results.append({
                    "id_dist": -1,
                    "dyn_dist": -1,
                    "id_hash": None,
                    "dyn_hash": None,
                    "rec_id_hash": None,
                    "rec_dyn_hash": None,
                    "rec_seq_num": None,
                    "boundaries": (start_frame, end_frame)
                })
                continue
            rec_seq_num = reference_digest[1]
            rec_id_hash = reference_digest[2]
            rec_dynamic_hash = reference_digest[3]
        
            # validate the features hashes
            print(f"------------- VIDEO WINDOW #: {i}. CORE UNIT SIGNATURE #: {rec_seq_num}.  -------------")

            # make sure the determined window is long enough for legitimate analysis
            if end_frame - start_frame < config.video_window_duration * fps * 0.9: # add some tolerance with 0.9, otherwise valid windows are discounted
                print(f"Window too short (should be at least {config.video_window_duration * fps * 0.9} frames, is {end_frame - start_frame} frames long). Skipping.")
                final_results.append({
                    "id_dist": -1,
                    "dyn_dist": -1,
                    "id_hash": None,
                    "dyn_hash": None,
                    "rec_id_hash": rec_id_hash,
                    "rec_dyn_hash": rec_dynamic_hash,
                    "rec_seq_num": rec_seq_num,
                    "boundaries": (start_frame, end_frame)
                })
                continue
            
            # validate the ID feature hash
            if rec_id_hash is None:
                print("ID hash could not be recovered for this window. If this window has an even window number, this is expected, as ID hashes are split across two consecutive windows.")
                id_dist = -1
                id_hash = None
            else:
                # verify every frame in the window against the recovered ID hash
                # max_window_id_hash_dist = -1
                # max_window_id_hash = None
                # for frame_num in range(start_frame, end_frame):
                #     ver_id_hash, ver_id_vec = vid_feature_extractor.get_id_features_hash(frame_num)
                #     if ver_id_hash is None: 
                #         id_hash_dist = -1
                #     else:
                #         id_hash_dist = hamming(ver_id_hash, rec_id_hash)
                #     if id_hash_dist > max_window_id_hash_dist:
                #         max_window_id_hash_dist = id_hash_dist
                #         max_window_id_hash = ver_id_hash
                # or only verify the first frame in the window ID feature (speeds up verification)
                # under assumption that the speaker's identity won't randomly change during the window
                id_hash, ver_id_vec = vid_feature_extractor.get_id_features_hash(start_frame)
                if id_hash is None:
                    max_window_id_hash_dist = -1
                    max_window_id_hash = None
                else:
                    max_window_id_hash_dist = hamming(id_hash, rec_id_hash)
                    max_window_id_hash = id_hash
                id_dist = max_window_id_hash_dist
                id_hash = max_window_id_hash
                print(f"Window max ID hash Hamming distance: {max_window_id_hash_dist}")

            # validate the dynamic feature hash
            if end_frame - start_frame > config.video_window_duration * fps * 1.05: # don't use excessively large windows
                start_frame = end_frame - int(config.video_window_duration * fps * 1.05)
            dynamic_hash_dist, dynamic_hash = get_dynamic_hash_dist(video_path, start_frame, rec_dynamic_hash, dynamic_features)
            final_results.append({
                "id_dist": id_dist,
                "dyn_dist": dynamic_hash_dist,
                "id_hash": id_hash,
                "dyn_hash": dynamic_hash,
                "rec_id_hash": rec_id_hash,
                "rec_dyn_hash": rec_dynamic_hash,
                "rec_seq_num": rec_seq_num,
                "boundaries": (start_frame, end_frame)
            })
            print(f"Window dynamic hash Hamming distance: {dynamic_hash_dist}.")
            print("------------------------------------------------")

            num_verified_seqs += 1
        
        # save final results
        with open(f"{output_path}/final_results.pkl", "wb") as f:
            pickle.dump(final_results, f)
        
    # print final results
    print("--------------------- FINAL RESULTS --------------------- ")
    max_id = max([result["id_dist"] for result in final_results])
    max_dyn = max([result["dyn_dist"] for result in final_results])
    if max_id > ID_THRESH:
        print(Fore.RED + f"Maximum ID feature hash distance across windows: {max_id}. Exceeds threshold of {ID_THRESH}. Evidence of identity manipulation." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"Maximum ID feature hash distance across windows: {ID_THRESH}. Below threshold of {ID_THRESH}. No evidence of identity manipulation." + Style.RESET_ALL)
    if max_dyn > DYN_THRESH:
        print(Fore.RED + f"Maximum dynamic feature hash distance across windows: {max_dyn}. Exceeds threshold of {DYN_THRESH}. Evidence of lip and/or facial motion manipulation." + Style.RESET_ALL)
    else:
        print(Fore.GREEN + f"Maximum dynamic feature hash distance across windows: {DYN_THRESH}. Below threshold of {DYN_THRESH}. No evidence of lip and/or facial motion manipulation." + Style.RESET_ALL)
    print_results_summary(final_results, ID_THRESH, DYN_THRESH)
    print("------------------------------------------------")
    print("Verification complete.")

if __name__ == "__main__":
    parser = ArgumentParser(description="Verify a video containing Verilight signatures.")
    parser.add_argument("video_path", type=str, help="Path to the video to verify")
    parser.add_argument("output_path", type=str, help="Path to save the output files")
    parser.add_argument("--force_recalculate_homography", "-frh", action="store_true", help="Force recalculation of homography even if it already exists in the output folder")
    parser.add_argument("--force_repredict_interwin", "-fri", action="store_true", help="Force recalculation of interwindow boundaries even if they already exist in the output folder")
    parser.add_argument("--force_rerecover_digests", "-frd", action="store_true", help="Force recalculation of digests even if they already exist in the output folder")
    parser.add_argument("--force_reverify", "-frv", action="store_true", help="Force re-verification even if final results already exist in the output folder")
    parser.add_argument("--visualization", "-vis", action="store_true", help="Generate video visualizing the verification process and results")
    args = parser.parse_args()

    verify(args.video_path, args.output_path, 
            force_recalculate_homography = args.force_recalculate_homography, 
            force_repredict_interwin = args.force_repredict_interwin, 
            force_rerecover_digests = args.force_rerecover_digests, 
            force_reverify = args.force_reverify)
    if args.visualization:
        visualize_ver_results(args.video_path, args.output_path, ID_THRESH, DYN_THRESH)
