
'''
Given a dynamic hash extracted from the digest, and a video, this script scans through videos 
starting/end at small shift (milliseconds-level) from the predicted window start marker 
to find the shift that leads to the smallest dynamic hash distance to the input dynamic hash.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
'''
import cv2
import warnings
from sklearn.exceptions import  InconsistentVersionWarning
warnings.filterwarnings(action='ignore', category =  InconsistentVersionWarning)
warnings.filterwarnings(action='ignore', category = FutureWarning)

import common.config as config
from common.digest_extraction import create_dynamic_hash_from_dynamic_features
from common.rp_lsh import hamming

def get_dynamic_hash_dist(video_path, win_start_frame, input_digest, dynamic_features):
    """
    Get the distance between the input dynamic hash and the optimal dynamic hash found by exploring a
    small range of frames around the predicted window start frame.

    Args:
        video_path (str): Path to the video file.
        win_start_frame (int): Predicted start frame of the window.
        input_digest (list): The dynamic hash to compare against.
        dynamic_features (list): List of dynamic features extracted from the video.
    
    Returns:
        tuple: A tuple containing the minimum dynamic hash distance and the optimal dynamic hash.
    """
    epsilon = config.window_alignment_epsilon

    # compare at minor (millisecond) offsets of predicted window boundary
    dynamic_fam = config.dynamic_fam
    dynamic_hash_funcs = config.dynamic_hash_funcs 
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    scan_start = win_start_frame - epsilon
    scan_end = win_start_frame + epsilon
    min_dyn_feat_dist = config.dynamic_hash_k
    opt_dynamic_hash = None
    for i in range(scan_start, int(scan_end+1)):
        if int(i + config.video_window_duration*fps+1) >= len(dynamic_features):
            break
        curr_dynamic_features = dynamic_features[i:int(i + config.video_window_duration*fps+1)]
        curr_dynamic_hash, curr_raw_signals, curr_proc_signals, curr_concat_processed_signal  = create_dynamic_hash_from_dynamic_features(curr_dynamic_features, dynamic_fam, dynamic_hash_funcs)
        curr_dynamic_hash_dist =  hamming(input_digest, curr_dynamic_hash)
        if curr_dynamic_hash_dist < min_dyn_feat_dist:
            min_dyn_feat_dist = curr_dynamic_hash_dist
            opt_dynamic_hash = curr_dynamic_hash

    return min_dyn_feat_dist, opt_dynamic_hash




