"""
Process to extract features from video frames and serialize them into a payload.
Run in parallel with the embedding process.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import os
import glob
from queue import Queue
import time
import numpy as np
from datetime import datetime
import threading
import pickle 
from colorama import Fore, Back, Style
import hmac
import hashlib
import random
import warnings
warnings.filterwarnings(action='ignore', category = FutureWarning)
warnings.filterwarnings(action='ignore', category = UserWarning)

import common.config as config
from common.digest_extraction import MPExtractor, IdentityExtractor, create_digest_from_features
from common.error_correctors import ReedSolomon, ConcatenatedViterbiRS, SoftViterbi
from common.bitstring_utils import pad_bitstring, bytes_to_bitstring, bitstring_to_bytes

got_kill_signal = False

def serialize(cv, feature_queue):
    """
    Convert features to a serialized signature, taking care of adding the extra components as well as encryption or tagging

    Args:
        cv (threading.Condition): Condition variable for thread synchronization
        feature_queue (Queue): Queue to get features from

    Returns:
        None. Saves all signatures and intermediate components as files accessed by other threads/procs
    """
    while True:
        with cv:
            cv.wait(0.00001)
            interrupted = got_kill_signal  # Read got_kill_signal while holding the lock
        if interrupted:
            break
        feature_seq_num, dynamic_features, identity_features, img_nums = feature_queue.get()
        
        print(Fore.MAGENTA + f"Got features in serialization thread at {datetime.fromtimestamp(time.time())}." + Style.RESET_ALL)
    
        digest, _, _ = create_digest_from_features(dynamic_features, identity_features, feature_seq_num, output_path = f"{config.session_output_path}/features/signals_seq{feature_seq_num}.pkl", img_nums = img_nums)
        
        file = open(f"{config.session_output_path}/embedded_data/raw_digest_{feature_seq_num}.txt", "w") # before encryption, error corection or anything
        file.write(digest)
        file.close()

        # use HMAC for integrity and authenticity. Digest contains the date ordinal and unit ID too.
        date_ordinal = datetime.now().toordinal() - config.creation_date_ordinal
        date_ordinal_bits = np.binary_repr(date_ordinal, width = config.date_ordinal_size)
        unit_id_bits = np.binary_repr(config.unit_id, width = config.unit_id_size)
        assert len(unit_id_bits) == config.unit_id_size, f"Unit ID {config.unit_id} is not of size {config.unit_id_size} bits. Decrease the unit ID or increase the unit_id_size in config.py."
        assert len(date_ordinal_bits) == config.date_ordinal_size, f"Date ordinal {date_ordinal} is not of size {config.date_ordinal_size} bits. Increase the creation_date_ordinal or increase the date_ordinal_size in config.py."
        digest = digest + unit_id_bits + date_ordinal_bits
        digest_bytes = bitstring_to_bytes(digest)
        hmac_obj = hmac.new(config.key, digest_bytes, hashlib.sha256) 
        hmac_tag = hmac_obj.digest()[:config.tag_size]
        hmac_tag_bits = bytes_to_bitstring(hmac_tag)
        signature = digest + hmac_tag_bits

        # dump signature, i.e., payload prior to any error correction
        file = open(f"{config.session_output_path}/embedded_data/payload_{feature_seq_num}.txt", "w") 
        file.write(signature)
        file.close()

        # add error correction
        if type(config.error_corrector) == ReedSolomon:
            coded_signature = config.error_corrector.encode_payload(signature)
        elif type(config.error_corrector) == ConcatenatedViterbiRS:
            coded_signature, rs_coded_signature = config.error_corrector.encode_payload(signature)
            file = open(f"{config.session_output_path}/embedded_data/previterbi_{feature_seq_num}.txt", "w") #here, previterbi is RS-encoded
            file.write(rs_coded_signature)
            file.close()
        elif type(config.error_corrector) == SoftViterbi:
            file = open(f"{config.session_output_path}/embedded_data/previterbi_{feature_seq_num}.txt", "w") #here, previterbi is the same as payload
            file.write(signature)
            file.close()
            coded_signature = config.error_corrector.encode_payload(signature) 

        # pad with 0 bits up to num_info_cells * target_freq * window duration
        padded_coded_signature = pad_bitstring(coded_signature, config.max_bits)
        
        # done constructing payload. save it at payloads path for it to now be ~optically embedded~
        file = open(f"{config.session_output_path}/embedded_data/final_{feature_seq_num}.txt", "w")
        file.write(padded_coded_signature)
        file.close()


def run_extraction(dump_feats = False):
    """
    Main function to run feature extraction and serialization process.
    Extracts features from video frames, creates digests, and serializes them into payloads

    Args:
        dump_feats (bool): If True, dumps intermediate features to files for debugging purposes.
    """
    #initialize MPExtractor and IdentityExtractor class
    mp_extractor = MPExtractor()
    id_extractor = IdentityExtractor()

    #start up serialization thread
    kill_lock= threading.Lock()
    kill_cv = threading.Condition(kill_lock)
    feature_queue = Queue()
    serialization_thread = threading.Thread(target=serialize, args=(kill_cv, feature_queue, ))
    serialization_thread.start()

    while not os.path.isfile(f"{config.session_output_path}/process_synchronization/go.txt"):
        time.sleep(0.1) # wait for the process to start
        pass

    time.sleep(0.5) # small delay to make sure the prcoess is ready for good
    
    try:
        img_files = glob.glob( f"{config.session_output_path}/imgs/*")
        latest_img_file = max(img_files, key=os.path.getctime)
        latest_img_file_num = int(latest_img_file.split("/")[-1].split(".npy")[0])
        img_num = latest_img_file_num

        curr_window_dynamic_features = []
        curr_window_num_frames = 0
        time.sleep(0.1)  # give a bit of down time for the images to be fully read to read
        frame = np.load(f"{config.session_output_path}/imgs/{img_num}.npy")
        first_win_img_save_time = None
        last_window_send = time.time()
        last_extracted_seq = -1
        first_win_img_num = 0
        last_win_img_num = 0
        approx_num_frames_per_window = int(config.video_window_duration * (config.target_fps - 5)) # minus a few fps to be safe
        id_frame_num = random.randint(0, approx_num_frames_per_window - 1) # randomly select a frame in the window to use for ID extraction
        overall_start = time.time()
        id_extracted = False
        curr_id_features = [0 for i in range(512)] 
        while True:
            #load img from npy array
            while True:
                try:
                    frame = np.load(f"{config.session_output_path}/imgs/{img_num}.npy")
                    img_save_time = os.path.getctime(f"{config.session_output_path}/imgs/{img_num}.npy")
                    break
                except Exception as e:
                    pass
            
            if curr_window_num_frames == 0:
                first_win_img_save_time = img_save_time   
                first_win_img_num = img_num  
            
            # get ID features only once per every other window using randomly selected frame
            if not id_extracted and (curr_window_num_frames >= id_frame_num): # in case the ID feature couldn't be extracted, try again next window
                if (last_extracted_seq + 1) % 2 == 0:
                    # run one-time identity feature extraction on frame
                    identity_features = id_extractor.extract(frame)
                    if identity_features is not None:
                        id_extracted = True
                        print(Fore.BLUE + f"Got Seq {last_extracted_seq + 1} ID features." + Style.RESET_ALL)
                        curr_id_features = identity_features
                    else:
                        identity_features = [0 for i in range(512)] 
                        # print(Fore.YELLOW + f"Couldn't get Seq {last_extracted_seq + 1} ID features." + Style.RESET_ALL)
                else:
                    # print(Fore.BLUE +f"Reusing Seq {last_extracted_seq} ID features." + Style.RESET_ALL)
                    identity_features = curr_id_features # use from last sequence 
        
            #run MP extraction on frame
            start = time.time()
            frame_feats, face_bbox, detection_result = mp_extractor.extract_features(frame)
            fps =  1/ (time.time() - start)
            #print(f"Img {img_num - latest_img_file_num + 1}. Extraction FPS: {fps}")
            overall_fps =  (img_num - latest_img_file_num + 1)/ (time.time() - overall_start)
            #print(f"Img {img_num - latest_img_file_num + 1}. Overall FPS: {overall_fps}")
            curr_window_dynamic_features.append(frame_feats)
            curr_window_num_frames += 1
   
            if dump_feats:
                with open(f"{config.session_output_path}/features/mp_img{img_num}.pkl", "wb") as pklfile:
                    pickle.dump(face_bbox, pklfile)
                    pickle.dump(detection_result, pklfile)
            if (img_save_time - first_win_img_save_time) >= config.video_window_duration: #a full window's worth of data has been processed
                last_win_img_save_time = img_save_time
                last_win_img_num = img_num
                effective_fps = len(curr_window_dynamic_features) / (last_win_img_save_time - first_win_img_save_time) 
                ready = time.time()
                print(Fore.BLUE + f"Window {last_extracted_seq} features ready")
                # print(f"{ready - last_window_send} sec elapsed since last send\n{last_win_img_save_time - first_win_img_save_time} sec of video in this window. Last frame of win at {datetime.fromtimestamp(last_win_img_save_time)}. Win video FPS: {effective_fps}")
                last_window_send = ready
                #give serialization thread features to hash, serialize, etc.
                feature_queue.put((last_extracted_seq, curr_window_dynamic_features, identity_features, [first_win_img_num, last_win_img_num]))
                last_extracted_seq += 1
                # empty/reset to start accruing for next frame
                curr_window_num_frames = 0
                curr_window_dynamic_features = []
                id_extracted = False
            img_num += 1

    except KeyboardInterrupt:
        pass
    except BaseException as e:
        print("Error!", e)
        pass
    
    #stop threads
    with kill_cv:
        got_kill_signal = True
        kill_cv.notify_all()
    feature_queue.put(None) #required to avoid deadlock in join
    serialization_thread.join()
   
run_extraction(dump_feats = True)
    
