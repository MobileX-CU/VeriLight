"""
Start and coordinate creation of signature and embedding into scene via adaptive encoding.
Note that signature generation process (signature_generation.py) has to be started in a 
separate terminal.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import cv2
import os
import threading
import multiprocessing
import time
import pickle
from datetime import datetime, timezone
import numpy as np
import threading
from queue import Queue, Empty
from colorama import Fore, Back
from paramiko import SSHClient
from scp import SCPClient

import common.config as config

from embedding.embedding_utils import stash_trial_data, prepare_dirs, write_config_summary, get_cam_id, parse_last_disp_info, confirm_cam_settings, adaptive_log, create_slm_config
from embedding.client_thread import ClientThread
from embedding.adaptation_process import adapt
from embedding.psk_encode_minimal import create_minimal_frames

got_kill_signal = False
global_image_timestamps = []
global_fpss = []


def frame_save(cv, frame_queue):
    """
    Save img frames passed from the cam_capture thread through the frame queue as npy files

    Args:
        cv (threading.Condition): Condition variable to check for kill signal.
        frame_queue (queue.Queue): Queue to receive frames from the cam_capture thread.
    """
    start = None
    num_saved = 0
    while True:
        with cv:
            cv.wait(0.00001)
            global got_kill_signal
            interrupted = got_kill_signal  # Read got_kill_signal while holding the lock
        if interrupted:
            return
        frame, i = frame_queue.get()
        
        if config.downsample_frames != 0:
            frame = cv2.resize(frame, (int(frame.shape[1] * config.downsample_frames), int(frame.shape[0] * config.downsample_frames)), interpolation=cv2.INTER_AREA)
    
        start_save = time.time()
        np.save("{}/{}.npy".format(f"{config.session_output_path}/imgs", num_saved), frame)
        end_save = time.time()
        num_saved += 1
        if i == 1:
            start = time.time()
        else:
            end = time.time()
            effective_fps = num_saved/(end -  start)
            global global_fpss
            global_fpss.append(effective_fps)
            if num_saved % 300 == 0:
                adaptive_log(f"Current effective FPS: {effective_fps}. Typical .npy save time: {end_save - start_save}.", "DEBUG", flags = [Fore.BLUE])

def cam_capture(cv, camera_name, frame_queue):
    """
    Continuoulsly record and associate each frame with a timestamp.
    Put all captured frames in the frame queue to be saved by the frame_save thread

    Args:
        cv (threading.Condition): Condition variable to check for kill signal.
        camera_name (str): Name of the camera to use for capture.
        frame_queue (queue.Queue): Queue to send captured frames to the frame_save thread.
    """
    cam_id = get_cam_id(camera_name)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return
    num_frames = 0
    last_cap = time.time()
    while True:
        with cv:
            cv.wait(0.00001)
            interrupted = got_kill_signal  # Read got_kill_signal while holding the lock
        if interrupted:
            cap.release()
            return
        
        now = time.time()
        if now - last_cap < 1/config.target_fps:
            time.sleep(1/config.target_fps - (now - last_cap))
        last_cap = time.time()
        ret, frame = cap.read()
        
        cap_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f") #match UTC timezone of SLM
        if not ret:
            break
        num_frames += 1
        
        frame_queue.put([frame, num_frames])
        global global_image_timestamps
        global_image_timestamps.append(cap_time)

def main():
    # save things and get readddyyy
    stash_trial_data("stash")
    prepare_dirs()
    write_config_summary()

    # create config on Pi
    create_slm_config()
    while not os.path.exists(f"{config.session_output_path}/config_slm.py"):
        time.sleep(0.1)
    print("Config file created on Pi. Transferring to Pi...")
    ssh = SSHClient() 
    ssh.load_system_host_keys()
    ssh.connect(config.host_ip, username=config.host_username)
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(f"{config.session_output_path}/config_slm.py", "config.py")
    ssh.close()
    print("Transffered config.")

   
    f_ber = open(f"{config.session_output_path}/ber_logging.csv", "w")
    f_ber.write("Seq,Hard Decoding Errors,Soft Decoding Errors\n")

    #######################
    # START UP THREADS/PROCESSES
    ########################
    try:
        # load homography
        with open("curr_hom.pkl", "rb") as file:
            Hom = pickle.load(file)

        #start camera capture thread 
        kill_lock= threading.Lock()
        kill_cv = threading.Condition(kill_lock)
        camera_name = "Arducam" 
        if "Arducam" in camera_name or "UC70" in camera_name:
            confirm_cam_settings(camera_name) 
        adaptive_log("Starting camera capture thread", "INFO")
        frame_queue = Queue()
        cam_save_thread = threading.Thread(target=frame_save, args=(kill_cv, frame_queue, ))
        cam_save_thread.start()
        cam_thread = threading.Thread(target=cam_capture, args=(kill_cv, camera_name, frame_queue, ))
        cam_thread.start()
        while len(global_fpss) == 0: #wait until camera capture actually begins before proceeding
            pass
    
        # start up client theread and connect to SLM
        adaptive_log("Starting connection with SLM", "INFO")
        req_queue = Queue() #queue to pass requests of either bmp transfer to SLM or sequence info from SLM to the client thread 
        seq_info_queue = Queue()
        exc_queue = Queue()
        client_thread = ClientThread(req_queue, seq_info_queue, exc_queue, args=(config.port, config.host_ip))
        client_thread.start()
        adaptive_log("Established connection with SLM", "INFO")

        if config.ADAPT:
            #start up adaptation process
            ad_queue = multiprocessing.Queue() 
            cell_colors_mins_queue = multiprocessing.Queue() 
            ready_adapt_queue = multiprocessing.Queue() 
            adaptation_thread = multiprocessing.Process(target=adapt, args = (ad_queue, cell_colors_mins_queue, ready_adapt_queue, ))
            adaptation_thread.start()

        # cue the feature extraction module!
        time.sleep(2) # give some time for config to finish importing, operator to be ready
        while True:
            raw = input("Signature generation script running and ready to go? (y/n):")
            if raw == "y":
                fps_file = open(f"{config.session_output_path}/process_synchronization/go.txt", "w")
                fps_file.write("hi :)")
                fps_file.close()
                break
  
        #######################
        # ACTION!
        ########################  
        # initliaze some state variables
        last_disp_seq_num = -1 #last calibration code sequence displayd by SLM
        last_queued_seq_num = -1 #last sequence that was added to the adaptation queue
        last_extracted_seq = -1 #last sequence to have features extracted
        final_payload_bitlists_by_seq = {} #dictionary format to allow easier deletion
        previterbi_payloads_by_seq = {}
        curr_cell_colors = [config.init_slm_bgr for x in range(config.max_cells)] #bgr order
        curr_cell_min_tot_slms = [config.init_min_tot_slm for i in range(config.max_cells)]
        if config.localization_N:
            curr_loc_marker_colors = [config.init_slm_bgr for x in range(4)]
            curr_loc_marker_min_tot_slms = [config.init_min_tot_slm for i in range(4)]
        else:
            curr_loc_marker_colors = None
            curr_loc_marker_min_tot_slms = None
    
        while True:
            if config.ADAPT:
                #check for new cell color recommendations from adaptation proc
                try:
                    new_cell_colors,  new_loc_marker_colors, new_cell_min_tot_slms, new_loc_marker_min_tot_slms,  soft_decoding_ber, normal_decoding_ber, last_scored_seq_num = cell_colors_mins_queue.get(block=False, timeout = 0.0001) 
                    adaptive_log(f"Del seq{last_scored_seq_num}", "DEBUG")
                    curr_cell_colors = new_cell_colors #update
                    curr_loc_marker_colors = new_loc_marker_colors
                    curr_cell_min_tot_slms = new_cell_min_tot_slms
                    curr_loc_marker_min_tot_slms = new_loc_marker_min_tot_slms
                    f_ber.write(f"{last_scored_seq_num},{normal_decoding_ber}/{len(final_payload_bitlists_by_seq[f'seq{last_scored_seq_num}'])},{soft_decoding_ber}/{len(previterbi_payloads_by_seq[f'seq{last_scored_seq_num}'])}\n")
                    del final_payload_bitlists_by_seq[f"seq{last_scored_seq_num}"] # no longer needed, save some space by removing from dictionary
                    del previterbi_payloads_by_seq[f"seq{last_scored_seq_num}"]
                except Empty:
                        pass

            #check for serialized features from feature extraction proc
            if os.path.isfile(f"{config.session_output_path}/embedded_data/final_{last_extracted_seq + 1}.txt"):
                last_extracted_seq += 1
                adaptive_log(f"Extracted features for seq {last_extracted_seq} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')}", "DEBUG", flags = [Fore.BLUE])
                
                # extract payload data
                time.sleep(0.1) #give a little time for the file to be fully written
                final_payload_file = open(f"{config.session_output_path}/embedded_data/final_{last_extracted_seq}.txt", "r")
                final_payload = final_payload_file.read() # the final payload (bits to actually transmit)
                previterbi = open(f"{config.session_output_path}/embedded_data/previterbi_{last_extracted_seq}.txt", "r").read() #the RS encoded data (prior to Viterbi encoding)
                previterbi_payloads_by_seq[f"seq{last_extracted_seq}"] = previterbi

                # create minimal bitmaps
                barcode_dir = f"{config.session_output_path}/bmps/seq{last_extracted_seq}"
                info_cell_bit_list = create_minimal_frames(final_payload, curr_cell_colors, curr_loc_marker_colors, barcode_dir)
                final_payload_bitlists_by_seq[f"seq{last_extracted_seq}"] = info_cell_bit_list

                # send bitmap dir to client thread to request send to/display on SLM
                adaptive_log(f"Sending bitmaps for window {last_extracted_seq} at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S.%f')}", "INFO", flags = [Fore.BLUE])
                client_thread.req_queue.put((barcode_dir, False))
                
                #quickly check for exceptions in client thread before proceeding
                try:
                    exc = exc_queue.get(block=False, timeout=0.05)
                except Empty:
                    pass
                else:
                    adaptive_log("Exception in client thread: {}.".format(exc), "ERROR")
                    break
            

            #when get message that has valid last display start/end, fetch the associate framestamps
            try:
                sync_message = seq_info_queue.get(block=False, timeout=0.01)
                last_disp_seq_num, last_disp_start_time, last_disp_end_time, _ = parse_last_disp_info(sync_message)

                adaptive_log(f"Received notification of {last_disp_seq_num} display finishing", "DEBUG")

                if config.ADAPT:
                    if last_disp_seq_num > last_queued_seq_num:
                        #use this img sequence to perform an adaption
                        ready_adapt_queue.get(block=False, timeout=0.00001) #only put something in if adaptation thread wants it
                        adaptive_log("Ready to adaptive process seq {}".format(last_disp_seq_num), "DEBUG")
                        adapt_params_input = [last_disp_start_time , last_disp_end_time, last_disp_seq_num, Hom, final_payload_bitlists_by_seq[f"seq{last_disp_seq_num}"], previterbi_payloads_by_seq[f"seq{last_disp_seq_num}"], global_image_timestamps.copy(), global_fpss.copy(), curr_cell_min_tot_slms, curr_loc_marker_min_tot_slms]
                        ad_queue.put(adapt_params_input)
                        last_queued_seq_num = last_disp_seq_num
            except Empty:
                pass    
    except KeyboardInterrupt:
        pass #fall down to kill/cleanup below
        
    #######################
    # CLEANUP
    ########################
    with kill_cv:
        global got_kill_signal
        got_kill_signal = True
        kill_cv.notify_all()
    client_thread.req_queue.put(None) #required to avoid deadlock in join
    client_thread.join()
    cam_thread.join()
    cam_save_thread.join()
    f_ber.close()
  

if __name__ == '__main__':
    main() 