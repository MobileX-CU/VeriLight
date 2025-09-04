"""
Calibration for planar projection via SLM

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
import numpy as np
import time
import pickle
import threading
import statistics
from queue import Queue
from datetime import datetime, timezone
from paramiko import SSHClient
from scp import SCPClient
import sys

from embedding.calibration_client_thread import ClientThread
from embedding.embedding_utils import prepare_dirs, save_img_seq, adaptive_log,  confirm_cam_settings, get_cam_id, parse_last_disp_info, create_slm_config
from embedding.calibration_utils import create_calibration_code, generate_calibration_reference_points, get_user_points, order_calibration_code_corners, detect_heatmap_cells
 
import common.config as config
from common.decoding_utils import get_homography
from verification.create_heatmap import create_calibration_heatmap 

use_curr_hom = True
seq_inc = 0

got_kill_signal = False
image_timestamps = []
fpss = []

def frame_save(cv, frame_queue):
    """
    Save img frames passed from the cam_capture thread through the frame queue as npy files

    Args:
        cv (threading.Condition): Condition variable to handle kill signal.
        frame_queue (Queue): Queue containing frames to be saved.
    """
    num_saved = 0
    while True:
        with cv:
            cv.wait(0.00001)
            interrupted = got_kill_signal  # Read got_kill_signal while holding the lock
        if interrupted:
            return
        frame, i = frame_queue.get()

        if config.downsample_frames != 0:
            frame = cv2.resize(frame, (int(frame.shape[1] * config.downsample_frames), int(frame.shape[0] * config.downsample_frames)), interpolation=cv2.INTER_AREA)
    
        np.save("{}/{}.npy".format( f"{config.session_output_path}/imgs", i - 1), frame)
        num_saved += 1
        if i == 1:
            start = time.time()
        else:
            end = time.time()
            effective_fps = num_saved/(end -  start)
            global fpss
            fpss.append(effective_fps)


def cam_capture(cv, camera_name, frame_queue):
    """
    Continuoulsly record and associate each frame with a timestamp.
    Put all captured frames in the frame queue to be saved by the frame_save thread

    Args:
        cv (threading.Condition): Condition variable to handle kill signal.
        camera_name (str): Name of the camera to use.
        frame_queue (Queue): Queue to put captured frames into.
    """
    cam_id = get_cam_id(camera_name)
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return
  
    start = time.time()
    num_frames = 0
    while True:
        with cv:
            cv.wait(0.00001)
            interrupted = got_kill_signal  # Read got_kill_signal while holding the lock
        if interrupted:
            cap.release()
            return
        ret, frame = cap.read()
        cap_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f") #match UTC timezone of SLM
        end = time.time()
        if not ret:
            break
        num_frames += 1
        frame_queue.put([frame, num_frames])
        effective_fps = num_frames/(end -  start)
        global image_timestamps
        image_timestamps.append(cap_time)
      

def get_imgseq(start_image_timestamp_i, end_image_timestamp_i):
    """
    Load image sequence from saved images between start and end indices.

    Args:
        start_image_timestamp_i (int): Index of the first image to load.
        end_image_timestamp_i (int): Index of the last image to load.
    
    Returns:
        numpy array: Array of loaded images.
    """
    adaptive_log(f"Start and end image timestamps of sequence: {start_image_timestamp_i} {end_image_timestamp_i}", "DEBUG")
    img_seq = []
    for i in range(start_image_timestamp_i, end_image_timestamp_i + 1): #+3 for good measure
        img = cv2.imread(f"{config.session_output_path}/imgs/hom_{i}.png")
        if config.analysis_colorspace == "ycrcb":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        img_seq.append(img)
    img_seq = np.array(img_seq)
    return img_seq


def get_imgseq_info(start_time, end_time):
    """
    Determined indices of first image (start_image_timestamp_i) and last image (end_image_timestamp_i)
    in the sequence. Also return the effective FPS of the core unit camera during this sequence

    Args:
        start_time (str): Start time of the sequence in "%Y-%m-%d %H:%M:%S.%f" format.
        end_time (str): End time of the sequence in "%Y-%m-%d %H:%M:%S.%f" format.
    
    Returns:
        tuple: (fps (float), start_image_timestamp_i (int), end_image_timestamp_i (int))
    """
    start_time_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f")
    start_image_timestamp_i = image_timestamps.index(min(image_timestamps, key=lambda b: abs(start_time_dt-datetime.strptime(b, "%Y-%m-%d %H:%M:%S.%f")))) - 1
    end_time_dt = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f")
    end_image_timestamp_i = image_timestamps.index(min(image_timestamps, key=lambda b: abs(end_time_dt-datetime.strptime(b, "%Y-%m-%d %H:%M:%S.%f")))) + 5
    fps = statistics.mean(fpss[start_image_timestamp_i:end_image_timestamp_i + 1])
    return fps, start_image_timestamp_i, end_image_timestamp_i

try:
    prepare_dirs()
    
    # create config on Pi
    create_slm_config()
    ssh = SSHClient() 
    ssh.load_system_host_keys()
    ssh.connect(config.host_ip, username=config.host_username)
    with SCPClient(ssh.get_transport()) as scp:
        scp.put(f"{config.session_output_path}/config_slm.py", "config.py")
    ssh.close()

    #######################
    # START UP THREADS
    ########################
    #start camera capture thread 
    kill_lock= threading.Lock()
    kill_cv = threading.Condition(kill_lock)
    camera_name = "Arducam" 
    if "Arducam" in camera_name:
        confirm_cam_settings(camera_name)

    adaptive_log("Starting camera capture thread", "INFO")
    frame_queue = Queue()
    cam_save_thread = threading.Thread(target=frame_save, args=(kill_cv, frame_queue, ))
    cam_save_thread.start()
    cam_thread = threading.Thread(target=cam_capture, args=(kill_cv, camera_name, frame_queue, ))
    cam_thread.start()

   
    while len(fpss) == 0:  #wait till camera capture actually begins before proceeding
        pass
    
    adaptive_log("Starting connection with SLM", "INFO")
    #start up connection with SLM via client thread
    req_queue = Queue() #either request transfer of bmps or seq info
    seq_info_queue = Queue()
    exc_queue = Queue()
    client_thread = ClientThread(req_queue, seq_info_queue, exc_queue, args=(config.port, config.host_ip))
    client_thread.start()
    adaptive_log("Established connection with SLM", "INFO")

    #start up adaptation thread
    kill_lock= threading.Lock()
    kill_cv = threading.Condition(kill_lock)
    
    print("Preparing directories...")
    prepare_dirs()
    print("Directories prepared.")

    #######################
    # CALIBRATION #
    ########################

    #create and project initial calibration code (just bright corners)
    calibration_dir = f"{config.session_output_path}/bmps/seq0"
    create_calibration_code(config.N, (0, 255, 255), calibration_dir)
    client_thread.req_queue.put((calibration_dir, True))

    print("Waiting for calibration code to display...")

    #wait for calibration code display start and end times
    last_disp_seq = -1
    while last_disp_seq < 0:
        client_thread.req_queue.put("REQSEQ")
        sync_message = seq_info_queue.get()
        last_disp_seq, last_seq_start_time, last_seq_end_time, last_disp_timestamps = parse_last_disp_info(sync_message)
        time.sleep(0.5)

    fps, start_image_timestamp_i, end_image_timestamp_i = get_imgseq_info(last_seq_start_time, last_seq_end_time)

    save_img_seq(start_image_timestamp_i, end_image_timestamp_i, f"{config.session_output_path}/calibration", fps)

    #get calibration code heatmap
    heatmap_chan1, heatmap_chan2, heatmap_chan3 = create_calibration_heatmap(f"{config.session_output_path}/imgs", start_image_timestamp_i, end_image_timestamp_i, fps, 
                               config.calibration_frequency, config.calibration_lower_range_start, config.calibration_lower_range_end,
                               config.calibration_upper_range_start, config.calibration_upper_range_end, 
                               config.calibration_target_lower_epsilon, config.calibration_target_upper_epsilon, 
                               config.target_channel, config.colorspace, 
                               denoise = True, display = False,
                                output_folder = config.session_output_path, output_name = "calibration")
    
    if config.target_channel == 0:
        heatmap = heatmap_chan1
    elif config.target_channel == 1:
        heatmap = heatmap_chan2
    elif config.target_channel == 2:
        heatmap = heatmap_chan3
    elif config.target_channel == "sum":
        heatmap = heatmap_chan1
    else:
        adaptive_log("Invalid target channel", "ERROR")
        sys.exit()

    #detect corners of calibration code and get manual verification since this is an important and one-time homography
    corner_centers, _ = detect_heatmap_cells(heatmap, erode=1, blurthensharp = True, otsu_inc = 8, display=True)

    #get source corners to use for homography
    inferred_corner_stuff = order_calibration_code_corners(corner_centers, heatmap, slope_epsilon=0.1, display=False)
    if inferred_corner_stuff is not None:
        sorted_corner_centers, labeled_heatmap = inferred_corner_stuff
        cv2.imshow("Inferred calibration corners", labeled_heatmap)
        cv2.waitKey(0)
        accept_inferred_corners = input("Are the inferred calibration corners ok? y/n")
    if inferred_corner_stuff is None or accept_inferred_corners == "n":
        vis_corners = heatmap.copy()
        vis_corners = cv2.cvtColor(vis_corners, cv2.COLOR_GRAY2BGR)
        for c in corner_centers:
            cv2.circle(vis_corners, c, 2, (0, 0, 255), -1)

        sorted_corner_centers = get_user_points(vis_corners) #corners MUST be in order topleft, topright, bottom left, bottom right
        if len(sorted_corner_centers) == 0:
            sys.exit()

    #perform calibration homography
    reference_img, reference_corner_centers = generate_calibration_reference_points(config.slm_W, config.slm_H, config.N)
    Hom = get_homography(sorted_corner_centers, reference_corner_centers, heatmap, reference_img, display=True)

    #store homography results for future runs
    with open("curr_hom.pkl", "wb") as file:
        pickle.dump(Hom, file)

except KeyboardInterrupt:
    pass #fall down to kill/cleanup below
    
with kill_cv:
    got_kill_signal = True
    kill_cv.notify_all()
client_thread.req_queue.put(None) #required to avoid deadlock in join
client_thread.join()
cam_thread.join()
cam_save_thread.join()

