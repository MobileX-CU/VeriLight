"""
Server running on Raspberry Pi to receive frames to display from the client (PC)
and display them via SLM

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""

import socket
from multiprocessing import Process, Queue, shared_memory
from queue import Empty
import time
import numpy as np
import subprocess
import os
import cv2
import pickle
import sys
from datetime import datetime
from colorama import Fore, Back, Style

PRINT_TIMING = True

def display(disp_signal_queue, disp_times_queue):
    """
    Display manager process to handle displaying frames on the screen.

    Args:
        disp_signal_queue (multiprocessing.Queue): Queue to receive display signals.
        disp_times_queue (multiprocessing.Queue): Queue to send back display times.
    """
    window_name = "Frame"
    create_window(window_name)

    # initalize shared memories
    existing_shm1 = shared_memory.SharedMemory(name = "frames1")
    existing_shm2 = shared_memory.SharedMemory(name = "frames2")

    #"warm up" display. 
    black_frame = cv2.imread("black.bmp")
    cv2.imshow(window_name, black_frame) 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        return

    sleep_time = None
    last_seq_end = time.time()
    target_inter_win_time = 0.5
    while True:
        seq_num, num_frames, freq, rep = disp_signal_queue.get()
        signal_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        if seq_num % 2 == 0:    
            frames = np.ndarray((NUM_FRAMES, 360, 640, 3), dtype = np.float32, buffer = existing_shm1.buf)
        else:
            frames = np.ndarray((NUM_FRAMES, 360, 640, 3), dtype = np.float32, buffer = existing_shm2.buf)
        if PRINT_TIMING:
            print(Fore.BLUE + f"Got signal for seq {seq_num} at {signal_time}. Got all the frames at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}" + Style.RESET_ALL)
    
        if sleep_time is None: #only set this once
            sleep_time = get_sleep_time(freq)
            if sleep_time is None:
                print(Fore.YELLOW + Back.RED + "No sleep time set for freq: ", freq, ". Exiting..." + Style.RESET_ALL)
                sys.exit()
            print(f"Sleep time: {sleep_time} for freq: {freq} Hz. Num frames: {len(frames)}. Reps: {rep}" )
        
        disp_times = ""
        # temp = []
        for i in range(num_frames * rep):
            frame = frames[i % num_frames]
            # temp.append(frame.copy())

            if i == 0:
                curr_time = time.time()
                if curr_time - last_seq_end < target_inter_win_time:
                    time.sleep(target_inter_win_time - (curr_time - last_seq_end))
                start_disp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            cv2.imshow(window_name, frame) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

            disp_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            disp_time_s = time.time()
            disp_times = disp_times + disp_time + ";"
            
            if i == num_frames * rep - 1: #last frame in this sequence
                end_disp  = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                last_seq_end =  disp_time_s
                print(Fore.GREEN + f"Done displaying seq {seq_num}. Start: {start_disp}. End: {end_disp}." + Style.RESET_ALL)
                disp_times_queue.put((seq_num, start_disp, end_disp, disp_times)) #report display times back to main thread
                cv2.imshow(window_name, black_frame)  #reset screent to black in between windows
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break
            
                # for q, hi in enumerate(temp):
                #     cv2.imwrite(f"seq{seq_num}_frame_{q}.png", hi*255)

            time.sleep(sleep_time)
      

def parse_metadata(metadata_str):
    """
    Parse metadata string received from client.
    
    Args:
        metadata_str (str): Metadata string in the format "NUMFRAMES: X. FRAMEFILESIZE: Y. REPEAT: Z. FREQUENCY: W."

    Returns:
        tuple: (num_frames (int), frame_size (int), rep (int), freq (float))
    """
    num_frames = int(metadata_str.split("NUMFRAMES: ")[1].split(".")[0])
    file_size = int(metadata_str.split("FRAMEFILESIZE: ")[1].split(".")[0])
    rep = int(metadata_str.split("REPEAT: ")[1].split(".")[0])
    freq = float(metadata_str.split("FREQUENCY: ")[1].strip("."))
    return num_frames, file_size, rep, freq


def xinit():
    os.system("sudo xinit -- -nocursor")


if __name__ == '__main__':
    # delete config, then import
    os.system("rm config.py")
    print("Waiting for config file...")
    while not os.path.exists("config.py"):
        time.sleep(0.1)
    time.sleep(1)
    from pi_display_utils import create_window, get_sleep_time
    from psk_encode_minimal import create_mask, create_full_frame # don't do this until here because we need to wait until updated config is here
    import config

    # start/configure display
    os.environ["DISPLAY"] = ":0"
    xinit_process = Process(target=xinit)
    xinit_process.start()

    #block until xinit display is ready
    display_ready = False
    while not display_ready:
        res = subprocess.run(f"ps aux | grep xinit",  stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True).stdout
        if "sudo xinit" in res:
            display_ready = True
    print("X server started.")

    # connect to client
    print("Setting up socket.")
    port = 6000
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", port)) 
    sock.listen(1) 
    print("Server ready to accept connections")
    conn, addr = sock.accept() 

    # create mask for full frame creation
    mask, rightmost, bottommost = create_mask(fade = config.fade)

    NUM_FRAMES = config.embedding_window_duration * config.localization_frequency * 2
   
    # start display thread
    disp_signal_queue = Queue()
    disp_times_queue = Queue()
    display_manager_thread = Process(target=display, args=(disp_signal_queue, disp_times_queue, ))
    display_manager_thread.start()

    # create two shared memories for frames. alternate between using each.
    dummy_frames = np.zeros((NUM_FRAMES, 360, 640, 3), dtype=np.float32)
    shm1 = shared_memory.SharedMemory(name = "frames1", create = True, size = dummy_frames.nbytes)
    shm2 = shared_memory.SharedMemory(name = "frames2", create = True, size = dummy_frames.nbytes)

    # initialize state/tracking variables
    last_recv_seq = 0
    last_disp_seq = -1
    last_disp_start = None
    last_disp_end = None
    last_disp_times = None

    try:
        while True:
            try:
                seq_num, start_disp, end_disp, disp_times = disp_times_queue.get(block=False, timeout=0.01)
                last_disp_seq = seq_num #update last_disp info
                last_disp_start = start_disp
                last_disp_end = end_disp
                last_disp_times = disp_times
            except Empty:
                pass

            # tell client we are ready to receive requests
            conn.send("READY RECEIVE REQUESTS".encode())

            #receive a request
            req = conn.recv(1024)
            if req is None: break
            
            req = req.decode()
            if req == "REQSEQ":
                conn.sendall("Last displayed sequence: {}. Start: {}. End: {}. Timestamps: {}. READY FOR METADATA."
                            .format(last_disp_seq, last_disp_start, last_disp_end, last_disp_times).encode())
            else:
                #receive display metadata
                metadata_str = req
                num_frames, frame_size, rep, freq = parse_metadata(metadata_str)
                print("Received metadata. Num frames: {}. Frame size: {}. Reps: {}. Freq: {}".format(num_frames, frame_size, rep, freq))
                
                #respond to metadata with latest display info 
                msg = "Last displayed sequence: {}. Start: {}. End: {}. Timestamps: {}. READY FOR METADATA.".format(last_disp_seq, last_disp_start, last_disp_end, last_disp_times).encode()
                conn.sendall(len(msg).to_bytes(20, byteorder='big') + msg)

                start = time.time()
                tot_num_frame_bytes = frame_size * num_frames
                all_frame_bytes = b''
                while len(all_frame_bytes) < tot_num_frame_bytes:
                    recv_bytes = conn.recv(20000)
                    all_frame_bytes += recv_bytes
                if PRINT_TIMING:
                    print(Fore.BLUE + f"Received all {len(all_frame_bytes)}/{tot_num_frame_bytes} seq {last_recv_seq} bytes in {time.time() - start} s. at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"+ Style.RESET_ALL)

                # iterate through, segmenting into individual frames. for each, resize, mask, and add to shared memory array
                if last_recv_seq % 2 == 0:
                    frames = np.ndarray(dummy_frames.shape, dtype=dummy_frames.dtype, buffer = shm1.buf)
                else:
                    frames = np.ndarray(dummy_frames.shape, dtype=dummy_frames.dtype, buffer = shm2.buf)
                start = time.time()
                frame_num = 0
                for i in range(0, tot_num_frame_bytes, frame_size):
                    minimal_frame_bytes = all_frame_bytes[i:i+frame_size]
                    minimal_frame = pickle.loads(minimal_frame_bytes)
                    frames[frame_num,:,:,:] = create_full_frame(minimal_frame, mask, rightmost, bottommost)
                    frame_num += 1
                    
                if PRINT_TIMING:
                    print(Fore.BLUE + f"Created all seq {last_recv_seq} frames in {time.time() - start} s. Signaling at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}." + Style.RESET_ALL)
                disp_signal_queue.put((last_recv_seq, num_frames, freq, rep)) # put signal in queue to tell display to read frames
                last_recv_seq += 1

    except ConnectionResetError:
        conn.close()
        xinit_process.join()
        shm1.close()
        shm1.unlink()
        shm2.close()
        shm2.unlink()


