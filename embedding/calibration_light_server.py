"""
Light server used for calibration, run on Pi.
Similar to light_server.py but slightly different values for sleep, given calibration doesn't have as much going on 
in parent thread to enable faster display, in support of higher calibration localization frequencies.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import socket
import threading
from multiprocessing import Process
from queue import Queue, Empty
import time
import sys
import os
import cv2
import glob
from datetime import datetime

def display(disp_queue, disp_times_queue):
    """
    Display frames received from the display queue.

    Args:
        disp_queue (Queue): Queue containing frames to be displayed along with sequence number, frequency, and repetitions.
        disp_times_queue (Queue): Queue to put display timing information into.
    """
    window_name = "Frame"
    create_window(window_name)

    #"warm up" display. 
    black_frame = cv2.imread("black.bmp")
    cv2.imshow(window_name, black_frame) 
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        return

    sleep_time = None
    last_seq_end = time.time()
    target_inter_win_time = 0.5
    while True:
        frames, seq_num, freq, rep = disp_queue.get()

        if sleep_time is None: #only set this once
            sleep_time = get_sleep_time(freq)
            if sleep_time is None:
                print("No sleep time set for freq: ", freq, ". Exiting...")
                sys.exit()
            print(f"Sleep time: {sleep_time} for freq: {freq} Hz. Num frames: {len(frames)}. Reps: {rep}")
        
        disp_times = ""
        for i in range(num_frames * rep):
            frame = frames[i % num_frames]

            if i == 0:
                curr_time = time.time()
                if curr_time - last_seq_end < target_inter_win_time:
                    time.sleep(target_inter_win_time - (curr_time - last_seq_end))
                start_disp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            
            # start_disp_s = time.time()
            cv2.imshow(window_name, frame) 
            if cv2.waitKey(25) & 0xFF == ord('q'): 
                break

            disp_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            disp_time_s = time.time()
            disp_times = disp_times + disp_time + ";"
            time.sleep(sleep_time)
        
            if i == num_frames * rep - 1: #last frame in this sequence
                end_disp  = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                last_seq_end =  disp_time_s
                print(f"Done displaying seq {seq_num}. Start: {start_disp}")
                disp_times_queue.put((seq_num, start_disp, end_disp, disp_times))
                cv2.imshow(window_name, black_frame) 
                if cv2.waitKey(25) & 0xFF == ord('q'): 
                    break


def parse_metadata(metadata_str):
    """
    Parse metadata string received from the client.

    Args:
        metadata_str (str): Metadata string in the format "NUMFRAMES: <num_frames>. FRAMEFILESIZE: <file_size>. REPEAT: <repeat>. FREQUENCY: <frequency>".
    
    Returns:
        tuple: (num_frames (int), file_size (int), rep (int), freq (float))
    """
    num_frames = int(metadata_str.split("NUMFRAMES: ")[1].split(".")[0])
    file_size = int(metadata_str.split("FRAMEFILESIZE: ")[1].split(".")[0])
    rep = int(metadata_str.split("REPEAT: ")[1].split(".")[0])
    freq = float(metadata_str.split("FREQUENCY: ")[1].strip("."))
    return num_frames, file_size, rep, freq


def xinit():
    os.system("sudo xinit -- -nocursor")

os.environ["DISPLAY"] = ":0"
xinit_process = Process(target=xinit)
xinit_process.start()
print("X server started.")

print("Setting up socket.")
port = 6000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(("", port)) 
sock.listen(1) 
print("Server ready to accept connections")
conn, addr = sock.accept() 

from pi_display_utils import create_window, get_sleep_time

display_queue = Queue()
disp_times_queue = Queue()
display_manager_thread = threading.Thread(target=display, args=(display_queue, disp_times_queue,))
display_manager_thread.start()

main_vid_dir = 'cal_bmps'
last_recv_seq = 0
last_disp_seq = -1
last_disp_start = None
last_disp_end = None
last_disp_times = None

os.system(f"rm -r {main_vid_dir}/*")
os.makedirs(main_vid_dir, exist_ok=True)

try:
    while True:
        try:
            seq_num, start_disp, end_disp, disp_times = disp_times_queue.get(block=False, timeout=0.01)
            last_disp_seq = seq_num #update last_disp info
            last_disp_start = start_disp
            last_disp_end = end_disp
            last_disp_times = disp_times

            os.system(f"rm -r {main_vid_dir}/seq{last_disp_seq}")

        except Empty:
            pass

        #signal ready to receive requests
        conn.send("READY RECEIVE REQUESTS".encode())

        #receive a request
        req = conn.recv(1024)
        # print(req)
        req = req.decode()
        if req == "REQSEQ":
            msg = "Last displayed sequence: {}. Start: {}. End: {}. Timestamps: {}. READY FOR METADATA.".format(last_disp_seq, last_disp_start, last_disp_end, last_disp_times).encode()
            conn.send(len(msg).to_bytes(8, byteorder='big') + msg)
        else:
            #receive display metadata
           # print("Waiting to receive metadata.")
            metadata_str = req
            #print("Metadata: ", metadata_str)
            _, file_size, rep, freq = parse_metadata(metadata_str)

            #respond to metadata with latest display info 
            msg = "Last displayed sequence: {}. Start: {}. End: {}. Timestamps: {}. READY FOR METADATA.".format(last_disp_seq, last_disp_start, last_disp_end, last_disp_times).encode()
            conn.send(len(msg).to_bytes(8, byteorder='big') + msg)
            
    
            zipped_filename = "{}/seq{}.zip".format(main_vid_dir, last_recv_seq) #filename = "{}/seq{}_{}.bmp".format(main_vid_dir, last_recv_seq, i)
            tot_recv_bytes = 0  
            with open(zipped_filename,'wb') as file:
                while tot_recv_bytes < file_size:
                    recvfile = conn.recv(4096)
                    tot_recv_bytes += len(recvfile)
                    if recvfile is None: break
                    file.write(recvfile)
            print(f"{zipped_filename} has been received at {datetime.fromtimestamp(time.time())}")

            #unzip the file and get names of files inside it
            os.system(f"unzip {zipped_filename} -d {main_vid_dir} >/dev/null 2>&1")
            os.system(f"rm {zipped_filename}")
            
            bmp_dir_path = f"{main_vid_dir}/seq{last_recv_seq}"
            filenames = glob.glob(f"{bmp_dir_path}/*")
            num_frames = len(filenames)
            filenames = sorted(filenames, key = lambda x : int(x.split("frame")[1].split(".bmp")[0]))

            frames = []
            for f, filename in enumerate(filenames):
                frame = cv2.imread(filename)
                frames.append(frame)
            display_queue.put((frames, last_recv_seq, freq, rep))
             
            last_recv_seq += 1

except ConnectionResetError:
    conn.close()
    xinit_process.join()



