"""
Operate a client thread that sends frames to the Pi for calibration

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import socket 
import sys
import os
import glob
import threading

import common.config as config
from embedding.embedding_utils import lcm

DISPLAY_MESSAGES = True

def client_log(message, log_level):
    """
    Log messages from the client thread based on the configured log level.

    Args:
        message (str): The message to log.
        log_level (str): The severity level of the message ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    if log_level == "DEBUG":
        if config.LOG_LEVEL == "DEBUG":
            print("CLIENT [DEBUG]: {}".format(message))
    elif log_level == "INFO":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO":
            print("CLIENT  [INFO]: {}".format(message))
    elif log_level == "WARNING":
        if config.LOG_LEVEL == "DEBUG" or config.LOG_LEVEL == "INFO" or config.LOG_LEVEL == "WARNING":
            print("CLIENT [WARNING]: {}".format(message))
    elif log_level == "ERROR":
        print("CLIENT [ERROR]: {}".format(message))
    

def display_message(message):
    """
    Display a message receiverd from the server if DISPLAY_MESSAGES is enabled.
    """
    if DISPLAY_MESSAGES:
        print("CLIENT [MESSAGE]: {}".format(message))


class ClientThread(threading.Thread):
    """
    Thread to handle communication with the Raspberry Pi server via socket.
    https://stackoverflow.com/questions/25904537/how-do-i-send-data-to-a-running-python-thread
    """
    def __init__(self, req_queue, seq_info_queue, exc_queue, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.req_queue = req_queue
        self.seq_info_queue = seq_info_queue
        self.exc_queue = exc_queue
        self.daemon = True
        self.port = args[0]
        self.host_ip = args[1]
   
    def run(self):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_log("Socket successfully created", "INFO")
        except socket.error as err:
            client_log("Socket creation failed with error %s" %(err), "ERROR")
            pass
        # connecting to the server
        sock.connect((self.host_ip, self.port))

        client_log("The socket has successfully connected.", "INFO")

        try:
            while True:           
                #wait until parent thread requests you do smoething
                req = self.req_queue.get()
                if req == None:
                    break

                #before sending anything, wait for confirmation from server that it is ready to receive things
                _ = sock.recv(22) # 22 bytes is len of ready receive requests message
   
                if "REQSEQ" in req:
                    # the driver requested updated display metadata from SLM
                    client_log("Sent request for sequence info", "DEBUG")
                    sock.send("REQSEQ".encode())
                    #wait here until SLM responds with it
                    msg_size = int.from_bytes(sock.recv(8), byteorder='big')
                    last_disp_info = b""
                    while len(last_disp_info) < msg_size:
                        last_disp_info += sock.recv(5000)
                    last_disp_info = last_disp_info.decode()
                    self.seq_info_queue.put(last_disp_info)
                else:
                    dir, cal = req
                    client_log("Got request from parent to send frames at {}".format(dir), "DEBUG")
                    files = glob.glob(dir + "/*")
                    
                    # set display params based on config
                    if cal:
                        repeat = config.calibration_repeat
                        frequency = config.calibration_frequency
                    else:
                        repeat = config.repeat
                        if config.localization_frequency is not None:
                            frequency = lcm(config.localization_frequency, config.frequency)
                        else:
                            frequency = config.frequency
                    # compress the folder
                    seq_num = int(dir.split("seq")[-1])
                    zipped_path = f"{config.session_output_path}/bmps/seq{seq_num}.zip"
                    os.system(f"cd {config.session_output_path}/bmps \ && zip seq{seq_num}.zip seq{seq_num}/* >/dev/null 2>&1")


                    file_size = os.path.getsize(zipped_path)

                    #send metadata to SLM
                    num_frames = len(files)
                    metadata = "NUMFRAMES: {}. FRAMEFILESIZE: {}. REPEAT: {}. FREQUENCY: {}".format(num_frames, file_size, repeat, frequency)
                    sock.send(metadata.encode())

          
                    # wait for receipt of last display info, which also serves to signal that 
                    # the server is ready to receive the first/another frame. Hand it over to driver
                    msg_size = int.from_bytes(sock.recv(8), byteorder='big')
                    last_disp_info = b""
                    while len(last_disp_info) < msg_size:
                        last_disp_info += sock.recv(4096)
                    last_disp_info = last_disp_info.decode()
                    self.seq_info_queue.put(last_disp_info)
                    
                    #send zipped folder to SLM
                    with open(zipped_path, 'rb') as file:
                        sendfile = file.read()
                    sock.sendall(sendfile)
                    client_log(f"{zipped_path} sent.", "DEBUG")
            
        
            # close the connection
            sock.close()    

        except BaseException as e:
            client_log("Client thread failed with error: {}".format(e), "ERROR")
            self.exc_queue.put(sys.exc_info())
    



    