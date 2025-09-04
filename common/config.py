"""
End-to-end configuration of the VeriLight system, spanning 
embedding, verification, and adaptation parameters.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
from Crypto.Random import get_random_bytes
import random
import pickle
import numpy as np
from colorama import Fore, Style, Back
import sys 
import os

from common.error_correctors import SoftViterbi, ConcatenatedViterbiRS, get_rs_params
from common.rp_lsh import CosineHashFamily

LOG_LEVEL  = "INFO"

session_output_path  = "hadleigh_aug10_speech4" # path to folder to store output for this session. will be created if it doesn't exist
common_abs_path = f"{os.getcwd()}/common" # absolute path to the common folder on this machine
host_ip = "raspberrypi.local" # IP address of the Pi
host_username = "hadleigh" # username on the Pi
scale_factor_disp = 1.2 # scale the SLM bitmaps by this factor for control of global display size

###########################################################################################
############################    DO NOT EDIT BELOW THIS LINE    ############################
###########################################################################################

#####################################
#######   ADAPTIVE EMBEDDING  #######
#####################################
embedding_window_duration = 4 #seconds
N = 29 # edge length of cell (SLM pixels)
buffer_space = 12 # num SLM pixels between cells
init_min_tot_slm = 140 # initial value for minimum total SLM intensity (will be incremented/decremented through adaptation)
init_slm_bgr =  [0, 255, 250] # color of the SLM cells before adaptation kicks in (first few sequences)
min_tot_slm_inc = 5 # increment for minimum total SLM intensity
min_tot_slm_dec = 5 # decrement for minimum total SLM intensity (applied when perceptibility is too high but BER is still low)
fade = False
calibration_frequency = 6
calibration_repeat = 30
frequency = 3
repeat = 1 
localization_frequency = 6
if localization_frequency % frequency != 0:
    print("WARNING: Localization frequency not divisible by comms frequency. Must change")
    sys.exit()

###################################
#######   MISCELLANEOUS   ########
###################################
ADAPT = True # can be set to False to disable adaptation and use init_min_tot_slm and init_slm_bgr throughout, for debugging

save_calibration_video = True
vis_calibration_video = False

port = 6000  #port on Rapsberry Pi to establish client-server connection with 

###################################
#######   VERIFICATION  ########
###################################
window_alignment_epsilon = 7 # explore shifts of +/- this many frames when computing dynamic hash distances
min_interwin_time = embedding_window_duration * .9 # used in trough detection for window start/end frame detection
interwin_detection_bandpass_tolerance = 2 

interwin_upper_env_rollingmax_n = 9
interwin_upper_env_rollingavg_n = 9

###################################
#######   DEMODULATION  ########
###################################
signal_rollingavg_n = 4
signal_detrending_rollingmin_n = signal_detrending_rollingavg_n = 11 # None if no detrending wanted

###################################
#######   SIGNATURE ASSEMBLY   ########
###################################
bin_seq_num_size = 8 # bits
date_ordinal_size = 15 # bits
creation_date_ordinal = 739313 # subtract this from all date ordinals to prevent overflow of our <date_ordinal_size> bits dedicated to days
unit_id_size = 8  #bits
if not os.path.exists(f"{common_abs_path}/unit_id.txt"):
    with open(f"{common_abs_path}/unit_id.txt", "w") as file:
        unit_id = random.getrandbits(unit_id_size)
        file.write(str(unit_id))
else:
    with open(f"{common_abs_path}/unit_id.txt", "r") as file:
        unit_id = int(file.read())

#################################
######   HEATMAP SETTINGS   #####
#################################
# frequencie ranges for seraching for heatmap
lower_range_start = 2 #2
lower_range_end = 5 #5
upper_range_start = 7 #7
upper_range_end = 11 #11
target_lower_epsilon = 0.5 #0.5
target_upper_epsilon = 0.5 #0.5

calibration_lower_range_start = 2
calibration_lower_range_end = 7
calibration_upper_range_start = 9
calibration_upper_range_end = 15
calibration_target_lower_epsilon = 0.5
calibration_target_upper_epsilon = 0.5 #0.5 if 6 hz or less

max_heatmap_pixels = 50000 # downsample images to this number of pixels max for faster generation of heatmap

############################
###  SLM BITMAP PARAMS  ###
############################
corner_markers = False
slm_W = 640
slm_H = 360
max_cells_W = int((slm_W ) / (N + buffer_space))
max_cells_H = int((slm_H + buffer_space) / (N + buffer_space))
localization_N = 2  #dimensions, in cells, over localization markers to be aded to corners, None if no localization corners used


if localization_N is not None:
    max_cells = max_cells_H * max_cells_W - 4*localization_N**2
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4 - ((2*localization_N - 1)*4)
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells - 4*(localization_N**2)
    reserved_localization_cells = []
else:
    max_cells = max_cells_H * max_cells_W 
    num_pilot_cells = max_cells_W * 2 + max_cells_H * 2 - 4
    num_info_cells = (max_cells_H * max_cells_W) - num_pilot_cells
    reserved_localization_cells = [] #"r-c" index format. leave empty if don't want any reserved.
    count = 0
    for c in range(max_cells_W):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{0}-{c}")
        count += 1
    for r in range(1, max_cells_H):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-{max_cells_W-1}")
        count += 1
    for c in range(max_cells_W-2, -1, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{max_cells_H-1}-{c}")
        count += 1
    for r in range(max_cells_H-2, 0, -1):
        if count % 2 == 0:
            reserved_localization_cells.append(f"{r}-0")
        count += 1

max_bits = int(frequency * embedding_window_duration * num_info_cells)

##################################
#########   ADAPTATION   #########
##################################
max_ber = 0.1 #not currently using
max_soft_decoding_err = 0 #bits
max_perceptibility = 5
"""
this is minimum ratio of the normalized power of the localization marker signals at the localization_frequency 
to the normalized power of the pilot signals at the localization frequency 
"""
min_norm_localization_ratio = 3

##################################
#########   ENCRYPTION/HMAC Generation   #########
##################################
tag_size = 10 # bytes 
key_size = 16 #bytes
key_path = f"{common_abs_path}/key{key_size}.pkl"
if not os.path.exists(key_path):
    key = get_random_bytes(key_size)
    with open(key_path, "wb") as pklfile:
        pickle.dump(key, pklfile)
else:
    with open(key_path, "rb") as pklfile:
        key = pickle.load(pklfile)

####################################
#####   MEDIAPIPE EXTRACTION   #####
####################################
target_fps = 25 # run core unit camera at this fps
intitial_face_detection = True
initial_bbox_padding = 30
target_features = ['0-17', '40-17', '270-17', '0-91', '0-321', 6, 7, 8, 9, 10, 11, 12, 23, 25, 50, 51] 
bbox_norm_dists = True

downsample_frames = 2/3 # speed up image saving/loading on core unit by downsampling cam frames by this factor
video_window_duration = embedding_window_duration + 0.5 #seconds
target_samples_per_second = 20 # <- downsample features to this rate
single_dynamic_signal_len = int(target_samples_per_second * video_window_duration)
concat_dynamic_signal_len = single_dynamic_signal_len * len(target_features)

##############################################
#####   HASH SIZE AND ERROR CORRECTION   #####
##############################################
# Viterbi k
vit_k = 5
# use user-specified hash sizes
identity_hash_k = 150
dynamic_hash_k = 150
digest_size = int(np.ceil((bin_seq_num_size + identity_hash_k//2 + dynamic_hash_k + unit_id_size + date_ordinal_size )/8)) * 8 
signature_size = digest_size + tag_size * 8

# determine how much RS encoding can be used
rs_n, rs_k = get_rs_params(max_bits, signature_size, viterbi_k=vit_k)
if rs_n <= rs_k:
    error_corrector = SoftViterbi(vit_k)
    viterbi_signature_size = signature_size * 2 + (vit_k * 2) # this many bits of the final encoded payload are actually carrying info (rest should be padding zeros)
else:
    error_corrector = ConcatenatedViterbiRS(vit_k, rs_n, rs_k)
    viterbi_signature_size = (rs_n * 8 * 2) + (vit_k * 2) # this many bits of the final encoded payload are actually carrying info (rest should be padding zeros)

#initialize random projection LSH functions
id_hash_func_name = f"len512_k{identity_hash_k}"
if not os.path.exists(f"{common_abs_path}/{id_hash_func_name}.pkl"):
    id_fam = CosineHashFamily(512)
    id_hash_funcs = [id_fam.create_hash_func() for h in range(identity_hash_k)]

    with open(f"{common_abs_path}/{id_hash_func_name}.pkl", "wb") as file:
        pickle.dump(id_fam, file)
        pickle.dump(id_hash_funcs, file)
else:
    with open(f"{common_abs_path}/{id_hash_func_name}.pkl", "rb") as file:
        id_fam = pickle.load(file)
        id_hash_funcs = pickle.load(file)

dyn_hash_func_name = f"len{concat_dynamic_signal_len}_k{dynamic_hash_k}"
if not os.path.exists(f"{common_abs_path}/{dyn_hash_func_name}.pkl"):
    dynamic_fam = CosineHashFamily(concat_dynamic_signal_len)
    dynamic_hash_funcs = [dynamic_fam.create_hash_func() for h in range(dynamic_hash_k)]

    with open(f"{common_abs_path}/{dyn_hash_func_name}.pkl", "wb") as file:
        pickle.dump(dynamic_fam, file)
        pickle.dump(dynamic_hash_funcs, file)
else:
    with open(f"{common_abs_path}/{dyn_hash_func_name}.pkl", "rb") as file:
        dynamic_fam = pickle.load(file)
        dynamic_hash_funcs = pickle.load(file)

#########################
#####   COLORSPACE  #####
#########################
colorspace = "bgr"
target_channel = "sum"

##########################
#####  CELL GROUPING #####
##########################
# parameters for grouping cells in heatmap into a line
slope_epsilon = 0.03
yint_epsilon = 30
max_vert_xdiff = 10
max_vert_sep = 10

##########################
##### PREPARE OUTPUT #####
##########################    
print("============================= CONFIG =============================")
print(f"Trial materials output at {session_output_path}")
   
if not ADAPT:
    print(f"""ADAPT= {Style.BRIGHT}{Back.MAGENTA}{ADAPT}{Style.RESET_ALL}""")
else:
    print(f"ADAPT = {ADAPT}")

print(f"Frequency = {frequency}")
print(f"Target lower frequency range = {lower_range_start}-{lower_range_end}")
print(f"Target upper frequency range = {upper_range_start}-{upper_range_end}")
print(f"Target frequency epsilon = {target_lower_epsilon}-{target_upper_epsilon}")
print(f"Fade = {fade}")
print(f"N = {N}, buffer_space = {buffer_space}")
print(f"Max cells_H = {max_cells_H}, Max cells_W = {max_cells_W}")
print(f"Num pilot cells = {num_pilot_cells}, Num info cells = {num_info_cells}")
print(f"Max bits per window = {max_bits}")
print("Raw payload size = ", signature_size)
if rs_n <= rs_k:
    print(f"""Error corrector: Soft Viterbi{Fore.YELLOW}. No room for RS encoding{Style.RESET_ALL}""")
else:
    print(f"Error corrector: Concatenated Viterbi + RS encoding with n = {rs_n}, k = {rs_k}")
print(f"Viterbi payload size = {viterbi_signature_size}")
print(f"Initial SLM BGR = {init_slm_bgr}")
print(f"Initial min_tot_slm = {init_min_tot_slm}")
print("=================================================================")
