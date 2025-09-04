"""
BPSK-encodeframes to send to SLM, ensuring the minimal data is transmited over the socket
for speed. 

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import numpy as np
import cv2
import time
import os
import pickle
import glob


import common.config as config
from embedding.embedding_utils import overall_cell_to_row_col, lcm

PRINT_TIMING = False

def create_mask(fade = True):
    """
    Create mask for full frame, either with localization corner or for localization border,
    based on config file settings

    Args:
        fade (bool): Whether to apply a fading effect to the info cells and localization markers.

    Returns:
        full_frame_mask (numpy array): The mask to be applied to the full frame.
        rightmost (int): The rightmost pixel coordinate of the info cells in the minimal frame.
        bottommost (int): The bottommost pixel coordinate of the info cells in the minimal frame
    """
    # make the info cell masks
    step_scale = 1.75 #fading parameter
    if fade:
        step = (step_scale / config.N) 
        info_cell = np.zeros((config.N, config.N, 3), dtype=np.float32)
        for i_index in range(config.N):
            for j_index in range(config.N):
                i = i_index + 1
                j = j_index + 1
                dist = np.sqrt(((i - config.N/2))**2 + ((j - config.N/2)**2))
                brightness_dec = step*dist
                info_cell[i_index, j_index, :] = [1 - brightness_dec, 1 - brightness_dec, 1 - brightness_dec]
    else:
        info_cell = np.ones((config.N, config.N, 3), dtype=np.float32) 

    full_frame_mask = np.zeros((config.slm_H, config.slm_W, 3), dtype = np.float32)

    # add info cell scalers to mask
    # keep track of right most and bottommost cells, as this is important when resizing the minimal frame
    rightmost = 0
    bottommost = 0
    for overall_cell_num in range(config.max_cells):
        r, c, = overall_cell_to_row_col(overall_cell_num)
        cell_top = r * (config.N + config.buffer_space)
        cell_bottom = cell_top + config.N 
        cell_left = c * (config.N + config.buffer_space)
        cell_right = cell_left + config.N 
        full_frame_mask[cell_top:cell_bottom, cell_left:cell_right, :] = info_cell
        if cell_right > rightmost:
            rightmost = cell_right
        if cell_bottom > bottommost:
            bottommost = cell_bottom

    # add localization corners, if needed
    if config.localization_N is not None:
        loc_cell_dim = config.localization_N * (config.N + config.buffer_space) - config.buffer_space

        if fade:
            step = (step_scale / loc_cell_dim) 
            loc_cell = np.zeros((loc_cell_dim, loc_cell_dim, 3))
            for i_index in range(loc_cell_dim):
                for j_index in range(loc_cell_dim):
                    i = i_index + 1
                    j = j_index + 1
                    dist = np.sqrt(((i - loc_cell_dim/2))**2 + ((j - loc_cell_dim/2)**2))
                    brightness_dec = step*dist
                    loc_cell[i_index, j_index, :] = [1 - brightness_dec, 1 - brightness_dec, 1 - brightness_dec]
        else:
            loc_cell = np.ones((loc_cell_dim, loc_cell_dim, 3), dtype=np.float32)
        
        bottom_top = (config.max_cells_H - config.localization_N) * (config.N + config.buffer_space) # pixel row correpsonding to top of bottom pair of markers
        right_left = (config.max_cells_W - config.localization_N) * (config.N + config.buffer_space)  # pixel column corresponding to left of right pair of markers
        
        full_frame_mask[0:loc_cell_dim,0:loc_cell_dim] = loc_cell
        full_frame_mask[0:loc_cell_dim,right_left:right_left+loc_cell_dim] = loc_cell
        full_frame_mask[bottom_top:bottom_top+loc_cell_dim,0:loc_cell_dim] = loc_cell
        full_frame_mask[bottom_top:bottom_top+loc_cell_dim,right_left:right_left+loc_cell_dim] = loc_cell
    
      
    return full_frame_mask, rightmost, bottommost


def create_minimal_frames(input_bitstring, cell_colors, loc_marker_colors, output_parent_dir):
    """
    Create minimal frames of correct type, according to config file specs.

    Args:
        input_bitstring (str): The bitstring to be encoded in the frames.
        cell_colors (list): List of colors for each cell in the format [R, G, B] with values from 0-255.
        loc_marker_colors (list): List of colors for the localization markers in the format [R, G, B] with values from 0-255.
        output_parent_dir (str): Directory to save the generated minimal frames.
    """
    if config.localization_N is not None:
        return create_minimal_frames_loc_corner(input_bitstring, cell_colors, loc_marker_colors, output_parent_dir)
    else:
        return create_minimal_frames_loc_border(input_bitstring, cell_colors, output_parent_dir)
           

def create_minimal_frames_loc_corner(input_bitstring, cell_colors, loc_marker_colors, output_parent_dir):
    """
    Create minimal frames with localization corners.

    Args:
        input_bitstring (str): The bitstring to be encoded in the frames.
        cell_colors (list): List of colors for each cell in the format [R, G, B] with values from 0-255.
        loc_marker_colors (list): List of colors for the localization markers in the format [R, G, B] with values from 0-255.
        output_parent_dir (str): Directory to save the generated minimal frames.

    Returns:
        info_cell_bit_list (list): List of bitstrings for each information cell.
    """

    os.makedirs(output_parent_dir, exist_ok=True)

    dummy_frame = np.zeros((config.max_cells_H, config.max_cells_W, 3), dtype=np.float32)
    max_bits_per_cell =  int(config.embedding_window_duration * config.frequency)
    max_tot_bits =  int(config.num_info_cells * max_bits_per_cell)
    if len(input_bitstring) > max_tot_bits:
        while True:
            raw = input(f"Input bitstring too long ({len(input_bitstring)} bits). Press y to truncate to maximum size of {max_tot_bits} bits, q to quit now.")
            if raw == "q":
                sys.exist()
            else:
                input_bitstring = input_bitstring[:max_tot_bits]
                break

    num_frames = int(config.embedding_window_duration * config.frequency * 2)
    lc = lcm(config.localization_frequency, config.frequency)
    num_frame_reps = int(lc / (config.frequency))

    for n in range(num_frames * num_frame_reps):
        start_create_frame = time.time()
        border_on = False
        corner_on = False
        if int(n / (lc / config.frequency)) % 2 == 0:
            border_on = True
        if int(n / (lc / config.localization_frequency)) % 2 == 0: 
            corner_on = True

        frame = dummy_frame.copy()
        info_cell_num = 0
        info_cell_bit_list = []
        for overall_cell_num in range(config.max_cells):
            r, c = overall_cell_to_row_col(overall_cell_num)
            if ((r == config.max_cells_H - 1) or (r == 0) or (c == config.max_cells_W - 1) or (c == 0)): 
                if border_on:
                    frame[r, c, :] = cell_colors[overall_cell_num]
            else:
                info_cell_bits = input_bitstring[info_cell_num*max_bits_per_cell:(info_cell_num + 1)*max_bits_per_cell]
                info_cell_bit_list.append(info_cell_bits)
                if (info_cell_bits[int(n / num_frame_reps /2)] == '0' and int(n / (lc / config.frequency)) % 2 == 0) or (info_cell_bits[int(n / num_frame_reps /2)] == '1' and int(n / (lc / config.frequency)) % 2 == 1):
                    frame[r, c, :] = cell_colors[overall_cell_num] 
                info_cell_num += 1
        if corner_on:
            frame[0:config.localization_N, 0:config.localization_N, :] = loc_marker_colors[0]
            frame[0:config.localization_N, config.max_cells_W-config.localization_N:config.max_cells_W, :]  = loc_marker_colors[1]
            frame[config.max_cells_H-config.localization_N:config.max_cells_H, 0:config.localization_N, :]= loc_marker_colors[2]
            frame[config.max_cells_H-config.localization_N:config.max_cells_H, config.max_cells_W-config.localization_N:config.max_cells_W, :] = loc_marker_colors[3]
        
        frame /= 255 #cv2 will interpret float32 arrays as images where 1,1,1 is white, as opposed to unit8 arrays, where 255,255,255 is white
        
        end_create_frame = time.time()

        start_save_frame = time.time()
        np.save(f"{output_parent_dir}/frame{n}.npy", frame)
        end_save_frame = time.time()

        if PRINT_TIMING:
            print(f"Minimal rame {n} created in {end_create_frame - start_create_frame:.3f} seconds. Saved in {end_save_frame - start_save_frame:.3f} seconds.")

    return info_cell_bit_list

def create_minimal_frames_loc_border(input_bitstring, cell_colors, output_parent_dir):
    """
    Create minimal frames with localization borders rathet than corners (experimental)

    Args:
        input_bitstring (str): The bitstring to be encoded in the frames.
        cell_colors (list): List of colors for each cell in the format [R, G, B] with values from 0-255.
        output_parent_dir (str): Directory to save the generated minimal frames.    
    
    Returns:
        info_cell_bit_list (list): List of bitstrings for each information cell.
    """
    os.makedirs(output_parent_dir, exist_ok=True)

    dummy_frame = np.zeros((config.max_cells_H, config.max_cells_W, 3), dtype=np.float32)
    max_bits_per_cell =  int(config.embedding_window_duration * config.frequency)
    max_tot_bits =  int(config.num_info_cells * max_bits_per_cell)
    if len(input_bitstring) > max_tot_bits:
        while True:
            raw = input(f"Input bitstring too long ({len(input_bitstring)} bits). Press y to truncate to maximum size of {max_tot_bits} bits, q to quit now.")
            if raw == "q":
                sys.exist()
            else:
                input_bitstring = input_bitstring[:max_tot_bits]
                break

    num_frames = int(config.embedding_window_duration * config.frequency * 2)
    lc = lcm(config.localization_frequency, config.frequency)
    num_frame_reps = int(lc / (config.frequency))

    for n in range(num_frames * num_frame_reps):
        start_create_frame = time.time()
        chan_border_on = False
        loc_border_on = False
        if int(n / (lc / config.frequency)) % 2 == 0:
            chan_border_on = True
        if int(n / (lc / config.localization_frequency)) % 2 == 0: 
            loc_border_on = True

        frame = dummy_frame.copy()
        info_cell_num = 0
        info_cell_bit_list = []
        for overall_cell_num in range(config.max_cells):
            r, c = overall_cell_to_row_col(overall_cell_num)
            if f"{r}-{c}" in config.reserved_localization_cells:
                if loc_border_on:  # loc cell should be on
                    frame[r, c, :] = cell_colors[overall_cell_num] 
            elif (r == config.max_cells_H - 1) or (r == 0) or (c == config.max_cells_W - 1) or (c == 0): #chan border cell
                if chan_border_on:
                    frame[r, c, :] = cell_colors[overall_cell_num]
            else: #info cell
                info_cell_bits = input_bitstring[info_cell_num*max_bits_per_cell:(info_cell_num + 1)*max_bits_per_cell]
                info_cell_bit_list.append(info_cell_bits)
                if (info_cell_bits[int(n / num_frame_reps / 2)] == '0' and int(n / (lc / config.frequency)) % 2 == 0) or (info_cell_bits[int(n / num_frame_reps / 2)] == '1' and int(n / (lc / config.frequency)) % 2 == 1):
                    frame[r, c, :] = cell_colors[overall_cell_num] 
                info_cell_num += 1

        frame /= 255 #cv2 will interpret float32 arrays as images where 1,1,1 is white, as opposed to unit8 arrays, where 255,255,255 is white
            
        end_create_frame = time.time()

        start_save_frame = time.time()
        np.save(f"{output_parent_dir}/frame{n}.npy", frame)
        end_save_frame = time.time()

        if PRINT_TIMING:
            print(f"Minimal rame {n} created in {end_create_frame - start_create_frame:.3f} seconds. Saved in {end_save_frame - start_save_frame:.3f} seconds.")

    return info_cell_bit_list


def create_full_frame(minimal_frame, mask, rightmost, bottommost):
    """
    Given a minimal frame and a mask correponding to the minimal frame's type (corner or border), 
    creates a full frame by scaling the minimal frame to the SLM's resolution and applying the mask.

    Args:
        minimal_frame (numpy array): The minimal frame to be converted to a full frame.
        mask (numpy array): The mask to be applied to the full frame.
        rightmost (int): The rightmost pixel coordinate of the info cells in the minimal frame.
        bottommost (int): The bottommost pixel coordinate of the info cells in the minimal frame
    
    Returns:
        final (numpy array): The full frame ready to be sent to the SLM.
    """
    start_create_full_frame = time.time()
    start_scale_factor = time.time()
    scale_factor = int(config.slm_W / minimal_frame.shape[1])
    final = cv2.resize(minimal_frame, (rightmost, bottommost), fx=scale_factor, fy=scale_factor, interpolation = cv2.INTER_NEAREST)
    # pad, if necessary
    right_pad = config.slm_W - final.shape[1]
    bottom_pad = config.slm_H - final.shape[0]
    final = cv2.copyMakeBorder(final, 0, bottom_pad, 0, right_pad, cv2.BORDER_CONSTANT, value = (0, 0, 0))
    end_scale_factor = time.time()
    start_mask = time.time()
    final *= mask
    end_mask = time.time()
    end_create_full_frame = time.time()
    if PRINT_TIMING:
        print(f"Full frame created in {end_create_full_frame - start_create_full_frame:.3f} s. Scaled in {end_scale_factor - start_scale_factor:.3f} s. Masked in {end_mask - start_mask:.3f} s.")
    return final


def create_sample_frame(fade = False, all_on = False, color = [1, .4, .4]):
    """
    Create an example SLM bitmap. Note: Colors are in scale 0-1 not 0-255.

    Args:
        fade (bool): Whether to apply a fading effect to the info cells and localization markers.
        all_on (bool): Whether to turn on all info cells (True) or randomly turn on half (False).
        color (list): The color to use for the info cells and localization markers in the format [R, G, B] with values from 0-1.
    
    Returns:
        full_frame (numpy array): The full frame ready to be sent to the SLM.
    """
    frame = np.zeros((config.max_cells_H, config.max_cells_W, 3), dtype=np.float32)
    
    # creating the minimal frame
    for overall_cell_num in range(config.max_cells):
        r, c = overall_cell_to_row_col(overall_cell_num)
        if ((r == config.max_cells_H - 1) or (r == 0) or (c == config.max_cells_W - 1) or (c == 0)):
            frame[r, c, :] = color  # [.38, .25, 1]
        else:
            if all_on:
                frame[r, c, :] = color
            else:
                # roll dice
                if np.random.rand() > .5:
                    frame[r, c, :] =   color
    frame[0:config.localization_N, 0:config.localization_N, :] =  color
    frame[0:config.localization_N, config.max_cells_W - config.localization_N:config.max_cells_W, :] =  color
    frame[config.max_cells_H - config.localization_N:config.max_cells_H, 0:config.localization_N, :] = color
    frame[config.max_cells_H - config.localization_N:config.max_cells_H, config.max_cells_W - config.localization_N:config.max_cells_W, :] = color
    mask, rightmost, bottommost = create_mask(fade = fade)
 
    full_frame = create_full_frame(frame, mask, rightmost, bottommost)
    return full_frame


def test():
    """
    Test function to create example minimal frames and corresponding full frames.
    """
    test_colors = [[100, 0, 100] for i in range(30)]
    for i in range(500):
        test_colors.append([0, 110, 0])
    for i in range(500):
        test_colors.append([0, 0, 155])
    for i in range(500):
        test_colors.append([155, 0, 0])

    base = '0010101110100001010010100001010101010101001001001001001000100100100100100101001'
    test_string = ''
    for i in range(800):
        test_string += base

    os.makedirs('test_minimal', exist_ok=True)
    os.makedirs('test_full', exist_ok=True)

    create_minimal_frames(test_string, test_colors, [(255, 255, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255)], 'test_minimal')
    mask, rightmost, bottommost = create_mask(fade = False)
    frames = glob.glob("test_minimal/*.npy")
    frames = sorted(frames, key=lambda x: int(x.split("frame")[-1].split(".")[0]))
    for i, f in enumerate(frames):
        minimal_frame = np.load(f)
        minimal_frame_bytes = pickle.dumps(minimal_frame)
        minimal_frame = pickle.loads(minimal_frame_bytes)
        full_frame = create_full_frame(minimal_frame, mask, rightmost, bottommost)
        cv2.imwrite(f"test_full/full_frame{i}.png", full_frame * 255) #NOTE! This is super weird, but if test_full has other conflicting files in it, this greatyl reduces run speed of this function on pi. 


def full_frames_from_folder(npys_path, fade = False):
    """
    Given a folder of minimal frames, creates full frames and returns them in a list. 
    For testing purposes.

    Args:
        npys_path (str): Path to the folder containing the minimal frames in .npy
        fade (bool): Whether to apply a fading effect to the info cells and localization markers.
    """
    output_path = npys_path + "/full_frames"
    os.makedirs(output_path, exist_ok=True)
    mask, rightmost, bottommost = create_mask(fade = fade)
    frames = glob.glob(f"{npys_path}/*.npy")
    frames = sorted(frames, key=lambda x: int(x.split("frame")[-1].split(".")[0]))
    for i, f in enumerate(frames):

        minimal_frame = np.load(f)
        full_frame = create_full_frame(minimal_frame, mask, rightmost, bottommost)
        cv2.imwrite(output_path + f"/full_frame{i}.png", full_frame * 255)



