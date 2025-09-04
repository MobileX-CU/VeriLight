"""
Utility functions for computing the perceptibility of projected cells/optical modulations

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import numpy as np
import cv2
import math
import skimage

import common.config as config

def get_on_off_cell_val_from_cell_signal(cell_signal, image_sequence, cell_boundaries, output_name = None):
    """
    the off and on cell valuse should just be the RGB value at the frame in which the cell's R + G + B is 
    minimized or maximized across the whole signal. This follows from the fact that the SLM always adds light, increasing
    the intensity in some channel(s). 

    Args:
        cell_signal (list or numpy array): The intensity signal of the cell over time.
        image_sequence (numpy array): The sequence of images (frames) from which to extract colors.
        cell_boundaries (tuple): The boundaries of the cell in the format (top, left, bottom, right).
        output_name (str, optional): If provided, saves a plot of the cell signal with min/max points marked.

    Returns:
        min_intensity_bgr (tuple): The BGR color of the cell at minimum intensity.
        max_intensity_bgr (tuple): The BGR color of the cell at maximum intensity
    """ 
    # to save some time, use existing cell signal to determine frame where min intensity occurs
    min_intensity = float('inf')
    min_intensity_i = 0
    max_intensity = float('-inf')
    max_intensity_i = 0
    for i, intensity in enumerate(cell_signal):
        if intensity < min_intensity:
            min_intensity = intensity
            min_intensity_i = i
        if intensity > max_intensity:
            max_intensity = intensity
            max_intensity_i = i
    
    cell_top, cell_left, cell_bottom, cell_right = cell_boundaries
    mask = np.zeros((config.slm_H, config.slm_W), np.uint8)
    cv2.rectangle(mask, (cell_left, cell_top), (cell_right, cell_bottom), 255, -1)   

    # for both min and max, apply mask and get bgr from those frames
    target_min_img = image_sequence[min_intensity_i, :,:, :]
    min_intensity_bgr = cv2.mean(target_min_img, mask=mask)[:3]

    target_max_img = image_sequence[max_intensity_i, :,:, :]
    max_intensity_bgr = cv2.mean(target_max_img, mask=mask)[:3]

    return min_intensity_bgr, max_intensity_bgr


def CIEDE2000(sRGB1, sRGB2):
    """
    Calculates CIEDE2000 color distance between two sRGB colors
    From https://github.com/lovro-i/CIEDE2000/blob/master/ciede2000.py, with addition of initial conversion 
    from srgb to lab

    Args:
        sRGB1 (tuple or list): First color in sRGB format (R, G, B) with values in [0, 255].
        sRGB2 (tuple or list): Second color in sRGB format (R, G, B) with values in [0, 255].
    
    Returns:
        float: The CIEDE2000 color distance between the two colors.
    """
    Lab_1 = list(skimage.color.rgb2lab([[[sRGB1[0] / 255, sRGB1[1] / 255, sRGB1[2] / 255]]])[0][0])
    Lab_2 = list(skimage.color.rgb2lab([[[sRGB2[0] / 255, sRGB2[1] / 255, sRGB2[2] / 255]]])[0][0])

    C_25_7 = 6103515625 # 25**7
    
    L1, a1, b1 = Lab_1[0], Lab_1[1], Lab_1[2]
    L2, a2, b2 = Lab_2[0], Lab_2[1], Lab_2[2]
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    C_ave = (C1 + C2) / 2
    G = 0.5 * (1 - math.sqrt(C_ave**7 / (C_ave**7 + C_25_7)))
    
    L1_, L2_ = L1, L2
    a1_, a2_ = (1 + G) * a1, (1 + G) * a2
    b1_, b2_ = b1, b2
    
    C1_ = math.sqrt(a1_**2 + b1_**2)
    C2_ = math.sqrt(a2_**2 + b2_**2)
    
    if b1_ == 0 and a1_ == 0: h1_ = 0
    elif a1_ >= 0: h1_ = math.atan2(b1_, a1_)
    else: h1_ = math.atan2(b1_, a1_) + 2 * math.pi
    
    if b2_ == 0 and a2_ == 0: h2_ = 0
    elif a2_ >= 0: h2_ = math.atan2(b2_, a2_)
    else: h2_ = math.atan2(b2_, a2_) + 2 * math.pi

    dL_ = L2_ - L1_
    dC_ = C2_ - C1_    
    dh_ = h2_ - h1_
    if C1_ * C2_ == 0: dh_ = 0
    elif dh_ > math.pi: dh_ -= 2 * math.pi
    elif dh_ < -math.pi: dh_ += 2 * math.pi        
    dH_ = 2 * math.sqrt(C1_ * C2_) * math.sin(dh_ / 2)
    
    L_ave = (L1_ + L2_) / 2
    C_ave = (C1_ + C2_) / 2
    
    _dh = abs(h1_ - h2_)
    _sh = h1_ + h2_
    C1C2 = C1_ * C2_
    
    if _dh <= math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2
    elif _dh  > math.pi and _sh < 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 + math.pi
    elif _dh  > math.pi and _sh >= 2 * math.pi and C1C2 != 0: h_ave = (h1_ + h2_) / 2 - math.pi 
    else: h_ave = h1_ + h2_
    
    T = 1 - 0.17 * math.cos(h_ave - math.pi / 6) + 0.24 * math.cos(2 * h_ave) + 0.32 * math.cos(3 * h_ave + math.pi / 30) - 0.2 * math.cos(4 * h_ave - 63 * math.pi / 180)
    
    h_ave_deg = h_ave * 180 / math.pi
    if h_ave_deg < 0: h_ave_deg += 360
    elif h_ave_deg > 360: h_ave_deg -= 360
    dTheta = 30 * math.exp(-(((h_ave_deg - 275) / 25)**2))
    
    R_C = 2 * math.sqrt(C_ave**7 / (C_ave**7 + C_25_7))  
    S_C = 1 + 0.045 * C_ave
    S_H = 1 + 0.015 * C_ave * T
    
    Lm50s = (L_ave - 50)**2
    S_L = 1 + 0.015 * Lm50s / math.sqrt(20 + Lm50s)
    R_T = -math.sin(dTheta * math.pi / 90) * R_C

    k_L, k_C, k_H = 1, 1, 1
    
    f_L = dL_ / k_L / S_L
    f_C = dC_ / k_C / S_C
    f_H = dH_ / k_H / S_H
    
    dE_00 = math.sqrt(f_L**2 + f_C**2 + f_H**2 + R_T * f_C * f_H)
    return dE_00