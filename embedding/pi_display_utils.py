"""
Utility functions for displaying images via SLM. Run on Raspberry Pi.

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import cv2
import config

def get_sleep_time(freq):
    """
    Get system-dependent time to sleep in between frame displays in order to achieve modulations
    at <freq> Hz.

    Args:
        freq (int): Desired frequency in Hz.
    
    Returns:
        float: Time to sleep in seconds.
    """
    if freq == 3:
        sleep_time = 1/7
    elif freq == 4:
        sleep_time = 1/10
    elif freq == 5:
        sleep_time = 1/14
    elif freq == 6:
        sleep_time = 1/18
    elif freq == 8:
        sleep_time = 1/26
    elif freq == 9:
        sleep_time = 1/34
    elif freq == 12:
        sleep_time = 1/63
    else:
        print("No sleep time set for freq: ", freq)
        sleep_time = None
    return sleep_time


def create_window(window_name):
    """
    Create an OpenCV window for displaying frames.

    Args:
        window_name (str): Name of the window.
    """
    # set up opencv window. this must be inside this function, becaused namedWindow is not thread safe!
    window_width = int(640 * config.scale_factor_disp)
    window_height = int(360 * config.scale_factor_disp)
    cv2.namedWindow(
        window_name,
        flags=(cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_FREERATIO))
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1.0)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, 1.0)
    cv2.resizeWindow(
        window_name,
        window_width,
        window_height)
    cv2.moveWindow(
        window_name,
        0,
    -1)