"""
Utility functions for core unit calibration

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import numpy as np
import cv2
import itertools
import os
import itertools

import common.config as config

###########################
# CALIBRATION HOMOGRAPHY UTILS #
###########################
def create_calibration_code(N, cell_color, output_dir_path, flip=False):
    """
    Create calibration frames, which consist of four blinking corner cells

    Args:
        N (int): Side dimension of each square corner cell, in BMP pixels
        cell_color (list of 4 3-tuples or individual 3-tuple): If list of 4 3-tuples, each tuple represents the RGB color of a corner cell, in the order of top left, top right, bottom left, bottom right. Otherwise, all corner cells will have the same color, specified by the single 3-tuple.
        output_dir_path (str): Path to save the calibration frames
        flip (bool): If True, flip the calibration frames vertically. Some SLMs may require this to display with correct orientation
    """
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    W = 640
    H = 360
    frame = np.zeros((H, W, 3))
    
    if len(cell_color) == 4: #specifiying separate color for each corner
        top_left = cell_color[0]
        top_right = cell_color[1]
        bottom_left = cell_color[2]
        bottom_right = cell_color[3]
    else:
        top_left = top_right = bottom_left = bottom_right = cell_color

    frame[:int(config.N ),:int(config.N ),0] = top_left[0]
    frame[:int(config.N ),:int(config.N ),1] = top_left[1]
    frame[:int(config.N ),:int(config.N ),2] = top_left[2]

    frame[:int(config.N ),int(config.slm_W )-int(config.N ):int(config.slm_W ),0] = top_right[0]
    frame[:int(config.N ),int(config.slm_W )-int(config.N ):int(config.slm_W ),1] = top_right[1]
    frame[:int(config.N ),int(config.slm_W )-int(config.N ):int(config.slm_W ),2] = top_right[2]

    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),:int(config.N ),0] = bottom_left[0]
    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),:int(config.N ),1] = bottom_left[1]
    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),:int(config.N ),2] = bottom_left[2]

    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),int(config.slm_W ) - int(config.N ):int(config.slm_W ),0] = bottom_right[0]
    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),int(config.slm_W ) - int(config.N ):int(config.slm_W ),1] = bottom_right[1]
    frame[int(config.slm_H ) - int(config.N ):int(config.slm_H ),int(config.slm_W ) - int(config.N ):int(config.slm_W ),2] = bottom_right[2]

    if flip:
        frame = cv2.flip(frame, 0)
    cv2.imwrite('{}/frame0.bmp'.format(output_dir_path), frame)
    blank_frame = np.zeros((H, W, 3))
    cv2.imwrite('{}/frame1.bmp'.format(output_dir_path), blank_frame)

def rectangle_is_within(r1, r2):
  """
  Check if rectangle r1 is within rectangle r2.

  Args:
    r1 (list of floats/ints): A rectangle represented by a tuple of (x1, y1, x2, y2).
    r2 (list of floats/ints): A rectangle represented by a tuple of (x1, y1, x2, y2).

  Returns:
    True if r1 is within r2, False otherwise.
  """

  # Check if the top-left corner of r1 is within r2.
  if r1[0] < r2[0] or r1[1] < r2[1]:
    return False

  # Check if the bottom-right corner of r1 is within r2.
  if r1[2] > r2[2] or r1[3] > r2[3]:
    return False

  # If both checks pass, then r1 is within r2.
  return True

def get_blob_surroundings_density(center_point, contours, d):
    """
    Get the density of contours around a center point within a square of side length 2d

    Args:
        center_point (tuple of ints): The (x, y) coordinates of the center point.
        contours (list of tuples of ints): A list of (x, y) coordinates representing the contours.
        d (int): Half the side length of the square area to consider around the center point.
        t (numpy array): Optional image for visualization purposes.

    Returns:
        density (float): The density of contours within the square area.
        covered_contour_indices (list of ints): Indices of contours that are within the square area.
    """
    contour_bbox = [center_point[0] - d, center_point[1] - d, center_point[0] + d, center_point[1] + d]
    count = 0
    covered_contour_indices = []
    for i, c in enumerate(contours):
        if c[0] > contour_bbox[0] and c[1] > contour_bbox[1] and c[0] < contour_bbox[2] and c[1] < contour_bbox[3]:
            count += 1
            covered_contour_indices.append(i)
    density = count #/ (d*2)**2
    return density, covered_contour_indices


def detect_heatmap_cells(heatmap, density_diameter = 0, density_threshold = 0.001, area_threshold = None, otsu_inc = 0, erode = 0, kernel_dim = 5, min_squareness = None, blurthensharp = True, display=False):
    """
    Detect the four localization cells in heatmap using contour detection and
    various criteria for filtering out erroneous contours, such as density of contours in the area,
    area of the contour, and squareness of the contour bounding box.

    Args:
        heatmap (numpy array): The input heatmap image.
        density_diameter (int): Diameter of area to consider for density filtering. If 0, no density filtering is applied.
        density_threshold (float): Maximum number of contours allowed within the density area for a contour to be kept.
        area_threshold (tuple of two floats/ints): Minimum and maximum area thresholds for filtering contours. If None, no area filtering is applied.
        otsu_inc (int): Increment to add to Otsu's threshold for binary thresholding.
        erode (int): Number of iterations for erosion and dilation to remove noise. If 0, no erosion/dilation is applied.
        kernel_dim (int): Dimension of the kernel used for erosion and dilation. Must be an odd integer.
        min_squareness (float): Minimum squareness ratio (width/height) for filtering contours. If None, no squareness filtering is applied.
        blurthensharp (bool): If True, apply Gaussian blur followed by a sharpening filter to heatmap. If False, only apply Gaussian blur.
        display (bool): If True, display intermediate images for debugging purposes.
    
    Returns:
        list of tuples: A list of (x, y) coordinates representing the centers of the four localization cells.
        list of lists: A list of bounding boxes for the detected localization cells, each represented as [x, y, w, h].
    """
    # blurring/sharpening of heatmap
    if blurthensharp:
        kernel = np.array([[0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]])
        heatmap = cv2.GaussianBlur(heatmap,(3,3),0)
        heatmap= cv2.filter2D(heatmap, -1, kernel)
    else:
        heatmap = cv2.GaussianBlur(heatmap,(3,3),0)
    
    #otsu threshold of heatmap
    otsu_ret, _ = cv2.threshold(heatmap, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th = cv2.threshold(heatmap, otsu_ret + otsu_inc, 255, cv2.THRESH_BINARY)
    if display:
        cv2.imshow('Otsu Thresholding Results', th)
        cv2.waitKey(0)
    
    #erode/dilate to remove noise
    if erode > 0:
        kernel = np.ones((kernel_dim, kernel_dim), np.uint8) #must use odd kernel to avoid shifting
        th = cv2.erode(th, kernel, iterations=erode)
        th = cv2.dilate(th, kernel, iterations=erode)
        if display:
            cv2.imshow('Dilate + Eroded', th)
            cv2.waitKey(0)
    
    #detect contours and get contour info
    contours, hierarchy = cv2.findContours(th, 1, 2)
    contour_centers = []
    contour_areas = []
    contour_bboxes = []
    if display:
        vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        th_vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x1,y1 = cnt[0][0]
        approx__vertices = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        contour_bboxes.append([x, y, w, h])
        contour_center = (int(x+(w/2)), int(y+(h/2)))
        contour_area = w*h
        contour_areas.append(contour_area)
        contour_centers.append(contour_center)
        if display:
            vis_img = cv2.drawContours(vis_img, [cnt], -1, (0,255,0), 1)
            vis_img = cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 0, 255), 1)
            th_vis = cv2.drawContours(th_vis, [cnt], -1, (0,255,0), 2)
    if display:
        cv2.imshow("Unfiltered countour detections.", vis_img)
        cv2.waitKey(0)
    contour_centers = np.array(contour_centers)
    contour_areas = np.array(contour_areas)
    contour_bboxes = np.array(contour_bboxes)

    # filter contours by density
    if density_diameter > 0:
        density_remove_indices = []
        # covered_contour_indices = []
        for i in range(density_diameter // 2, heatmap.shape[1], density_diameter):
            for j in range(density_diameter // 2, heatmap.shape[0], density_diameter):
                center_point = (i, j)
                vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
                density, this_contour_covered_indices = get_blob_surroundings_density(center_point, contour_centers, density_diameter // 2) 
                if density > density_threshold:
                    density_remove_indices.extend(this_contour_covered_indices)
        contour_centers = np.delete(contour_centers, density_remove_indices, axis=0)
        contour_areas = np.delete(contour_areas, density_remove_indices, axis=0)
        contour_bboxes = np.delete(contour_bboxes, density_remove_indices, axis=0)
        if display:
            vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
            for i in range(len(contour_bboxes)):
                bbox = contour_bboxes[i]
                cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
                cv2.circle(vis_img, contour_centers[i], 2, (0, 0, 255), -1)

            cv2.imshow(f"Filtered countour bboxes (after density filtering w/ threshold {density_threshold})", vis_img)
            cv2.waitKey(0)
    
    # group any contours that are within each other
    contour_indices = [i for i in range(len(contour_centers))]
    pairs = list(itertools.combinations(contour_indices, 2))
    remove_indices = []
    for pair in pairs:
        top_left1 = contour_bboxes[pair[0]][:2]
        bottom_right1 = [contour_bboxes[pair[0]][0] + contour_bboxes[pair[0]][2], contour_bboxes[pair[0]][1] + contour_bboxes[pair[0]][3]]
        top_left2 = contour_bboxes[pair[1]][:2]
        bottom_right2 = [contour_bboxes[pair[1]][0] + contour_bboxes[pair[1]][2], contour_bboxes[pair[1]][1] + contour_bboxes[pair[1]][3]]
        if rectangle_is_within((top_left1[0], top_left1[1], bottom_right1[0], bottom_right1[1]), (top_left2[0], top_left2[1], bottom_right2[0], bottom_right2[1])):
            remove_indices.append(pair[0])
        if rectangle_is_within((top_left2[0], top_left2[1], bottom_right2[0], bottom_right2[1]), (top_left1[0], top_left1[1], bottom_right1[0], bottom_right1[1])):
            remove_indices.append(pair[1])
    contour_centers = np.delete(contour_centers, remove_indices, axis=0)
    contour_areas = np.delete(contour_areas, remove_indices, axis=0)
    contour_bboxes = np.delete(contour_bboxes, remove_indices, axis=0)
    if display:
        vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        for i in range(len(contour_bboxes)):
            bbox = contour_bboxes[i]
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
            cv2.circle(vis_img, contour_centers[i], 2, (0, 0, 255), -1)
            cv2.putText(vis_img, f"{contour_areas[i]:.1f}", (contour_centers[i][0], contour_centers[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Contour bboxes after grouping", vis_img)
        cv2.waitKey(0)

    # optional filtering by area
    if area_threshold is not None and len(contour_centers) > 4:
        contour_centers = contour_centers[np.where((contour_areas >= area_threshold[0]) & (contour_areas < area_threshold[1]))[0]]
        contour_bboxes = contour_bboxes[np.where((contour_areas >= area_threshold[0]) & (contour_areas < area_threshold[1]))[0]]
        contour_areas = contour_areas[np.where((contour_areas >= area_threshold[0]) & (contour_areas < area_threshold[1]))[0]]
    if display:
        vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        for i in range(len(contour_bboxes)):
            bbox = contour_bboxes[i]
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
            cv2.circle(vis_img, contour_centers[i], 2, (0, 0, 255), -1)

        cv2.imshow("Filtered countour bboxes (after area filtering)", vis_img)
        cv2.waitKey(0)
    
    # optional filtering by squareness
    if min_squareness is not None and len(contour_centers) > 4:
        squareness = contour_bboxes[:, 2] / contour_bboxes[:, 3]
        contour_centers = contour_centers[np.where((squareness >= min_squareness) & (squareness < 1/min_squareness))[0]]
        contour_bboxes = contour_bboxes[np.where((squareness >= min_squareness) & (squareness < 1/min_squareness))[0]]
        contour_areas = contour_areas[np.where((squareness >= min_squareness) & (squareness < 1/min_squareness))[0]]
    if display:
        vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        for i in range(len(contour_bboxes)):
            bbox = contour_bboxes[i]
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 1)
            cv2.circle(vis_img, contour_centers[i], 2, (0, 0, 255), -1)
        cv2.imshow("Filtered countour bboxes (after squarness filtering)", vis_img)
        cv2.waitKey(0)
        
    # sort by area and return only top four
    sorted_indices = np.argsort(contour_areas)[::-1]
    candidate_indices = sorted_indices[:4]
    filtered_contour_centers = list(contour_centers[candidate_indices])
    filtered_contour_bboxes = list(contour_bboxes[candidate_indices])
    if display:
        vis_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
        for i in range(len(filtered_contour_bboxes)):
            bbox = filtered_contour_bboxes[i]
            cv2.rectangle(vis_img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 0, 255), 1)
            cv2.circle(vis_img, filtered_contour_centers[i], 2, (0, 0, 255), -1)
        cv2.imshow("Contours to be returned", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # weird workadound to make destroyAllWindows work
        # https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
        for i in range (1,5):
            cv2.waitKey(1)  
    
    return filtered_contour_centers, filtered_contour_bboxes


def find_embedding_rows_simple(contour_centers, heatmap, slope_epsilon=0.05, y_epsilon=5, target_slope = None, display=False):
    """
    Groups contour centers into rows by finding the most popular slope between pairs of contours
    Using this slope, infer the rotation of the rows and visualize rotation so that all rows are horizontal
    Finally, group the rotated contour centers into rows based on y coordinate proximity

    Args:
        contour_centers (list of tuples): A list of (x, y) coordinates representing the centers of the detected contours.
        heatmap (numpy array): The input heatmap image, used for visualization purposes.
        slope_epsilon (float): Maximum difference in slope for two pairs of contours to be considered part of the same slope group.
        y_epsilon (int): Maximum difference in y-coordinate for two rotated contour centers to be considered part of the same row.
        target_slope (float): If provided, only consider slopes within slope_epsilon of this target slope.
        display (bool): If True, display intermediate images for debugging purposes.
    
    Returns:
        most_popular_slope (float): The most popular slope detected among the contour centers.
        rows (list of lists): A list of rows, where each row is a list of (x, y) coordinates representing the centers of the contours in that row.
    """
    pairs = list(itertools.combinations(contour_centers, 2))
    
    slopes = {}
    slope_updater = {}
    for pair in pairs:
        pair = [pair[0], pair[1]]
        if pair[0][0] - pair[1][0] == 0:
            slope = float('inf')
        else:
            slope = (pair[0][1] - pair[1][1]) / (pair[0][0] - pair[1][0])

        if target_slope is not None:
            if np.abs(slope - target_slope) > slope_epsilon:
                continue

        added = False
        for key in slopes.keys():
            updated_slope = np.mean(np.array(slope_updater[key]))
            if np.abs(slope - updated_slope) < slope_epsilon:
                slopes[key].append(pair)
                added = True
                #update slope in slope updater
                slope_updater[key].append(slope)
                break
        if not added:
            slopes[slope] = [pair]
            slope_updater[slope] = [slope]

    #reassign slopes keys to be the average of all pairwise slopes in that dict entry
    slopes_temp = {}
    for key, value in slopes.items():
        updated_slope = np.mean(np.array(slope_updater[key]))
        slopes_temp[updated_slope] = value
    slopes = slopes_temp

    slopes = sorted(slopes, key=lambda k: len(slopes[k]), reverse=True)

    most_popular_slope = slopes[0]
    
    theta = np.degrees(np.arcsin(most_popular_slope))
    M = cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), theta, 1)
    M_inv =  cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), -theta, 1)

    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_heatmap_rot = cv2.warpAffine(src=vis_heatmap, M=M, dsize=(heatmap.shape[1], heatmap.shape[0]))
    
    rot_cs = []
    for c in contour_centers:
        c = list(c)
        c.append(1)
        cprime = np.array(c).T
        c_rot =  M@cprime
        c_rot = c_rot[:2].astype(int)
        #cv2.circle(vis_heatmap_rot, c_rot, 1, (0, 0, 255), -1)
        rot_cs.append(c_rot.tolist())
    
    if display:
        cv2.imshow("Temp rot", vis_heatmap_rot)
        cv2.waitKey(0)

    rot_rows = []
    rows = []
    for cnum, rot_c in enumerate(rot_cs):
        added = False
        for i in range(len(rows)):
            rot_row = rot_rows[i]
       
            curr_row_y_avg = np.mean(np.array(rot_row)[:,1])
            if np.abs(rot_c[1]-curr_row_y_avg) < y_epsilon:
                rows[i].append(contour_centers[cnum])
                rot_rows[i].append(rot_c)
                added=True
                break
        if not added:
            rows.append([contour_centers[cnum]])
            rot_rows.append([rot_c])
    
    for l_num, line in enumerate(rows):
        if len(line) == 1:
            p1 = (line[0][0]-10, int(line[0][1] -  10*most_popular_slope))
            p2 = (line[0][0]+10, int(line[0][1] + 10*most_popular_slope))
            cv2.line(vis_heatmap, p1, p2, (255, 255, 255), 2)
            continue
        for i in range(0, len(line)):
            if i + 1 >= len(line):
                continue
            cv2.line(vis_heatmap, line[i], line[i+1], (255, 255, 255), 2)
    
    if display:
        cv2.imshow('Detected barcode rows', vis_heatmap)
        cv2.waitKey(0)

    return most_popular_slope, rows


def generate_calibration_reference_points(W, H, N, display=False):
    """
    Return center points of four corners comprising calibration code, used as correspondences
    for homography computation

    Args:
        W (int): Width of the reference image, in pixels
        H (int): Height of the reference image, in pixels
        N (int): Side dimension of each square corner cell, in pixels
        display (bool): If True, display the generated reference image
    
    Returns:
        reference_img (numpy array): The generated reference image with four white corner cells on a black background.
        corners (list of lists): A list of the (x, y) coordinates of the centers of the four corner cells, in the order of top left, top right, bottom left, bottom right
    """
    top_left = [int(N/2), int(N/2)]
    top_right = [ W-int(N/2), int(N/2),]
    bottom_left = [int(N/2), H - int(N/2), ]
    bottom_right = [W-int(N/2), H - int(N/2), ]
    corners = [top_left, top_right, bottom_left, bottom_right]
    
    reference_img = np.zeros((H, W)).astype(np.float32)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_GRAY2BGR)

    for cell_center in corners:
        cell_top = cell_center[1] - int(N/2)
        cell_bottom = cell_center[1] + int(N/2)
        cell_left = cell_center[0] - int(N/2)
        cell_right = cell_center[0] + int(N/2)

        reference_img[cell_top:cell_bottom+1, cell_left:cell_right+1] = 255
        
        cv2.circle(reference_img, cell_center, 2, (0, 0, 255), -1) 
    
    if display:
        cv2.imshow('Reference cell image', reference_img)
        cv2.waitKey(0)

    return reference_img, corners


def order_calibration_code_corners(contour_centers, heatmap, slope_epsilon=0.05, display=False):
    """
    Given contour centers assumed to be from calibration code corners, order them as
    top left, top right, bottom left, bottom right by first inferring the two rows (top and bottom),
    and then inferring the order based on x/y coordinates 

    Args:
        contour_centers (list of tuples): A list of (x, y) coordinates representing the centers of the detected contours.
        heatmap (numpy array): The input heatmap image, used for visualization purposes.
        slope_epsilon (float): Maximum difference in slope for two pairs of contours to be considered part of the same slope group.
        display (bool): If True, display intermediate images for debugging purposes.
    
    Returns:
        sorted_contour_centers (list of tuples): A list of the (x, y) coordinates of the centers of the four corner cells, 
                        in the order of top left, top right, bottom left, bottom right
        vis_heatmap (np array): The input heatmap image with the inferred corner order visualized.
    """

    if len(contour_centers) < 4:
        print("Not enough detected blobs to order")
        return None

    slope, rows = find_embedding_rows_simple(contour_centers, heatmap, slope_epsilon=slope_epsilon, y_epsilon=5, display=display)

    theta = np.degrees(np.arcsin(slope))
    M = cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), theta, 1)
    M_inv =  cv2.getRotationMatrix2D((int(heatmap.shape[1]/2), int(heatmap.shape[0]/2)), -theta, 1)
    
    vis_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2BGR)
    vis_heatmap_rot = cv2.warpAffine(src=vis_heatmap, M=M, dsize=(heatmap.shape[1], heatmap.shape[0]))
    line_ordered_contour_centers = []
    rot_cs = []
    for line in rows:
        for c in line:
            line_ordered_contour_centers.append(c)
            c = list(c)
            c.append(1)
            cprime = np.array(c).T
            c_rot =  M@cprime
            c_rot = c_rot[:2].astype(int).tolist()
           # cv2.circle(vis_heatmap_rot, c_rot, 1, (0, 0, 255), -1)
            rot_cs.append(c_rot)
            
    if display:
        for r in rot_cs:
            cv2.putText(vis_heatmap_rot, f"{r}", r, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow("Rows made horizontal", vis_heatmap_rot)
        cv2.waitKey(0)

    #infer corner order
    sorted_by_y = sorted(rot_cs , key=lambda k: [k[1], k[0]])
    top_row = sorted(sorted_by_y[:2], key=lambda k: k[0])
    top_left = line_ordered_contour_centers[rot_cs.index(top_row[0])]
    top_right = line_ordered_contour_centers[rot_cs.index(top_row[1])]
    bottom_row = sorted(sorted_by_y[2:], key=lambda k: k[0])
    bottom_left = line_ordered_contour_centers[rot_cs.index(bottom_row[0])]
    bottom_right = line_ordered_contour_centers[rot_cs.index(bottom_row[1])]
    sorted_contour_centers = [top_left, top_right, bottom_left, bottom_right]

    #visualize inferred order
    vis_heatmap = heatmap.copy()
    vis_heatmap = cv2.cvtColor(vis_heatmap, cv2.COLOR_GRAY2BGR)
    for i, c in enumerate(sorted_contour_centers):
        cv2.circle(vis_heatmap, c, 2, (0, 0, 255), -1)
        cv2.putText(vis_heatmap, str(i), c, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    if display:
        cv2.imshow("Inferred pilot nums", vis_heatmap)
        cv2.waitKey(0)

    return sorted_contour_centers, vis_heatmap 
      

def get_user_points(img):
    """
    Accept user input of calibration corners from clicks on the displayed img
    https://stackoverflow.com/questions/32770654/get-pixel-location-using-mouse-click-events

    Args:
        img (numpy array): The image on which the user will click to select points.
        
    Returns:
        clicks (list of lists): A list of the (x, y) coordinates of the points clicked by the user.
    """
    clicks = []
    
    #this function will be called whenever the mouse is right-clicked
    def mouse_callback(event, x, y, flags, params):
        nonlocal img
     
        if event == 1:
            #store the coordinates of the right-click event
            clicks.append([x, y])

            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow("Click corner points or press q to exit", img)

    cv2.namedWindow("Click corner points or press q to exit")
    # highgui function called when mouse events occur
    cv2.setMouseCallback("Click corner points or press q to exit", mouse_callback)

    k = 0
    while k!=113:
        # Display the image
        cv2.imshow("Click corner points or press q to exit", img)
        k = cv2.waitKey(0)

    cv2.destroyAllWindows()

    return clicks


