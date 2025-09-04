"""
Utilities for visualization the verification process/results for a video

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
import cv2
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from rich.console import Console
from rich.table import Table
import pickle
import os
import ffmpeg

import common.config as config
from common.signal_utils import single_feature_signal_processing 

def print_results_summary(final_results, ID_THRESH, DYN_THRESH):
    console = Console()
    table = Table(title="Summary of Verification Results")
    table.add_column("Video Window Number", no_wrap=True)
    table.add_column("ID Feat. Hash Distance", no_wrap=True)
    table.add_column("Dynamic Feat. Hash Distance", no_wrap=True)
    for i in range(len(final_results)):
        if final_results[i]["rec_seq_num"] is None:
            rec_seq_num = "N/A"
        else:
            rec_seq_num = str(final_results[i]["rec_seq_num"])

        if final_results[i]["id_dist"] == -1:
            id_dist_str = "N/A"
            id_color = " "
        else:
            id_dist_str = str(final_results[i]["id_dist"])
            if final_results[i]["id_dist"] > ID_THRESH:
                id_color = "[red] [/red]"
            else:
                id_color = "[green] [/green]"

        if final_results[i]["dyn_dist"] == -1:
            dyn_dist_str = "N/A"
            dyn_color = " "
        else:
            dyn_dist_str = str(final_results[i]["dyn_dist"])
            if final_results[i]["dyn_dist"] > DYN_THRESH:
                dyn_color =  "[red] [/red]"
            else:
                dyn_color = "[green] [/green]"
        table.add_row(f"{i}", f"{id_color.split(' ')[0]}{id_dist_str}{id_color.split(' ')[1]}", f"{dyn_color.split(' ')[0]}{dyn_dist_str}{dyn_color.split(' ')[1]}")
    console.print(table)

#####################################
#####   PLOTTING COLORS/LABELS  #####
#####################################
colors = ['#377eb8', "green", "cyan", "red", "orchid", "darkorchid", "crimson", "lime", "fuchsia", "#ff7f00", "#f781bf", "darkcyan", "yellowgreen", "#4daf4a", "cornflowerblue",  "peru"]

blendshape_names = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight"
]

#####################################
####   EMBEDDING VISUALIZATION  #####
#####################################
def vis_window_sync_signal(i, main_sync_signal, bp_main_pred_interwin, plot_title = None, display = False, save_path = None):
    """
    Visualize the synchronization signal for a given window.

    Args:
        i (int): Index of the window to visualize.
        main_sync_signal (list): The main synchronization signal.
        bp_main_pred_interwin (list): List of predicted window start markers.
        plot_title (str, optional): Title for the plot. Defaults to None.
        display (bool, optional): If True, display the plot. Defaults to False.
        save_path (str, optional): If provided, save the plot to this path. Defaults to None.
    """
    fig, axes = plt.subplots(2, tight_layout=True, figsize = (12, 6))
    axes[0].plot(main_sync_signal)
    if i == len(bp_main_pred_interwin) - 1:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[1].plot(main_sync_signal[bp_main_pred_interwin[i] - 20:])
        axes[1].vlines([20], min(main_sync_signal[bp_main_pred_interwin[i] - 20:]), max(main_sync_signal[bp_main_pred_interwin[i] - 20:]), color = 'g', linestyles = 'dashed')
    else:
        axes[0].vlines([bp_main_pred_interwin[i]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        axes[0].vlines([bp_main_pred_interwin[i + 1]], min(main_sync_signal), max(main_sync_signal), color = 'g', linestyle = 'dashed')
        if bp_main_pred_interwin[i] - 20 < 0:
            front_sub = 0
        else:
            front_sub = 20
        if bp_main_pred_interwin[i+1] + 20 > len(bp_main_pred_interwin):
            end_add = 0
        else:
            end_add = 20
        sig = main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1] + end_add]
        axes[1].plot(sig)
        axes[1].vlines([front_sub, len(main_sync_signal[bp_main_pred_interwin[i] - front_sub:bp_main_pred_interwin[i+1]]) - end_add], min(sig), max(sig), color = 'g', linestyles = 'dashed')
    
    if plot_title:
        title = plot_title 
    else:
        title = f"Start Pred Marker {i}"
    plt.suptitle(title, fontsize = 10)

    if display:
        plt.show()

    if save_path:
        plt.savefig(save_path)
    
    plt.close()
    
#####################################
#####   FEATURE VISUALIZATION  ######
#####################################
def annotate_face(frame, face_bbox, detection_result, 
                    landmark_dists = None, landmark_dist_colors = None, draw_mesh = True):
    """"
    Annotate a video frame with face landmarks, blendshapes, and other features.

    Args: 
        frame (numpy array): The video frame to annotate.
        face_bbox (tuple): Bounding box of the face in the format (x1, y1, x2, y2).
        detection_result (mediapipe FaceMesh detection result): The detection result containing landmarks and blendshapes.
        landmark_dists (list, optional): List of landmark distances to visualize. Defaults to None.
        landmark_dist_colors (list, optional): List of colors for each landmark distance. Defaults to None.
        draw_mesh (bool, optional): If True, draw the face mesh on the annotated frame. Defaults to True.

    Returns:
        vis_frame (numpy array): The annotated video frame.
    """
    vis_frame = frame.copy()
    if face_bbox is not None:
        bottom = max(face_bbox[1] - config.initial_bbox_padding, 0)
        top = min(face_bbox[3]+1 + config.initial_bbox_padding, frame.shape[0])
        left = max( face_bbox[0] - config.initial_bbox_padding, 0)
        right = min(face_bbox[2] + 1 + config.initial_bbox_padding, frame.shape[1])
        vis_frame = vis_frame[bottom:top,left:right]
        if draw_mesh:
            vis_frame = draw_landmarks_on_image(vis_frame, detection_result)
            vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

        if landmark_dists is not None:
            face_landmarks_list = detection_result.face_landmarks
            face_landmarks = face_landmarks_list[0] 
            H, W, _ = vis_frame.shape #not the same as self.input_H, self.input_W if initial face detection (and thus cropping) is being used!
            # MediaPipe by deafult returns facial landmark coordinates normalized to 0-1 based on the input frame dimensions. Here we 
            # un-normalize to get coordinates ranging from 0-W/0_H (i.e., actual pixel coordinates)
            landmark_coords = [(landmark.x * W, landmark.y * H, landmark.z) for landmark in face_landmarks] 
            for dist_num, p in enumerate(landmark_dists):
                p1, p2 = p.split("-")
                p1 = int(p1)
                p2 = int(p2)
                x1, y1, _ = landmark_coords[p1]
                x2, y2, _ = landmark_coords[p2]
                c = [chan * 255 for chan in mcolors.to_rgb(landmark_dist_colors[dist_num])[::-1]] # to RGB, then BGR as expected by OpenCV
                cv2.line(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), c, 2)
                
    return vis_frame

def draw_landmarks_on_image(cv_image, detection_result):
  """
  Draw MediaPipe FaceMesh output on the input image.

  Args:
      cv_image (numpy array): The input image to annotate.
      detection_result (mediapipe FaceMesh detection result): The detection result containing landmarks and blendshapes.

  Returns:
      numpy array: The annotated image with landmarks and blendshapes drawn on it.
  """
  cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)  
  rgb_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_image)
  rgb_image = rgb_image.numpy_view()
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

class LegendTitle(object):
    """
    Custom legend with subtitle support for matplotlib.
    https://stackoverflow.com/questions/38463369/subtitles-within-matplotlib-legend
    """
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle,  **self.text_props)
        handlebox.add_artist(title)
        return title


    
def format_hash_str(hash_str, max_chars_per_line=75):
    """
    Formats a hash string into multiple lines with a maximum number of characters per line.

    Args:
        hash_str (str): The hash string to format.
        max_chars_per_line (int): Maximum number of characters per line.
    
    Returns:
        str: The formatted hash string with line breaks.
    """
    if hash_str is None:
        return ""
    lines = [hash_str[i:i + max_chars_per_line] for i in range(0, len(hash_str), max_chars_per_line)]
    return "\n".join(lines)



def merge_audio_video(video_source, audio_source, output_folder_path, output_name):
    """
    Merge audio from one video with visuals from another using ffmpeg-python.
    
    Args:
        video_source (str): Path to the video file (visuals to keep)
        audio_source (str): Path to the video/audio file (audio to use)
        output_folder_path (str): Path to the output folder where the merged video will be saved
        output_name (str): Name of the output video file (without extension)
    """
    # ffmpeg.input(input_video_path).output(output_audio_path, acodec='copy').run()
    ffmpeg.input(audio_source).output(f"{output_folder_path}/audio.aac", acodec='copy', loglevel="quiet").run()
    video = ffmpeg.input(video_source)
    audio = ffmpeg.input(f"{output_folder_path}/audio.aac")
    ffmpeg.output(audio, video, f"{output_folder_path}/{output_name}.mp4", loglevel="quiet").run(overwrite_output = True)
    os.remove(f"{output_folder_path}/audio.aac")  # remove the temporary audio file


def visualize_ver_results(video_path, output_path, ID_THRESH, DYN_THRESH, vis_feature_idx = [0, 6, 7, 10, 11, 14, 15]):
    """
    Visualize the features extracted from a video, including face landmarks, blendshapes, and dynamic features.
    Also show the results of comparison with the hashes extracted from the video's signature.
    Outputs the video to {output_path}/visualization.mp4.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path to the output directory where verify() outputs/results were saved.
        ID_THRESH (float): Threshold for identity feature distance.
        DYN_THRESH (float): Threshold for dynamic feature distance.
        vis_feature_idx (list, optional): Indices of config.target_features to visualize. If None, all features are visualized. 
                                          Defaults to the subset [0, 6, 7, 10, 11, 14, 15] which cover a range of 
                                          features while preventing cluttering of the plot with all 16 examined features.
    """
    print("Generating visualization video...")
    if vis_feature_idx is None:
        target_vis = range(len(config.target_features))
    else:
        target_vis = vis_feature_idx
    landmark_dist_idxs = [f for f in range(len(target_vis)) if type(config.target_features[target_vis[f]]) == str]
    landmark_dist_colors = np.array(colors)[landmark_dist_idxs]
    target_distances = [config.target_features[target_vis[f]] for f in landmark_dist_idxs]
    blendshape_idxs = [f for f in range(len(target_vis)) if type(config.target_features[target_vis[f]]) == int]
    
    with open(f"{output_path}/final_results.pkl", "rb") as pklfile:
        final_results = pickle.load(pklfile)
    with open(f"{output_path}/homography.pkl", "rb") as pklfile:
        Hom = pickle.load(pklfile)
        sorted_contour_centers = pickle.load(pklfile)

    # raw MP results, for annotating the frames
    with open(f"{output_path}/video_signals.pkl", "rb") as pklfile:
        dynamic_features = pickle.load(pklfile)
        poses = pickle.load(pklfile)
        face_bboxes = pickle.load(pklfile)
        raw_detection_results = pickle.load(pklfile)

    # opt_end_frame = opt_start_frame + int(config.video_window_duration*fps+1)
    input_cap = cv2.VideoCapture(video_path)
    fps = input_cap.get(cv2.CAP_PROP_FPS)
    # input_cap.set(cv2.CAP_PROP_POS_FRAMES, opt_start_frame)
    width = int(input_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(input_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(f"{output_path}/visualization_no_audio.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width * 2, height * 2))
    plt.rcParams['font.size'] = 24

    # make raw dynamic features list into a list of signals, one for each. 
    # unlike the above signals, this is for entire video, not just the current window
    raw_signals = [[] for i in range(len(config.target_features))]
    for frame_feats in dynamic_features:
        for i in range(len(config.target_features)):
            raw_signals[i].append(frame_feats[i])
    
    first_win_start = final_results[0]["boundaries"][0]
    last_win_end = final_results[-1]["boundaries"][1]
    curr_win_start, curr_win_end = final_results[0]["boundaries"]
    curr_win_idx = 0
    j = 0
    while True:
        ret, frame = input_cap.read()
        if not ret:
            break
        raw_detection_result = raw_detection_results[j]
        face_bbox = face_bboxes[j]
        annotated_face = annotate_face(frame, face_bbox, raw_detection_result, landmark_dists=target_distances, landmark_dist_colors = landmark_dist_colors, draw_mesh = False)
        # replace face area in original frame with annotated face
        annotated_frame = frame.copy()
        if face_bbox is not None:
            bottom = max(face_bbox[1] - config.initial_bbox_padding, 0)
            top = min(face_bbox[3]+1 + config.initial_bbox_padding, frame.shape[0])
            left = max( face_bbox[0] - config.initial_bbox_padding, 0)
            right = min(face_bbox[2] + 1 + config.initial_bbox_padding, frame.shape[1])
            annotated_frame[bottom:top,left:right] = annotated_face
            # draw rectangle around face
            cv2.rectangle(annotated_frame, (face_bbox[0], face_bbox[1]), (face_bbox[2], face_bbox[3]), (255, 0, 255), 4)
        
        # draw lines connecting the sorted contour centers
        cv2.line(annotated_frame, sorted_contour_centers[0], sorted_contour_centers[1], (255, 0, 255), 4)
        cv2.line(annotated_frame, sorted_contour_centers[1], sorted_contour_centers[3], (255, 0, 255), 4)
        cv2.line(annotated_frame, sorted_contour_centers[0], sorted_contour_centers[2], (255, 0, 255), 4)
        cv2.line(annotated_frame, sorted_contour_centers[2], sorted_contour_centers[3], (255, 0, 255), 4)
        contour_xs = [c[0] for c in sorted_contour_centers]
        contour_ys = [c[1] for c in sorted_contour_centers]
        min_x = min(contour_xs)
        min_y = min(contour_ys)
        cv2.putText(annotated_frame, "Localized Signature", (min_x, min_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

        curr_frame_blendshape_values_proc = []
        blendshape_lines = []
        landmark_lines = []
        blendshape_labels = []
        landmark_labels = []
        dist_count = 0
        for f, s in enumerate(raw_signals):
            if f not in target_vis:
                continue
            proc_s = single_feature_signal_processing(s, resample_signal = False, scaler = "minmax")
            if type(config.target_features[f]) == int:
                label = blendshape_names[config.target_features[f]]
                curr_frame_blendshape_values_proc.append(proc_s[j])
                linestyle = '--'
                linewidth = 2
            else:
                label = f"Landmark pair {dist_count + 1}"#config.target_features[f] 
                linestyle = '-'
                linewidth = 4
                dist_count += 1
            [line] = plt.plot(proc_s[:j + 1],  c = colors[f], linewidth = linewidth, linestyle = linestyle) # plot up to current frame within this window's signals
            if type(config.target_features[f]) == int:
                blendshape_lines.append(line)
                blendshape_labels.append(label)
            else:
                landmark_lines.append(line)
                landmark_labels.append(label)

        # signals line plot
        if j < 50:
            xrange = (0, 100)
        else:
            xrange = (j -  50, j + 50)
        plt.xlim(xrange)
        plt.ylim((-0.05, 1))
        plt.xlabel('Frame Number')
        plt.title("MediaPipe FaceMesh Signals")
        plt.legend(['Landmark Pair Dist.'] + landmark_lines + ['Blendshape'] + blendshape_lines, [''] + landmark_labels + [''] + blendshape_labels, 
                        handler_map={str: LegendTitle({"fontsize" : 24})})
        figure = plt.gcf()
        figure.set_dpi(100)
        figure.set_size_inches(0.01*width*2, 0.01*height) 
        figure.canvas.draw()
        fig_img = np.array(figure.canvas.buffer_rgba())
        fig_img = cv2.cvtColor(fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        plt.close()

        # data report
        fig, ax = plt.subplots()
        if j > first_win_start and j <= last_win_end:
            if final_results[curr_win_idx]["rec_seq_num"] is not None:
                cx = 6
                cy = 14
                ax.annotate(f"Signature Validation", (cx, cy + 1.5), color='black', ha='center', va='center', fontsize = 26)
                ax.annotate(f"Video Window # {curr_win_idx}", (cx, cy + 0.5), color='black', ha='center', va='center') #final_results[curr_win_idx]['rec_seq_num'] - 1 <- extracted signature #
                max_chars_per_line = 75

                # only display the hashes and distances if they are not -1 (i.e., if the extracted one is corrupt)
                if final_results[curr_win_idx]["id_dist"] != -1:
                    id_color = 'green' if final_results[curr_win_idx]["id_dist"] <= ID_THRESH else 'red'
                    id_hash_str = format_hash_str(final_results[curr_win_idx]["id_hash"], max_chars_per_line) # format the hash strings for display
                    rec_id_hash_str = format_hash_str(final_results[curr_win_idx]["rec_id_hash"], max_chars_per_line)
                    ax.annotate("ID Feature Hashes", (cx, cy - 1.5), color='black',  ha='center', va='center')
                    ax.plot([cx - 3, cx + 3], [cy - 1.5 - 0.25, cy - 1.5 - 0.25], color='black', lw=1)
                    # ax.text(cx, cy - 2, r'\underline{ID Feature Hashes}', color='black', usetex=True, horizontalalignment='center', verticalalignment='center')
                    ax.annotate(f"From embedded signature:\n{rec_id_hash_str}", (cx, cy - 3), color='black', ha='center', va='center')
                    ax.annotate(f"From portrayed speech:\n{id_hash_str}", (cx, cy - 5.2), color='black', ha='center', va='center')
                    ax.annotate(f"Distance: {final_results[curr_win_idx]['id_dist']}", (cx, cy - 6.7), color=id_color, ha='center', va='center')
                if final_results[curr_win_idx]["dyn_dist"] != -1:
                    dyn_color = 'green' if final_results[curr_win_idx]["dyn_dist"] <= DYN_THRESH else 'red'
                    dyn_hash_str = format_hash_str(final_results[curr_win_idx]["dyn_hash"], max_chars_per_line)  # format the hash strings for display
                    rec_dyn_hash_str = format_hash_str(final_results[curr_win_idx]["rec_dyn_hash"], max_chars_per_line)
                    ax.annotate("Dynamic Feature Hashes", (cx, cy - 8.5), color='black', ha='center', va='center')
                    ax.plot([cx - 4, cx + 4], [cy - 8.5 - 0.25, cy - 8.5 - 0.25], color='black', lw=1)
                    ax.annotate(f"From embedded signature:\n{rec_dyn_hash_str}", (cx, cy - 10), color='black', ha='center', va='center')
                    ax.annotate(f"From portrayed speech:\n{dyn_hash_str}", (cx, cy - 12.2), color='black', ha='center', va='center')
                    ax.annotate(f"Distance: {final_results[curr_win_idx]['dyn_dist']}", (cx, cy - 13.7), color=dyn_color, ha='center', va='center')
        
        ax.set_xlim((0, 16))
        ax.set_ylim((0, 16))
        ax.set_axis_off()
        ax.set_aspect('equal')
        figure = plt.gcf()
        figure.set_dpi(100)
        figure.set_size_inches(0.01*width, 0.01*height)
        figure.canvas.draw()
        data_fig_img = np.array(figure.canvas.buffer_rgba())
        data_fig_img = cv2.cvtColor(data_fig_img, cv2.COLOR_RGBA2BGR)
        plt.clf()
        plt.close()

        # stack and write
        out_frame = np.vstack((np.hstack((annotated_frame, data_fig_img)), fig_img)) 
        out.write(out_frame)

        j += 1
        # print(curr_win_start, curr_win_end, curr_win_idx)
        if j >= curr_win_end: 
            curr_win_idx += 1
            if curr_win_idx >= len(final_results):
                break
            curr_win_start, curr_win_end = final_results[curr_win_idx]["boundaries"]

    out.release()
    input_cap.release()

    merge_audio_video(f"{output_path}/visualization_no_audio.mp4", video_path, output_path, "visualization")
    os.remove(f"{output_path}/visualization_no_audio.mp4")
    print("Visualization video saved to " + f"{output_path}/visualization.mp4")




