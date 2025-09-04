"""
Adaptation process running in parallel alongside projection

Hadleigh Schwartz - Columbia University
Last updated 8/9/2025

© 2025 The Trustees of Columbia University in the City of New York.  
This work may be reproduced, distributed, and otherwise exploited for academic non-commercial purposes only.  
To obtain a license to use this work for commercial purposes, 
please contact Columbia Technology Ventures at techventures@columbia.edu.
"""
import pickle
import numpy as np
from colorama import Fore, Back

import common.config as config
from common.error_correctors import SoftViterbi, ConcatenatedViterbiRS
from common.decode_sequence import decode_sequence
from common.decoding_utils import get_nonloc_cell_signal_from_imgseq, get_localization_signal_from_imgseq, get_normalized_mag_at_freq, valid_r_c

from embedding.embedding_utils import adaptive_log, hom_images, get_imgseq_info, get_imgseq
from embedding.perceptibility_utils import get_on_off_cell_val_from_cell_signal, CIEDE2000

def get_slm_bgr(ref_bgr, min_tot_slm = 80):
    """
    Given BGR of target patch (<ref_bgr>), determine the SLM BGR by scaling BGR channels
    equally until their sum (and thus intensity of light outputted by SLM)
    is at least <min_tot_slm>

    Args:
        ref_bgr (list): The reference BGR values.
        min_tot_slm (int): The minimum total SLM intensity value.

    Returns:
        list: The adjusted SLM bitmap BGR values.
    """
    s  = min_tot_slm / (ref_bgr[0] + ref_bgr[1] + ref_bgr[2])
    return [ref_bgr[0]*s,ref_bgr[1]*s,ref_bgr[2]*s]
    

def adapt(ad_queue, cell_colors_mins_queue, ready_adapt_queue): 
    """
    Main adaptation process function, running in parallel alongside projection.
    
    Args:
        ad_queue (multiprocessing.Queue): Queue to receive adaptation parameters.
        cell_colors_mins_queue (multiprocessing.Queue): Queue to send updated cell colors and minimum SLM values.
        ready_adapt_queue (multiprocessing.Queue): Queue to signal readiness for next adaptation.
    """

    f_adaptation = open(f"{config.session_output_path}/adaptation_logging.csv", "w")
    f_adaptation.write("Seq,r,c,Ref B,Ref G,Ref R,Target B,Target G,Target B,Perceptibility,Curr Tot Min SLM,Target Tot Min SLM,Hard Decoding Errors,Soft Decoding Errors,Loc Mag,Loc Mag Ratio,Mean Pilot Mag\n")

    ready_adapt_queue.put(True)
    while True:
        adapt_params_input = ad_queue.get()

        if adapt_params_input == None:
            break

        last_disp_start_time , last_disp_end_time, last_disp_seq_num, Hom, last_disp_payload, last_disp_rs_encoded, image_timestamps, fpss, curr_cell_min_tot_slms, curr_loc_marker_min_tot_slms = adapt_params_input

        fps, start_image_timestamp_i, end_image_timestamp_i = get_imgseq_info(last_disp_start_time, last_disp_end_time, image_timestamps, fpss) 
        adaptive_log(f"Adapting using window {last_disp_seq_num}data.", "INFO", flags = [Fore.BLUE])
        # , with core unit FPS of {fps}", "INFO", flags = [Fore.BLUE])

        # apply homography to raw images in the sequence, and then generate the sequence as an array 
        hom_images(Hom, start_image_timestamp_i, end_image_timestamp_i)
        image_sequence = get_imgseq(start_image_timestamp_i, end_image_timestamp_i) 

        # get channel cell signals and magnitudes (magnitudes for localization magnitude reference)
        adaptive_log(f"Getting channel pilot cell signals and mags for {last_disp_seq_num}.", "DEBUG", flags = [Fore.BLUE])
        
        # decode cell signals, getting both hard values and probabilities. also get on and off cell BGR values, for use in perceptibilty adaptation
        tot_hard_pred, tot_probs, tot_gt, pilot_mags, all_perceptibilities, all_off_cell_bgrs = decode_sequence(image_sequence, fps, get_perceptibilities = True, gts = last_disp_payload, pkl_path_prefix = f"{config.session_output_path}/signal_pkls/seq{last_disp_seq_num}", display = False)
        
        # perform soft Viterbie decoding from probabilities obtained above to recover the payload, still RS-encoded
        # ignore the boolean returned by decode_payload indicating or not the sequence was actually correctable, because 
        # if it was not, this will be reflected in the calculated error rate based on the ground truth
        if type(config.error_corrector) == SoftViterbi:
            pred_previterbi_encoded, _ = config.error_corrector.decode_payload(tot_probs[:config.viterbi_signature_size])# rs here is a misnomer but it's ok
        elif type(config.error_corrector) == ConcatenatedViterbiRS:
            if len(tot_probs) < config.viterbi_signature_size:
                for i in range(config.viterbi_signature_size - len(tot_probs)):
                    tot_probs.append(0)
                print("Padded.", len(tot_probs))
            _, pred_previterbi_encoded, _ = config.error_corrector.decode_payload(tot_probs[:config.viterbi_signature_size])
        else:
            print("Error corrector not supported.")
        soft_decoding_err = sum(c1!=c2 for c1,c2 in zip(pred_previterbi_encoded, last_disp_rs_encoded)) 
        hard_decoding_err = sum(c1!=c2 for c1,c2 in zip(tot_gt, tot_hard_pred))

        # adapt all channel pilots + info cells based on errors, any localization border cells based on mag
        adaptive_log(f"Adapting cell colors for {last_disp_seq_num}.", "INFO", flags = [Fore.BLUE])
        pilot_mags = np.array(pilot_mags)
        mean_pilot_mag = np.mean(pilot_mags)
        new_min_tot_slms = []
        new_cell_colors = []
        overall_cell_num = 0 
        for r in range(config.max_cells_H):
            for c in range(config.max_cells_W): 
                if not valid_r_c(r, c, config.max_cells_W, config.max_cells_H, config.localization_N):
                    continue

                if f"{r}-{c}" in config.reserved_localization_cells:
                    # this is a localization border cell. adap based on mag
                    cell_signal = get_nonloc_cell_signal_from_imgseq(image_sequence, r, c)
                    with open(f"{config.session_output_path}/signal_pkls/seq{last_disp_seq_num}_loc_r{r}_c{c}.pkl", "wb") as pklfile:
                        pickle.dump(cell_signal, pklfile)
                    
                    cell_mag = get_normalized_mag_at_freq(cell_signal, fps)
                    localization_ratio = cell_mag / mean_pilot_mag # the idea here is that the localization cells should have a higher mag at the localizaiton freq than the rest of the scene. 
                    # we can use the pilot cell mags to estimate this, and the ratio of the two to indicate the "goodness" of the localization signal.
                    if localization_ratio < config.min_norm_localization_ratio:
                        target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num] - config.min_tot_slm_dec
                    else:
                        if all_perceptibilities[overall_cell_num] > config.max_perceptibility:
                            target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num]  - config.min_tot_slm_dec
                        else:
                            target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num] 
                        
                    new_min_tot_slms.append(target_min_tot_slm)
                    target_slm_bgr = get_slm_bgr(all_off_cell_bgrs[overall_cell_num], min_tot_slm = target_min_tot_slm)
                    new_cell_colors.append(target_slm_bgr)
                    
                    f_adaptation.write(f"{last_disp_seq_num},{r},{c},{all_off_cell_bgrs[overall_cell_num][0]},{all_off_cell_bgrs[overall_cell_num][1]},{all_off_cell_bgrs[overall_cell_num][2]},{target_slm_bgr[0]},{target_slm_bgr[1]},{target_slm_bgr[2]},{all_perceptibilities[overall_cell_num]},{curr_cell_min_tot_slms[overall_cell_num]},{target_min_tot_slm},-,-,{cell_mag},{localization_ratio},{mean_pilot_mag}\n")
                else:
                    # this is a channel border cell or info cell
                    if soft_decoding_err > config.max_soft_decoding_err:
                        target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num] + config.min_tot_slm_inc #always increment if BER too high
                        new_min_tot_slms.append(target_min_tot_slm)
                    else:
                        if all_perceptibilities[overall_cell_num] > config.max_perceptibility:
                            target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num] - config.min_tot_slm_dec  #decrement if BER ok and perecptibility too high
                        else:
                            target_min_tot_slm = curr_cell_min_tot_slms[overall_cell_num]#if BER and perceptibility ok, leave as is
                    
                    new_min_tot_slms.append(target_min_tot_slm) 
                    target_slm_bgr = get_slm_bgr(all_off_cell_bgrs[overall_cell_num], min_tot_slm = target_min_tot_slm)
                    new_cell_colors.append(target_slm_bgr)
                    f_adaptation.write(f"{last_disp_seq_num},{r},{c},{all_off_cell_bgrs[overall_cell_num][0]},{all_off_cell_bgrs[overall_cell_num][1]},{all_off_cell_bgrs[overall_cell_num][2]},{target_slm_bgr[0]},{target_slm_bgr[1]},{target_slm_bgr[2]},{all_perceptibilities[overall_cell_num]},{curr_cell_min_tot_slms[overall_cell_num]},{target_min_tot_slm},{hard_decoding_err}/{config.num_info_cells*config.frequency*config.embedding_window_duration},{soft_decoding_err}/{len(pred_previterbi_encoded)},-,-,-\n")
   
                overall_cell_num += 1
        
        # if using special localization corners, get those signals + perceptibilities, to adapt. otherwise leave this blank
        if config.localization_N is not None:
            new_loc_marker_min_tot_slms = []
            new_loc_marker_colors = []
            for i in range(4):
                loc_marker_signal, cell_boundaries = get_localization_signal_from_imgseq(image_sequence, i, config.target_channel, config.slm_W, config.slm_H, config.N, config.buffer_space, config.localization_N, config.max_cells_W, config.max_cells_H)
                with open(f"{config.session_output_path}/signal_pkls/seq{last_disp_seq_num}_loc{i}.pkl", "wb") as pklfile:
                    pickle.dump(loc_marker_signal, pklfile)
        
                off_marker_bgr, on_marker_bgr = get_on_off_cell_val_from_cell_signal(loc_marker_signal, image_sequence, cell_boundaries) #output_name = f"seq{last_disp_seq_num}_r{r}_c{c}_minmax" 
                
                perceptibility = CIEDE2000(off_marker_bgr[::-1], on_marker_bgr[::-1])

                loc_marker_mag = get_normalized_mag_at_freq(loc_marker_signal, fps)
                localization_ratio = loc_marker_mag / mean_pilot_mag
                if localization_ratio < config.min_norm_localization_ratio:
                    target_min_tot_slm = curr_loc_marker_min_tot_slms[i] + config.min_tot_slm_inc
                else:
                    if perceptibility > config.max_perceptibility:
                        target_min_tot_slm = curr_loc_marker_min_tot_slms[i] - config.min_tot_slm_dec
                    else:
                        target_min_tot_slm = curr_loc_marker_min_tot_slms[i] 
                      
                target_slm_bgr = get_slm_bgr(off_marker_bgr, min_tot_slm = target_min_tot_slm)
                new_loc_marker_colors.append(target_slm_bgr)
                new_loc_marker_min_tot_slms.append(target_min_tot_slm)

                f_adaptation.write(f"{last_disp_seq_num},loc{i},-,{off_marker_bgr[0]},{off_marker_bgr[1]},{off_marker_bgr[2]},{target_slm_bgr[0]},{target_slm_bgr[1]},{target_slm_bgr[2]},{perceptibility},{curr_loc_marker_min_tot_slms[i]},{target_min_tot_slm},-,-,{loc_marker_mag},{localization_ratio},{mean_pilot_mag}\n")
        else:
            new_loc_marker_colors = None
            new_loc_marker_min_tot_slms = None

        cell_colors_mins_queue.put((new_cell_colors, new_loc_marker_colors, new_min_tot_slms, new_loc_marker_min_tot_slms, soft_decoding_err, hard_decoding_err, last_disp_seq_num ))
        
        adaptive_log(f"Total soft decoding errors for seq{last_disp_seq_num}: {soft_decoding_err}/{len(pred_previterbi_encoded)} bits. Hard decoding errors: {hard_decoding_err}/{len(tot_gt)}", "INFO", flags = [Fore.MAGENTA])
        ready_adapt_queue.put(True)
