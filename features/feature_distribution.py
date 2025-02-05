# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 17:00:20 2024

@author: aneta.kartali

This script is used for calculating training set distribution based on
hand-crafted features of eye-tracking data.

"""

import datetime
import glob
import json
import numpy as np
import os
from pathlib import Path
import pickle
import scipy.io
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from features.build_features import feature_extraction
import data.params
from data.params.data_params_Benfatto import data_params

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

split = 'train'
# This folder contains train, valid and test tsv files produced by data_manifest.py script
manifest_path = data_params.manifest_path
# Statistics for standardizing images if standardize_data == True
save_dir = manifest_path + "feature-distribution/"

process_seq_names = False

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    manifest = os.path.join(manifest_path, "{}.tsv".format('train'))

    seqNames = []
    with open(manifest, "r") as f:
        files_path = f.readline().strip()
        for line in f:
            seqNames.append(line.strip().split("\t")[0])
    
    curr_time = datetime.datetime.now()
    print(f'{curr_time} | INFO | generate_train_statistics | Found {len(seqNames)} {split} sequences')

    # Process sequence names if needed
    if process_seq_names:
        for i in range(len(seqNames)):
            seqName = seqNames[i]
            seqName = seqName.replace("\\","/")
            seqNames[i] = seqName
            
    class_names = ["HEALTHY", "DYSLEXIA"]
    
    active_read_times = {"HEALTHY": [], "DYSLEXIA": []}
    fixation_intersection_coeffs = {"HEALTHY": [], "DYSLEXIA": []} 
    saccade_variabilities = {"HEALTHY": [], "DYSLEXIA": []}
    fixation_intersection_variabilities = {"HEALTHY": [], "DYSLEXIA": []}
    fixation_fractal_dimensions = {"HEALTHY": [], "DYSLEXIA": []} 
    fixation_counts = {"HEALTHY": [], "DYSLEXIA": []}
    fixation_total_durs = {"HEALTHY": [], "DYSLEXIA": []}
    fixation_freqs = {"HEALTHY": [], "DYSLEXIA": []} 
    fixation_avg_durs = {"HEALTHY": [], "DYSLEXIA": []}
    saccade_counts = {"HEALTHY": [], "DYSLEXIA": []}
    saccade_total_durs = {"HEALTHY": [], "DYSLEXIA": []}
    saccade_freqs = {"HEALTHY": [], "DYSLEXIA": []}
    saccade_avg_durs = {"HEALTHY": [], "DYSLEXIA": []}
    total_read_times = {"HEALTHY": [], "DYSLEXIA": []}
    
    path = Path(save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
    norm_data = {}
    data_path = os.path.join(data_params.data_path)
    dir_path = os.path.realpath(data_path)
    search_path = os.path.join(dir_path, data_params.segmented_dataset, "*\*\*\*." + data_params.norm_ext)
    
    if data_params.process_seq_names:
        split_char = '/'
    else:
        split_char = '\\'
    
    for fname in glob.iglob(search_path, recursive=True):
        if 'task' not in fname.split(split_char)[-1]:
            subject = fname.split(split_char)[-3]
            with open(fname, "rb") as f:
                norm_data[subject] = pickle.load(f)
    
    for i in range(len(seqNames)):
        file = seqNames[i]
    
        file_path = os.path.join(files_path, file)
        
        file_data = json.loads(open(file_path, encoding='utf-8').read())
        lx = file_data['lx']
        ly = file_data['ly']
        rx = file_data['rx']
        ry = file_data['ry']
        label = file_data['label']
        subject = file_data['subject']
        Fs = file_data['Fs']
        
        lx = (lx - norm_data[subject]['lx_min']) / (norm_data[subject]['lx_max'] - norm_data[subject]['lx_min'])
        ly = (ly - norm_data[subject]['ly_min']) / (norm_data[subject]['ly_max'] - norm_data[subject]['ly_min'])
        rx = (rx - norm_data[subject]['rx_min']) / (norm_data[subject]['rx_max'] - norm_data[subject]['rx_min'])
        ry = (ry - norm_data[subject]['ry_min']) / (norm_data[subject]['ry_max'] - norm_data[subject]['ry_min'])
        
        x = np.mean(np.stack((lx, rx), axis=0), axis=0)
        y = np.mean(np.stack((ly, ry), axis=0), axis=0)
        readTime = len(x) / Fs
        t = np.linspace(0, readTime*1000, len(x), 1/Fs*1000)  # Time in milliseconds
        
        # Calculate features --------------------------------------------------
        success, active_read_time, fixation_intersection_coeff, saccade_variability, fixation_intersection_variability, fixation_fractal_dimension, fixation_count, fixation_total_dur, fixation_freq, fixation_avg_dur, saccade_count, saccade_total_dur, saccade_freq, saccade_avg_dur, total_read_time = feature_extraction(t, x, y, Fs)
                
        active_read_times[class_names[label]].append(active_read_time)
        fixation_intersection_coeffs[class_names[label]].append(fixation_intersection_coeff)
        saccade_variabilities[class_names[label]].append(saccade_variability)
        fixation_intersection_variabilities[class_names[label]].append(fixation_intersection_variability)
        fixation_fractal_dimensions[class_names[label]].append(fixation_fractal_dimension) 
        fixation_counts[class_names[label]].append(fixation_count)
        fixation_total_durs[class_names[label]].append(fixation_total_dur)
        fixation_freqs[class_names[label]].append(fixation_freq)
        fixation_avg_durs[class_names[label]].append(fixation_avg_dur)
        saccade_counts[class_names[label]].append(saccade_count)
        saccade_total_durs[class_names[label]].append(saccade_total_dur)
        saccade_freqs[class_names[label]].append(saccade_freq)
        saccade_avg_durs[class_names[label]].append(saccade_avg_dur)
        total_read_times[class_names[label]].append(total_read_time)
        
    mdict = {'active_read_times': active_read_times, 
             'fixation_intersection_coeffs': fixation_intersection_coeffs, 
             'saccade_variabilities': saccade_variabilities,
             'fixation_intersection_variabilities': fixation_intersection_variabilities, 
             'fixation_fractal_dimensions': fixation_fractal_dimensions, 
             'fixation_counts': fixation_counts,
             'fixation_total_durs': fixation_total_durs, 
             'fixation_freqs': fixation_freqs,
             'fixation_avg_durs': fixation_avg_durs, 
             'saccade_counts': saccade_counts,
             'saccade_total_durs': saccade_total_durs,
             'saccade_freqs': saccade_freqs,
             'saccade_avg_durs': saccade_avg_durs,
             'total_read_times': total_read_times,
             }
    
    fname = save_dir + split + '_feature_distribution.pkl'
    
    with open(fname, 'wb') as f:
        pickle.dump(mdict, f)