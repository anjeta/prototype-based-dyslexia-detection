# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 20:02:35 2025

@author: aneta.kartali
"""

from matplotlib import pyplot as plt
import numpy as np
import os
import json
import pandas as pd
import porespy as ps
import pickle
from PIL import Image
import sys

from features.build_features import fixation_saccade_data, intersection, fixation_intersection_coefficient, fixation_fractal_dimension, feature_extraction
from params.data_params_Benfatto import data_params


def getReadingData(filepath, verbose=False):
    # returns gaze data for both eyes
    data = json.loads(open(filepath, encoding='utf-8').read())
    lx = data['lx']
    ly = data['ly']
    rx = data['rx']
    ry = data['ry']
    t = data['t']
    
    subject = data['subject']
    task = data['task']
    disease = data['diagnosis']
    
    if data_params.process_seq_names:
        split_char = '/'
    else:
        split_char = '\\'
    
    norm_path = split_char.join(os.path.realpath(filepath).split(split_char)[:-1])
    norm_fname = subject + '-' + task + '-normalization_data.pkl'
    with open(os.path.join(norm_path, norm_fname), "rb") as f:
        norm_data = pickle.load(f)
        
    lx = (lx - norm_data['lx_min']) / (norm_data['lx_max'] - norm_data['lx_min'])
    ly = (ly - norm_data['ly_min']) / (norm_data['ly_max'] - norm_data['ly_min'])
    rx = (rx - norm_data['rx_min']) / (norm_data['rx_max'] - norm_data['rx_min'])
    ry = (ry - norm_data['ry_min']) / (norm_data['ry_max'] - norm_data['ry_min'])
    
    Fs = data['Fs']
    readTime = len(t) / Fs
    t = np.linspace(0, readTime*1000, len(t), 1/Fs*1000)  # Time in milliseconds

    return t, Fs, lx, ly, rx, ry, subject, task, disease

def calcAttsAveragedEyes(filepath, verbose=False):
    
    markers = {}

    if verbose:
        print('ANALYSIS')
        print('  File: %s' % filepath)
        print()

    # separate reading analysis (fake instructions)
    if verbose: print('READING')
    if not filepath:
        if verbose:
            print('  Reading recordings are missing.')
            print()
        markers.update({'active_read_time':'?', 'fixation_intersection_coeff':'?', 
                        'saccade_variability':'?', 'fixation_intersection_variability':'?', 
                        'fixation_fractal_dimension':'?', 'fixation_count':'?',
                        'fixation_total_dur':'?', 'fixation_freq':'?', 'fixation_avg_dur':'?', 
                        'saccade_count':'?', 'saccade_total_dur':'?',
                        'saccade_freq':'?', 'saccade_avg_dur':'?', 'total_read_time': '?'})
    else:
        t, Fs, lx, ly, rx, ry, subject, task, disease = getReadingData(filepath, verbose=False)
        
        x = np.mean(np.stack((lx, rx), axis=0), axis=0)
        y = np.mean(np.stack((ly, ry), axis=0), axis=0)
        
        success, active_read_time, fixation_intersection_coeff, saccade_variability, fixation_intersection_variability, fixation_fractal_dimension, fixation_count, fixation_total_dur, fixation_freq, fixation_avg_dur, saccade_count, saccade_total_dur, saccade_freq, saccade_avg_dur, total_read_time = feature_extraction(t, x, y, Fs)
        markers.update({'active_read_time':active_read_time, 'fixation_intersection_coeff':fixation_intersection_coeff, 
                        'saccade_variability':saccade_variability, 'fixation_intersection_variability':fixation_intersection_variability, 
                        'fixation_fractal_dimension':fixation_fractal_dimension, 'fixation_count':fixation_count,
                        'fixation_total_dur':fixation_total_dur, 'fixation_freq':fixation_freq, 'fixation_avg_dur':fixation_avg_dur, 
                        'saccade_count':saccade_count, 'saccade_total_dur':saccade_total_dur,
                        'saccade_freq':saccade_freq, 'saccade_avg_dur':saccade_avg_dur, 'total_read_time': total_read_time})

        return success, markers, subject, task, disease

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    
    manifest_path = data_params.manifest_path
    splits = ["train", "valid"]
    readings = []
    
    for split in splits:
        manifest = os.path.join(manifest_path, "{}.tsv".format(split))
            
        seqNames = []
        with open(manifest, "r") as f:
            files_path = f.readline().strip()
            for line in f:
                seqNames.append(line.strip().split("\t")[0])
        
        for i in range(len(seqNames)):
            file = seqNames[i]
            filepath = os.path.join(files_path, file)
            success, markers_tmp, subject, task, disease = calcAttsAveragedEyes(filepath, True)
            if success:
                markers = markers_tmp
            else:
                markers = {feature: 0.0 for feature in markers_tmp}
            updict = {"File": file, "Subject" : subject, "Task" : task, 'Disease': disease}
            updict.update(markers)
            reading = pd.DataFrame.from_dict(updict, orient='index').T
            readings.append(reading)                
        
    df = pd.concat(readings, axis=0, ignore_index=False)
    df.to_csv(os.path.join(manifest_path, f'Segmented-{data_params.time_context}s-AttsAveragedEyes.csv'), index=False)

if __name__ == "__main__":
    main()