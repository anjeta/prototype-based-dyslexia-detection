# -*- coding: utf-8 -*-
"""
Created on Tue May 21 17:46:22 2024

@author: aneta.kartali
"""

import glob
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from make_dataset import MISS, getReadingData
from params.data_params_Benfatto import data_params

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    
    # -------------------------------------------------------------------------
    # Load EYE-TRACKING data and segment it
    # -------------------------------------------------------------------------
    logger = None
    
    if data_params.process_seq_names:
        split_char = '/'
    else:
        split_char = '\\'
        
    norm_data = {}
    data_path = os.path.join(data_params.data_path)
    dir_path = os.path.realpath(data_path)
    search_path = os.path.join(dir_path, data_params.segmented_dataset, "*\*\*\*." + data_params.norm_ext)
    
    for fname in glob.iglob(search_path, recursive=True):
        subject = fname.split(split_char)[-3]
        with open(fname, "rb") as f:
            norm_data[subject] = pickle.load(f)

    data_path = os.path.join(data_params.data_path)
    dir_path = os.path.realpath(data_path)
    search_path = os.path.join(dir_path, data_params.dataset, "*\*\*." + data_params.data_ext) 
        
    start_end = pd.read_excel(os.path.join(dir_path, data_params.dataset, data_params.processing_file),sheet_name='Sheet1')
    
    fileData = {'subject':[], 'diagnosis': [], 'task': [], 'nfix_per_sec': [], 
                'fix_dur': [], 'forw_dur': [], 'backw_dur': [],
                'reading_time': [], 'missing': [], 'valid': []}
                # 'forw_speed': [], 'backw_speed': []}
        
    readTimes = []
    missingData = []
    validData = []
    
    fixationData = {'nfix_per_sec': [], 'fix_dur': []}
    saccadeData = {'forw_dur': [], 'backw_dur': [],'forw_speed': [], 'backw_speed': []}
        
    for fname in glob.iglob(search_path, recursive=True):
            
        if fname.split(split_char)[-2] not in list(start_end['Subject ID']):
            continue
            
        file_path = split_char.join(os.path.realpath(fname).split(split_char)[:-1])
            
        task = fname.split(split_char)[-1].split('.')[0]
        if task not in data_params.use_tasks:
            continue
        
        diagnosis = fname.split(split_char)[-3]
        subject = fname.split(split_char)[-2]
        
        if subject in norm_data.keys():
        
            t, lx, ly, rx, ry = getReadingData(fname, task, data_params.use_tasks, logger=logger)
            
            fileData['subject'].append(subject)
            fileData['diagnosis'].append(diagnosis)
            fileData['task'].append(task)
            
            thrStart = int(start_end[start_end['Subject ID']==subject]['Trial start'].values[0]) / len(t)
            thrEnd = 1 - (int(start_end[start_end['Subject ID']==subject]['Trial end'].values[0]) + 1) / len(t)
            
            low  = int(start_end[start_end['Subject ID']==subject]['Trial start'].values[0])     # lower cutoff index (start of data)
            high = int(start_end[start_end['Subject ID']==subject]['Trial end'].values[0]) - 1   # upper cutoff index (end of data)
            
            if high == len(t):
                high -= 1
            
            success = False
            readTime = 0
    
            if high > 0:
                # LEFT EYE
                lx = np.array(lx[low:high])
                ly = np.array(ly[low:high])
                # RIGHT EYE
                rx = np.array(rx[low:high])
                ry = np.array(ry[low:high])
                # TIME
                readingTime = (t[high] - t[low]) / 1000   # reading time in seconds (for the time between both thresholds)
                fileData['reading_time'].append(readingTime)
                t = np.array(t[low:high])
                
                if logger is not None: 
                    logger.info(f'  Reading time: {readTime} [s]')
            
                Fs = np.round(1000 / np.median(np.diff(t)))
                assert Fs == data_params.Fs, f"Specified sampling rate {data_params.Fs} does not match true sampling rate {Fs}"
    
                blink_dur = 0.15 * Fs
                
                # Calculate the number of missing (0) samples in the data
                lx_missing = MISS(((np.array(lx) - np.min(lx)) > 0.1) * 1.0, max_nan=blink_dur)
                ly_missing = MISS(((np.abs(np.array(ly) - np.max(ly))) > 0.1) * 1.0, max_nan=blink_dur)
                rx_missing = MISS(((np.array(rx) - np.min(rx)) > 0.1) * 1.0, max_nan=blink_dur)
                ry_missing = MISS(((np.abs(np.array(ry) - np.max(ry))) > 0.1) * 1.0, max_nan=blink_dur)
                missing = np.mean((lx_missing, ly_missing, rx_missing, ry_missing))
                fileData['missing'].append(missing)
                
                # Calculate the number of valid ([0, 1]) samples in the data
                valid = 1.0 
                fileData['valid'].append(valid)
    
    save_path = data_params.manifest_path + "dataset-statistics/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(save_path + "Dataset statistics.txt", "w") as f:
        f.write(f'Mean reading time: {np.mean(fileData["reading_time"])} [s]\n')
        f.write(f'Standard deviation of the reading time: {np.std(fileData["reading_time"])} [s]\n')
        f.write(f'Median reading time: {np.median(fileData["reading_time"])} [s]\n')
        f.write(f'Number of recordings whith more that 10% of missing data samples: {np.sum((np.array(fileData["missing"])>0.1)*1.0)}\n')
        f.write(f'Number of recordings whith more that 10% of invalid data samples: {np.sum((np.array(fileData["valid"])<0.9)*1.0)}\n')
        
    fig, ax = plt.subplots(dpi=600)
    plt.hist(np.array(fileData['missing']))
    ax.set_xlim([0, 1])
    plt.xlabel('Percentage of missing data [%]')
    plt.ylabel('Number of recordings')
    plt.title('Percentage of missing data in individual recordings')
    plt.grid()
    plt.savefig(save_path + "Percentage of missing data in individual recordings.png" )
    plt.show()
    
    fig, ax = plt.subplots(dpi=600)
    plt.hist(np.array(fileData['valid']))
    ax.set_xlim([0, 1])
    plt.xlabel('Percentage of valid data [%]')
    plt.ylabel('Number of recordings')
    plt.title('Percentage of valid data in individual recordings')
    plt.grid()
    plt.savefig(save_path + "Percentage of valid data in individual recordings.png" )
    plt.show()