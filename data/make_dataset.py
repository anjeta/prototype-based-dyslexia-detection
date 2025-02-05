# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 12:10:16 2024

@author: aneta.kartali

This script is used for segmenting EYE-TRACKING files for downstream 
classification task.
More info in "Screening for Dyslexia Using Eye Tracking during Reading" 
by Benfatto et al.: 
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0165508

"""

import glob
import json
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from scipy.interpolate import interp1d

from params.data_params_Benfatto import data_params

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------

def LT(x, low, high, reverse=False):

    """
    Counts how many samples from x are in an interval. Returns also the ratio of NaNs.
    """
    not_nan = np.count_nonzero(~np.isnan(x))
    if not_nan == 0:
        return 0.0, 1.0
    if reverse:
        x = 1-x

    return np.count_nonzero((x > low) & (x <= high))/not_nan, 1.0 - not_nan/len(x)

def MISS(x, max_nan = 0):

    """
    Counts missing values but only gaps that are bigger than max_nan.
    Takes as input an array of booleans 0 - missing, 1 - valid.
    """

    # Only sum gaps in missing values that are bigger than a certain threshold a
    idx = np.where(np.concatenate(([1], x, [1]), axis=0))  # Finds the ones
    d = np.diff(idx) - 1   # Gaps in valid values
    d[d<max_nan] = 0
    return np.sum(d)/len(x)

def interpolateData(t, lx, ly, rx, ry):
    
    # Use linear interpolation to handle zero (missing) values
    t_new = np.arange(len(t))
    
    idx = np.nonzero(lx)
    interp = interp1d(t_new[idx], np.array(lx)[idx], fill_value='extrapolate')
    lx = interp(t_new)
    
    idx = np.nonzero(ly)
    interp = interp1d(t_new[idx], np.array(ly)[idx], fill_value='extrapolate')
    ly = interp(t_new)
    
    idx = np.nonzero(rx)
    interp = interp1d(t_new[idx], np.array(rx)[idx], fill_value='extrapolate')
    rx = interp(t_new)
    
    idx = np.nonzero(ry)
    interp = interp1d(t_new[idx], np.array(ry)[idx], fill_value='extrapolate')
    ry = interp(t_new)
    
    return lx.tolist(), ly.tolist(), rx.tolist(), ry.tolist()

def standardizeData(lx, ly, rx, ry, mean_x, mean_y, std_x, std_y):
    lx = (lx - mean_x) / std_x
    ly = (ly - mean_y) / std_y
    rx = (rx - mean_x) / std_x
    ry = (ry - mean_y) / std_y
    
    return lx, ly, rx, ry    

def getReadingData(filename, task, use_tasks, logger=None):
    # returns gaze data for both eyes
    ds = pd.read_csv(filename, sep="\t")

    t = list(ds['T'])
    lx = list(pd.Series([element.replace(',', '.') for element in ds['LX']]).astype(dtype='float64'))  # data for left eye
    ly = list(pd.Series([element.replace(',', '.') for element in ds['LY']]).astype(dtype='float64'))  # data for left eye
    rx = list(pd.Series([element.replace(',', '.') for element in ds['RX']]).astype(dtype='float64'))  # data for right eye   
    ry = list(pd.Series([element.replace(',', '.') for element in ds['RY']]).astype(dtype='float64'))  # data for right eye 
                
    # Use linear interpolation to handle zero (missing) values
    # lx, ly, rx, ry = interpolateData(t, lx, ly, rx, ry)
    
    return t, lx, ly, rx, ry

def segmentReadingData(t, lx, ly, rx, ry, file_path, subject, diagnosis, task=None, thrStart=0.05, thrEnd=0.10, time_context=5, time_overlap=0, logger=None):
    """
    thrStart and thrEnd are the amount (given as percentage) of data to cut off from the start and end of data
    windowLen: length of the segmentation window in seconds
    overlap: overlapping of the segmentation windows
    """

    low  = round(thrStart * len(t))              # lower cutoff index (start of data)
    high = round(len(t) - thrEnd * len(t)) - 1   # upper cutoff index (end of data)
    
    if high == len(t):
        high -= 1
    
    success = False
    readTime = 0

    if high > 0:
        # LEFT EYE
        lx_r = np.array(lx[low:high])
        ly_r = np.array(ly[low:high])
        # RIGHT EYE
        rx_r = np.array(rx[low:high])
        ry_r = np.array(ry[low:high])
        # TIME
        t_r = np.array(t[low:high])
        
        seqLen = len(t_r)  # sequence length in samples
        
        readTime = (t[high] - t[low]) / 1000   # reading time in seconds (for the time between both thresholds)
        if logger is not None: 
            logger.info(f'  Reading time: {readTime} [s]')
    
        Fs = np.round(1000 / np.median(np.diff(t)))
        assert Fs == data_params.Fs, f"Specified sampling rate {data_params.Fs} does not match true sampling rate {Fs}"
        time_context = int(time_context * Fs)
        
        success = True
        seg_num = 0
        
        lx_n = np.zeros_like(lx_r)
        ly_n = np.zeros_like(ly_r)
        rx_n = np.zeros_like(rx_r)
        ry_n = np.zeros_like(ry_r)
        
        try:
            for start in range(0, seqLen, int(time_context*(1-time_overlap))):
                if (start + time_context) >= seqLen:
                    break
                
                seg_num += 1
                
                # Segment the eye-tracking data
                lx_s = lx_r[start : start + time_context].tolist()
                ly_s = ly_r[start : start + time_context].tolist()
                rx_s = rx_r[start : start + time_context].tolist()
                ry_s = ry_r[start : start + time_context].tolist()
                t_s = t_r[start : start + time_context].tolist()
                
                blink_dur = 0.15 * Fs  # Average blink duration is 100 ms
                
                # Calculate the number of missing (0) samples in the data
                lx_missing = MISS(((np.array(lx_s) - np.min(lx_r)) > 0.1) * 1.0, max_nan=blink_dur)
                ly_missing = MISS(((np.abs(np.array(ly_s) - np.max(ly_r))) > 0.1) * 1.0, max_nan=blink_dur)
                rx_missing = MISS(((np.array(rx_s) - np.min(rx_r)) > 0.1) * 1.0, max_nan=blink_dur)
                ry_missing = MISS(((np.abs(np.array(ry_s) - np.max(ry_r))) > 0.1) * 1.0, max_nan=blink_dur)
                missing = np.mean((lx_missing, ly_missing, rx_missing, ry_missing))
                # If more than 10% of data samples are missing skip this segment
                if missing > 0.1:
                    continue
                
                # Calculate the number of valid ([0, 1]) samples in the data
                # lx_valid, _ = LT(np.array([np.nan if i==0 else i for i in lx_s]), 0, 1)
                # ly_valid, _ = LT(np.array([np.nan if i==0 else i for i in ly_s]), 0, 1)
                # rx_valid, _ = LT(np.array([np.nan if i==0 else i for i in rx_s]), 0, 1)
                # ry_valid, _ = LT(np.array([np.nan if i==0 else i for i in ry_s]), 0, 1)
                # valid = np.mean((lx_valid, ly_valid, rx_valid, ry_valid))
                # # If more than 10% of data samples are out of range [0,1] skip this segment
                # if valid < 0.9:
                #     continue
                # Replace out of range values with 0 and interpolate them later
                lx_s = [0 if (i - np.min(lx_r)) < 0.1 else i for i in lx_s]
                ly_s = [0 if np.abs(i - np.max(ly_r)) < 0.1 else i for i in ly_s]
                rx_s = [0 if (i - np.min(rx_r)) < 0.1 else i for i in rx_s]
                ry_s = [0 if np.abs(i - np.max(ry_r)) < 0.1 else i for i in ry_s]
                
                lx_s, ly_s, rx_s, ry_s = interpolateData(t_s, lx_s, ly_s, rx_s, ry_s)
                
                lx_n[start : start + time_context] = lx_s
                ly_n[start : start + time_context] = ly_s
                rx_n[start : start + time_context] = rx_s
                ry_n[start : start + time_context] = ry_s
                
                segment = {}
                segment["subject"] = subject
                segment["diagnosis"] = diagnosis
                segment["task"] = task
                segment["label"] = data_params.class_mapping[diagnosis]
                segment["lx"] = lx_s
                segment["ly"] = ly_s
                segment["rx"] = rx_s
                segment["ry"] = ry_s
                segment["t"] = t_s
                segment["missing"] = missing
                segment["Fs"] = Fs
                # segment["non_valid"] = 1 - valid
                segment["segment_number"] = seg_num
                
                subject_path = subject
                
                if (seg_num < 10):
                    segment_path = '0' + str(seg_num)
                else:
                    segment_path = str(seg_num) 
                    
                # write JSON object to file
                file_name = subject_path + '-' + task + '-' + segment_path + '.json'
                with open(os.path.join(file_path, file_name), "w") as outfile: 
                    json.dump(segment, outfile)
        except:
            logger.info('  Data segmentation unsuccessful.')
            success = False
    
    if success:
        if np.any(lx_n) and np.any(ly_n) and np.any(rx_n) and np.any(ry_n):
            lx_n = lx_n[lx_n != 0]
            lx_min = np.min(lx_n)
            lx_max = np.max(lx_n)
        
            ly_n = ly_n[ly_n != 0]
            ly_min = np.min(ly_n)
            ly_max = np.max(ly_n)
            
            rx_n = rx_n[rx_n != 0]
            rx_min = np.min(rx_n)
            rx_max = np.max(rx_n)
            
            ry_n = ry_n[ry_n != 0]
            ry_min = np.min(ry_n)
            ry_max = np.max(ry_n)
        
            file_name = subject_path + '-' + task + '-normalization_data.pkl'
            with open(os.path.join(file_path, file_name), "wb") as f:
                norm_data = {'lx_min': lx_min, 'lx_max': lx_max, 'ly_min': ly_min, 
                              'ly_max': ly_max, 'rx_min': rx_min, 'rx_max': rx_max,
                              'ry_min': ry_min, 'ry_max': ry_max}
                pickle.dump(norm_data, f)
            
    return success, readTime
    
# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # -------------------------------------------------------------------------
    # Load EYE-TRACKING data and segment it
    # -------------------------------------------------------------------------
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')

    if data_params.process_seq_names:
        split_char = '/'
    else:
        split_char = '\\'

    data_path = os.path.join(data_params.data_path)
    dir_path = os.path.realpath(data_path)
    search_path = os.path.join(dir_path, data_params.dataset, "*\*\*." + data_params.data_ext) 
        
    readTimes = []
    start_end = pd.read_excel(os.path.join(dir_path, data_params.dataset, data_params.processing_file),sheet_name='Sheet1')
    
    logger.info('Segmenting eye-tracking data')
        
    for fname in glob.iglob(search_path, recursive=True):
            
        if fname.split(split_char)[-2] not in list(start_end['Subject ID']):
            continue
            
        file_path = split_char.join(os.path.realpath(fname).split(split_char)[:-1])
            
        task = fname.split(split_char)[-1].split('.')[0]
        if task not in data_params.use_tasks:
            continue
        
        logger.info('  Processing: %s' % fname)
        
        diagnosis = fname.split(split_char)[-3]
        subject = fname.split(split_char)[-2]
        
        t, lx, ly, rx, ry = getReadingData(fname, task, data_params.use_tasks, logger=logger)
            
        segment_path = os.path.join(data_path, data_params.segmented_dataset, diagnosis, subject, task)
        path = Path(segment_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            
        thrStart = int(start_end[start_end['Subject ID']==subject]['Trial start'].values[0]) / len(t)
        thrEnd = 1 - (int(start_end[start_end['Subject ID']==subject]['Trial end'].values[0]) + 1) / len(t)
        
        success, readingTime = segmentReadingData(t, lx, ly, rx, ry, segment_path, subject, diagnosis, task, thrStart=thrStart, thrEnd=thrEnd, time_context=data_params.time_context, time_overlap=data_params.time_overlap, logger=logger)
    
        if success:
            readTimes.append(readingTime)
        else:
            logger.info('  No reading found.')
    
    logger.info(f'  Total reading time: {np.sum(readTimes)} [s]')
    logger.info(f'  Mean reading time: {np.mean(readTimes)} [s]')
    logger.info(f'  Standard deviation of the reading time: {np.std(readTimes)} [s]')
    logger.info(f'  Median reading time: {np.median(readTimes)} [s]')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=data_params.logging_path + "make_dataset_Benfatto_log_6s_50%.log", 
                        filemode='a', encoding='utf-8', level=logging.INFO, format=log_fmt)
    
    main()
    