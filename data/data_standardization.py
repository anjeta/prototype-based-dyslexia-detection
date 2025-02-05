# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:25:15 2024

@author: aneta.kartali

This script is used for calculating training set global statistics based on
eye-tracking data.

"""

import datetime
import json
import logging
import numpy as np
import os
import scipy.io
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from params.data_params_Benfatto import data_params

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

split = 'train'
# This folder contains train, valid and test tsv files produced by data_manifest.py script
manifest_path = data_params.manifest_path
# Statistics for standardizing images if standardize_data == True
save_path = manifest_path + "standardization-data/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

process_seq_names = False

# -----------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------

def main():
    
    logger = logging.getLogger(__name__)
    logger.info('Generating dataset statistics from the training data')
    
    manifest = os.path.join(manifest_path, "{}.tsv".format('train'))

    seqNames = []
    with open(manifest, "r") as f:
        files_path = f.readline().strip()
        for line in f:
            seqNames.append(line.strip().split("\t")[0])
    
    curr_time = datetime.datetime.now()
    logger.info(f'{curr_time} | INFO | generate_train_statistics | Found {len(seqNames)} {split} sequences')

    # Process sequence names if needed
    if process_seq_names:
        for i in range(len(seqNames)):
            seqName = seqNames[i]
            seqName = seqName.replace("\\","/")
            seqNames[i] = seqName
    
    data = []
    for i in range(len(seqNames)):
        file = seqNames[i]
    
        file_path = os.path.join(files_path, file)
        
        file_data = json.loads(open(file_path, encoding='utf-8').read())
        lx = file_data['lx']
        ly = file_data['ly']
        rx = file_data['rx']
        ry = file_data['ry']
        x = np.mean(np.stack((lx, rx), axis=0), axis=0)
        y = np.mean(np.stack((ly, ry), axis=0), axis=0)
        data.append(np.stack((x, y), axis=0))
        
    data = np.transpose(np.stack(data, axis=0), axes=[1, 0, 2])
    data = data.reshape(data.shape[0], (data.shape[1] * data.shape[2]))
    
    mean_val = np.mean(data, axis=1, keepdims=True)
    std_val = np.std(data, axis=1, keepdims=True)
    
    # Saving normalization statistics for spectrogram
    mdict = {"mean_val": mean_val}
    os.makedirs(save_path, exist_ok=True)
    fname = save_path + split + '_set_mean_val.mat'
    scipy.io.savemat(fname, mdict)

    mdict = {"std_val": std_val}
    os.makedirs(save_path, exist_ok=True)
    fname = save_path + split + '_set_std_val.mat'
    scipy.io.savemat(fname, mdict)


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename="make_dataset.log", encoding='utf-8', level=logging.INFO, format=log_fmt)
    
    main()
