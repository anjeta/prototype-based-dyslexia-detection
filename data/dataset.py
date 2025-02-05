# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:42:21 2024

@author: aneta.kartali

This module contains functions used for loading eye-tracking data for training
using PyTorch.

"""

import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import scipy.io
import scipy.signal
import scipy.stats
import torch
from torch.utils.data import Dataset
import sys

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
 
class EyeDataset(Dataset):
    
    def __init__(self, params, criterion, split, process_seq_filenames=False, logger=None):
        super(EyeDataset, self).__init__()
        self.params = params
        self.split = split
        self.criterion = criterion
        if self.params.standardize_data:
            self.mean = scipy.io.loadmat(self.params.norm_dir + 'train_set_mean_val.mat')['mean_val']
            self.std = scipy.io.loadmat(self.params.norm_dir + 'train_set_std_val.mat')['std_val']
        if self.params.use_handcrafted_features:
            self.data, self.features, self.labels = self._load_data(process_seq_filenames, logger)
        if self.params.data_info:
            self.data, self.labels, self.subjects, self.tasks, self.files = self._load_data(process_seq_filenames, logger)
        else:
            self.data, self.labels = self._load_data(process_seq_filenames, logger)
    
    def _get_labels(self, label):
        if self.criterion == 'BCE_with_logits':
            label = torch.tensor(label).float()
            label = label.unsqueeze(0)
        if self.criterion == 'CE':
            label = torch.tensor(label, dtype=torch.long)
        else:
            label = torch.tensor(label).float()
        return label
        
    def __getitem__(self, index):
        # One can measure the time needed to get a batch of data
        # start_time = time.perf_counter()
        data = self.data[index]
        
        data = torch.tensor(data, dtype=torch.float32)
        # elapsed = time.perf_counter() - start_time
        # curr_time = datetime.datetime.now()
        # print(f"{curr_time} | INFO | train_model | data utils | Indexing list while loading took {1000.0 * elapsed} ms or {elapsed} s")
        
        if self.params.use_handcrafted_features:
            features = self.features[index]
            features = torch.tensor(features, dtype=torch.float32)
        if self.params.data_info:
            info = [self.subjects[index], self.tasks[index], self.files[index]]
        # start_time = time.perf_counter()
        # Get labels info
        if self.params.objective == "classification":
            label = self.labels[index]
            label = self._get_labels(label)
        elif self.params.objective == "reconstruction":
            label = data
        else:
            print(f"Invalid training objective : {self.params.objective}")
            sys.exit()
        # elapsed = time.perf_counter() - start_time
        # curr_time = datetime.datetime.now()
        # print(f"{curr_time} | INFO | train_model | data utils | Fetching data attributes took {1000.0 * elapsed} ms or {elapsed} s")
        if self.params.use_handcrafted_features:
            return (data, features, label)
        if self.params.data_info:
            return (data, info, label)
        else:
            return (data, label)
    
    def __len__(self):
        return len(self.labels)
    
    def _standardizeData(self, x, y):
        x = (x - self.mean[0, 0]) / self.std[0, 0]
        y = (y - self.mean[1, 0]) / self.std[1, 0]
        return x, y
    
    def _load_data(self, process_seq_names=False, logger=None):
            # Load a given dataset split filenames and create a dataset.
            
            manifest = os.path.join(self.params.manifest_path, "{}.tsv".format(self.split))
            
            seq_filenames = []
            with open(manifest, "r") as f:
                files_path = f.readline().strip()
                for line in f:
                    seq_filenames.append(line.strip().split("\t")[0])
            
            curr_time = datetime.datetime.now()
            if logger is not None:
                logger.info(f'{curr_time} | INFO | data_utils | Found {len(seq_filenames)} {self.split} sequences')
            
            # Process sequence names if needed
            if process_seq_names:
                split_char = '/'
                for i in range(len(seq_filenames)):
                    seqName = seq_filenames[i]
                    seqName = seqName.replace("\\","/")
                    seq_filenames[i] = seqName
            else:
                split_char = '\\'
            
            # Load the dataset into memory ------------------------------------        
            data = []
            files = []
            labels = []
            subjects = []
            tasks = []
            if self.params.use_handcrafted_features == True:
                feature_data = pd.read_csv(self.params.features_filename)
                columns = ['Subject', 'Task', 'Disease', 'active_read_time', 'fixation_intersection_coeff', 
                                'saccade_variability', 'fixation_intersection_variability', 
                                'fixation_fractal_dimension', 'fixation_count',
                                'fixation_total_dur', 'fixation_freq', 'fixation_avg_dur', 
                                'saccade_count', 'saccade_total_dur',
                                'saccade_freq', 'saccade_avg_dur', 'total_read_time']
                feature_columns = columns[3:]
                features = []
            
            for i in range(len(seq_filenames)):
                file = seq_filenames[i]
                file_path = os.path.join(files_path, file)
                file_data = json.loads(open(file_path, encoding='utf-8').read())
                subject = file_data['subject']
                task = file_data['task']
                lx = file_data['lx']
                ly = file_data['ly']
                rx = file_data['rx']
                ry = file_data['ry']
                
                norm_path = split_char.join(os.path.realpath(file_path).split(split_char)[:-1])
                norm_fname = subject + '-' + task + '-normalization_data.pkl'
                with open(os.path.join(norm_path, norm_fname), "rb") as f:
                    norm_data = pickle.load(f)
                    
                lx = (lx - norm_data['lx_min']) / (norm_data['lx_max'] - norm_data['lx_min'])
                ly = (ly - norm_data['ly_min']) / (norm_data['ly_max'] - norm_data['ly_min'])
                rx = (rx - norm_data['rx_min']) / (norm_data['rx_max'] - norm_data['rx_min'])
                ry = (ry - norm_data['ry_min']) / (norm_data['ry_max'] - norm_data['ry_min'])
                
                if self.params.left_right_average:
                    x = np.mean(np.stack((lx, rx), axis=0), axis=0)
                    y = np.mean(np.stack((ly, ry), axis=0), axis=0)
                    if self.params.standardize_data:
                        x, y  = self._standardize_data(x, y)
                    if self.params.x_axis_only:
                        pos_data = np.array(x, ndmin=2)
                    else:
                        pos_data = np.stack((x, y), axis=0)
                else:
                    if self.params.x_axis_only:
                       pos_data = np.stack((lx, rx), axis=0)
                    else:
                       pos_data = np.stack((lx, rx, ly, ry), axis=0)
                if self.params.use_speed:
                    # dt = 1/hp.Fs
                    # speed_data = np.abs(np.gradient(data, dt, axis=1))
                    speed_data = np.abs(np.gradient(pos_data, axis=1))
                    data.append(speed_data)
                    # data.append(np.concatenate((pos_data, speed_data)))
                else:
                    data.append(pos_data)
                if self.params.use_handcrafted_features:
                    feature = feature_data.loc[feature_data['File'] == file]
                    features.append(np.array(feature[feature_columns]))
                label = file_data['label']
                labels.append(label)
                subjects.append(subject)
                tasks.append(task)
                files.append(file)
            
            if self.params.use_handcrafted_features:
                return data, features, labels
            if self.params.data_info:
                return data, labels, subjects, tasks, files
            else:
                return data, labels


class Subset(Dataset):
    
    def __init__(self, dataset, indices):
        super(Subset, self).__init__()
        self.dataset = dataset
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]