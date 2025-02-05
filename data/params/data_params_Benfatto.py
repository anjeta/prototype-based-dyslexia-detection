# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:01:09 2024

@author: aneta.kartali
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

data_params = {
    'dataset': "Benfatto-Dyslexia",
    'processing_file': "Benfatto et al - trial start and end.xlsx",
    'segmented_dataset': "Benfatto-Dyslexia-segmented-6s-overlap-20%",
    'data_path': "C:/Users/aneta.kartali/Documents/PhD/Research/Eye-tracking/Datasets",
    'logging_path': "../logs/",
    'data_ext': "txt",
    'segmented_data_ext': "json",
    'norm_ext': "pkl",
    'manifest_path': "../data/data_manifest/Benfatto/train-valid-test_6s/classification/leave-k-out/",
    'process_seq_names': False,  # False for Windows and True for Linux
    
    'use_tasks': ['A1R'],
    'Fs': 50,
    'time_context': 6,  # [s] Number of seconds to segment the data
    'time_overlap': 0.2,
    
    'train_valid_test_partition': "leave-k-out",
    'train_percent': 0.8,
    'valid_percent': 0.1,
    'test_percent': 0.1,
    'class_mapping': {'bp': 0, 'dys': 1},
    'balance_data': False,  # Whether to balance the dataset by forcing the equal number of examples per diagnosis
    
    'objective': "classification",
    }

params={'data_params':data_params}
segment_path = os.path.join(data_params['data_path'], data_params['segmented_dataset'])
path = Path(segment_path)
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)
with open(os.path.join(path, 'data_params.json'), 'w') as file:
    json.dump(params, file)
    
data_params = SimpleNamespace(**data_params)