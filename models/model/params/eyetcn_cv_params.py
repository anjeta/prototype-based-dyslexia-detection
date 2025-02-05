# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:39:47 2024

@author: aneta.kartali
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace

dataset = 'Benfatto'
time_context = 6  # [s] sequence length
Fs = 50  # [Hz] sample rate

model_params = {
    'batch_size_GPU': 32,
    'lr': 0.001,
    'scheduler': False,
    'optimizer': 'adamw',
    'L2': 0.005,
    'L1': 0,
    'momentum': 0.9,
    'criterion': "BCE_with_logits",
    'clip_gradients': False,
    'clip': 1.0,
    'input_seq_len': time_context * Fs,
    'output_size': 1,
    'num_input_channels': 2,
    'encoder_embedding_sizes': [32, 32, 32],
    'encoder_kernel_size': 3,
    'encoder_stride': 1,
    'encoder_dilations': [[1, 1], [2, 4], [8, 12]],
    'encoder_dropout': 0.2,
    'encoder_causal': False,
    'encoder_norm': "batch_norm",
    'encoder_activation': "relu",
    'encoder_init': "xavier_uniform",
    'encoder_skip_connections': True,
    'use_handcrafted_features': False,  # DOESN'T WORK PROPERLY NOW!
    'num_handcrafted_features': 11,
    'data_info': True,
    'feature_aggregation': "pooling",
    'num_decoder_layers': 2,
    'decoder_dropout': 0.3,
    'decoder_activation': "relu",
    'decoder_init': "xavier_uniform",
    }

train_params = {
    'device': "cuda",
    'save_step': 50,
    'num_epochs': 100,
    'finetuning': False,
    'freeze_finetuning_epochs': 100,
    'balanced_training': False,
    'num_folds': 10,
    'shuffle': True,
    'random_state': 1234567,
    }

data_params = {
    'num_workers': 8,
    'manifest_path': f"../data/data_manifest/{dataset}/train-valid-test_{time_context}s/classification/leave-k-out/",
    'process_seq_names': False,  # False for Windows and True for Linux
    'left_right_average': True,
    'x_axis_only': False,
    'use_speed': False,
    'standardize_data': False,
    'norm_dir': f"../data/data_manifest/{dataset}/train-valid-test_{time_context}s/classification/leave-k-out/dataset-statistics/",
    'use_handcrafted_features': False,  # USING FEATURES DOESN'T WORK PROPERLY NOW!
    'features_filename': f"../data/data_manifest/{dataset}/train-valid-test_{time_context}s/classification/leave-k-out/Segmented-{time_context}s-AttsJoinedEyes.csv",
    'data_info': True,
    'objective': "classification",
    }

log_params = {
    'load_checkpoint': False,
    'checkpoint_path': f"../results/{dataset}/{time_context}s_segments/eyetcn/cross_validation/",
    'checkpoint_name': "checkpoint_100.pt",
    'results_path': f"../results/{dataset}/{time_context}s_segments/eyetcn/cross_validation/",
    }

params={'model_params':model_params, 
        'train_params':train_params, 
        'data_params':data_params, 
        'log_params':log_params
        }

save_path = f"../results/{dataset}/{time_context}s_segments/eyetcn/cross_validation/"
path = Path(save_path)
if not path.exists():
    path.mkdir(parents=True, exist_ok=True)
with open(os.path.join(path, 'eyetcn_cv_params.json'), 'w') as file:
    json.dump(params, file)
    
model_params = SimpleNamespace(**model_params)
train_params = SimpleNamespace(**train_params)
data_params = SimpleNamespace(**data_params)
log_params = SimpleNamespace(**log_params)