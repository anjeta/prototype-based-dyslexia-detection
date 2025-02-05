# -*- coding: utf-8 -*-
"""
Created on Wed May 29 09:27:58 2024

@author: aneta.kartali
"""

from copy import deepcopy
import datetime
import importlib
import json
import logging
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, WeightedRandomSampler, DataLoader
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data.dataset import EyeDataset, Subset
from visualization.visualizations import visualize_logs

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
model_architecture = 'eyecnn'
# 'eyetcn', 'eyecnn', 'protoeyetcn'
pretrained_architecture = 'eyetcn'
# 'eyetcn'

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------
def initialize_model(stage, model_params, train_params, log_params, logger, 
                     model_architecture, Model, PretrainedModel=None):
    
    # Create loss function ----------------------------------------------------
    if model_params.criterion == 'BCE':
        criterion = nn.BCELoss()
    elif model_params.criterion == 'BCE_with_logits':
        criterion = nn.BCEWithLogitsLoss()
    elif model_params.criterion == 'CE':
        criterion = nn.CrossEntropyLoss()
    else:
        logger.info(f"ERROR: Unsupported loss function: {model_params.criterion}")
        sys.exit()
    
    regularizer = None
    if model_params.L1 != 0:
        regularizer = nn.L1Loss(size_average=False)
        
    # Instantiate and initiate the model --------------------------------------
    if 'proto' in model_architecture:
        model = Model(train_params.device, model_params, criterion, regularizer, stage, logger)
    else:
        model = Model(train_params.device, model_params, criterion, regularizer, logger) 
        
    logger.info("\n")
    logger.info("Model architecture:")
    logger.info(model.encoder)
    if 'proto' in model_architecture:
        logger.info(model.prototype_layer)
    logger.info(model.decoder)
    logger.info("\n")
    
    total_params, trainable_params = model.get_parameter_count()
    logger.info(f"Total number of model parameters: {total_params}")
    logger.info(f"Number of trainable model parameters: {trainable_params}")
    logger.info("\n")
    
    # Create logs -------------------------------------------------------------
    logs = {"epoch": [], "iter": [],
           'train_loss': [], 'valid_loss': [], 'train_acc': [], 'valid_acc': [],
           'epoch_train_loss': [],'epoch_valid_loss': [],'epoch_train_acc': [],
           'epoch_valid_acc': [], 'test_loss': [], 'test_acc': [],
           }
        
    # Load model checkpoint ---------------------------------------------------
    if log_params.load_checkpoint:
        
        if stage > 0 and PretrainedModel is None:
            logger.info(f"ERROR: Pretrained Model architecture cannot be none at stage {stage}")
            sys.exit()
        
        if stage == 0:
            if 'proto' in model_architecture:
                model.load_model(os.path.join(log_params.checkpoint_path, log_params.checkpoint_name), Model)
            else:
                model.load_model(os.path.join(log_params.checkpoint_path, log_params.checkpoint_name))
            with open(os.path.join(log_params.checkpoint_path, 'checkpoint_logs.json'), 'rb') as file:
                logs = json.load(file)
        if stage == 1:
            pretrained = PretrainedModel(train_params.device, model_params, criterion, regularizer, logger)
            pretrained.load_encoder(os.path.join(log_params.checkpoint_path, log_params.pretrained_checkpoint_name))
            model.encoder = pretrained.encoder
        else:
            pretrained = PretrainedModel(train_params.device, model_params, criterion, regularizer, stage, logger)
            pretrained.load_model(os.path.join(log_params.checkpoint_path, log_params.warmup_checkpoint_name), PretrainedModel)
            model = pretrained
    
    return model, logs


def save_logs(data, logs_path, logs_filename):
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    with open(logs_path + logs_filename, 'w') as file:
        json.dump(data, file, indent=2)


def run(trainLoader, validLoader, 
        Model, train_params, log_params, logs, logger):

    curr_time = datetime.datetime.now()
    logger.info(f"{curr_time} | INFO | train_model | Training for {train_params.num_epochs} epochs")
    logger.info(f"{curr_time} | INFO | train_model | Training dataset {len(trainLoader)} batches | Validation dataset {len(validLoader)} batches")
    
    start_epoch = len(logs["epoch"])
    best_acc = 0  # Initialize best (highest) accuracy on validation set
    best_loss = 1000  # Initialize best (lowest) loss on validation set
    start_time = time.time()
        
    # Model training ----------------------------------------------------------
    for epoch in range(start_epoch, train_params.num_epochs):

        curr_time = datetime.datetime.now()
        logger.info(f"{curr_time} | INFO | train_model | Starting epoch {epoch+1}")
        
        finetuning = False
        if train_params.finetuning and epoch <= train_params.freeze_finetuning_epochs:
            finetuning = True

        # Training ------------------------------------------------------------
        train_loss, train_acc, train_logs = Model.train_epoch(epoch, trainLoader, finetuning)
        curr_time = datetime.datetime.now()
        logger.info(f"{curr_time} | INFO | train_model | One epoch training | loss: {train_loss} | acc: {train_acc}")
        
        # Validation ----------------------------------------------------------
        valid_loss, valid_acc, valid_logs = Model.eval_epoch(epoch, validLoader, finetuning)
        curr_time = datetime.datetime.now()
        logger.info(f"{curr_time} | INFO | train_model | One epoch validation | loss: {valid_loss} | acc: {valid_acc}")
        
        # Saving logs ---------------------------------------------------------
        curr_time = datetime.datetime.now()
        logger.info(f"{curr_time} | INFO | train_model | Ran {epoch + 1} epochs in {time.time() - start_time:.2f} seconds")
        
        torch.cuda.empty_cache()

        current_acc = float(valid_acc)
        current_loss = float(valid_loss)
        if current_acc > best_acc:
            best_acc = current_acc
            logs["best_valid_acc"] = best_acc
            logs["loss_at_best_valid_acc"] = current_loss
            logs["epoch_at_best_valid_acc"] = epoch + 1
        if current_loss < best_loss:
            best_loss = current_loss
            logs["best_valid_loss"] = best_loss
            logs["acc_at_best_valid_loss"] = current_acc
            logs["epoch_at_best_valid_loss"] = epoch + 1

        curr_time = datetime.datetime.now()
        logger.info(f"{curr_time} | INFO | train_model | Saving training logs")
        
        for key, value in dict(train_logs, **valid_logs).items():
            if key not in logs:
                logs[key] = [None for x in range(epoch)]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logs[key].append(value)

        logs["epoch"].append(epoch+1)
        logs["epoch_train_loss"].append(train_loss)
        logs["epoch_train_acc"].append(train_acc)
        logs["epoch_valid_loss"].append(valid_loss)
        logs["epoch_valid_acc"].append(valid_acc)
        save_logs(logs, log_params.results_path, "checkpoint_logs.json")
        
        if log_params.results_path is not None and ((epoch+1) % train_params.save_step == 0 or (epoch+1) == train_params.num_epochs):
            if train_params.save_step != -1 and epoch != 0:
                Model.save_model(log_params.results_path + f"checkpoint_{epoch+1}.pt")
            save_logs(logs, log_params.results_path, "checkpoint_logs.json")
            
        if (epoch+1) == int(train_params.num_epochs / 2):
            eval_logs = Model.get_training_metrics(validLoader)
            for key, value in dict(eval_logs).items():
                if key not in logs:
                    logs[key] = [None for x in range(epoch)]
                if isinstance(value, np.ndarray):
                    value = value.tolist()
                logs[key].append(value)
            save_logs(logs, log_params.results_path, "checkpoint_logs.json")
            
    eval_logs = Model.get_training_metrics(validLoader)
    for key, value in dict(eval_logs).items():
        if key not in logs:
            logs[key] = [None for x in range(epoch)]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key].append(value)
    save_logs(logs, log_params.results_path, "checkpoint_logs.json")
    
    visualize_logs(logs, log_params)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    
    set_spawn_method = False  # For multiprocessing on GPU training - True when using multiple GPUs
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Load modules and params -------------------------------------------------
    module = importlib.import_module(f'model.{model_architecture}')
    Model = module.Model
    params = importlib.import_module(f'model.params.{model_architecture}_cv_params')
    
    model_params = params.model_params
    data_params = params.data_params
    log_params = params.log_params
    train_params = params.train_params
    
    # Create logger -----------------------------------------------------------
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    fname = f"{model_architecture}_cv_training_log.log"
    logging.basicConfig(filename=log_params.results_path + fname, 
                        encoding='utf-8', level=logging.INFO, format=log_fmt)
    
    logger = logging.getLogger(__name__)
    logger.info(f'Training model {model_architecture}')
    
    # Use CUDA if available ---------------------------------------------------
    if set_spawn_method:
        torch.multiprocessing.set_start_method('spawn', force=True)
    num_GPU = torch.cuda.device_count()  # Get number of GPUs
    logger.info(f"Found {num_GPU} GPUs.")
    logger.info(f"Let's use {num_GPU} GPUs!")
    batch_size = num_GPU * model_params.batch_size_GPU
    
    # Load the data -----------------------------------------------------------
    start_time_loading = time.perf_counter()
    
    trainDataset = EyeDataset(data_params, model_params.criterion, 'train', data_params.process_seq_names, logger)
    validDataset = EyeDataset(data_params, model_params.criterion, 'valid', data_params.process_seq_names, logger)
    testDataset = EyeDataset(data_params, model_params.criterion, 'test', data_params.process_seq_names, logger)
    
    elapsed = time.perf_counter() - start_time_loading
    curr_time = datetime.datetime.now()
    logger.info(f"{curr_time} | INFO | train_model | main | Loading all data to CPU took {1.0 * elapsed} s")
    
    # Initiate K-fold cross validation ----------------------------------------
    dataset = ConcatDataset([trainDataset, validDataset, testDataset])
    
    X =  np.arange(len(dataset)) 
    y = [item for dtst in dataset.datasets for item in dtst.labels]
    groups = [item for dtst in dataset.datasets for item in dtst.subjects]
    group_labels = [item for dtst in dataset.datasets for item in dtst.labels]
    results_path = log_params.results_path
    
    if train_params.LOSOCV:
        tmp_groups = []
        idx = 1
        for i in range(len(groups)):
            tmp_groups.append(idx)
            if i != len(groups)-1:
                if groups[i+1] != groups[i]:
                    idx += 1
        groups = np.array(tmp_groups)
        kfold = LeaveOneGroupOut()
    else:
        kfold = StratifiedGroupKFold(n_splits=train_params.num_folds, 
                                     shuffle=train_params.shuffle, 
                                     random_state=train_params.random_state
                                     )
    
    # Initialize model and run cross-validation -------------------------------
    if pretrained_architecture is None:
        model, logs = initialize_model(0, model_params, train_params, log_params, 
                                       logger, model_architecture, Model)
        
        # Run K-fold cross validation -----------------------------------------
        for fold, (train_indices, valid_indices) in enumerate(kfold.split(X, y, groups)):
            curr_time = datetime.datetime.now()
            logger.info(f"{curr_time} | INFO | train_model | main | Training fold {fold + 1}")
            trainDataset = Subset(dataset, train_indices)
            validDataset = Subset(dataset, valid_indices)
            
            # Create data loaders ---------------------------------------------
            if train_params.balanced_training:
                target = [int(label.detach().cpu().numpy()[0]) for _, _, label in trainDataset]
                class_sample_count = np.unique(target, return_counts=True)[1]
                weight = 1. / class_sample_count
                samples_weight = weight[target]
                
                samples_weight = torch.from_numpy(samples_weight)
                sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
                
                trainLoader = DataLoader(dataset=trainDataset, drop_last=False, batch_size=batch_size, num_workers=data_params.num_workers, sampler=sampler)
            
            else:
                trainLoader = DataLoader(dataset=trainDataset, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=data_params.num_workers)
            
            validLoader = DataLoader(dataset=validDataset, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=data_params.num_workers)
            
            log_params.results_path = results_path + "fold_" + str(fold+1) + "/"
            run(trainLoader, validLoader, deepcopy(model), train_params, log_params, deepcopy(logs), logger)
    
    else:
        module = importlib.import_module(f'model.{pretrained_architecture}')
        PretrainedModel = module.Model
        pretrain_model_params = params.pretrain_model_params
        pretrain_train_params = params.pretrain_train_params
        pretrain_log_params = params.pretrain_log_params
        
        # Run K-fold cross validation ---------------------------------------------
        for fold, (train_indices, valid_indices) in enumerate(kfold.split(X, y, groups)):
            
            logger.info(f"Fold {fold+1}")
            logger.info(f"Training subjects: {np.array(groups)[train_indices]}")
            train_labels = np.array(group_labels)[train_indices]
            logger.info(f"Percentage of positive cases in the training set: {np.sum(train_labels==1)/len(train_labels)}")
            logger.info(f"Validation subjects: {np.array(groups)[valid_indices]}")
            valid_labels = np.array(group_labels)[valid_indices]
            logger.info(f"Percentage of positive cases in the validation set: {np.sum(valid_labels==1)/len(valid_labels)}")
            
            curr_time = datetime.datetime.now()
            logger.info(f"{curr_time} | INFO | train_model | main | Training fold {fold + 1}")
            trainDataset = Subset(dataset, train_indices)
            validDataset = Subset(dataset, valid_indices)
            
            trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, shuffle=True, drop_last=False, batch_size=batch_size, num_workers=data_params.num_workers)
            validLoader = torch.utils.data.DataLoader(dataset=validDataset, shuffle=False, drop_last=False, batch_size=batch_size, num_workers=data_params.num_workers)
            
            # Stage 0: Pre-training -----------------------------------------------
            logger.info(f"{curr_time} | INFO | train_model | main | Stage 0: Pretraining")
            model, logs = initialize_model(0, pretrain_model_params, pretrain_train_params, pretrain_log_params,
                                            logger, pretrained_architecture, PretrainedModel)
            pretrain_log_params.results_path = results_path + "fold_" + str(fold+1) + "/stage_0/"
            pretrain_train_params.save_step = 100
            pretrain_train_params.num_epochs = 100
            run(trainLoader, validLoader, deepcopy(model), pretrain_train_params, pretrain_log_params, deepcopy(logs), logger)
            
            # Stage 1: Warm-up ----------------------------------------------------
            logger.info(f"{curr_time} | INFO | train_model | main | Stage 1: Warm-up")
            log_params.checkpoint_path = pretrain_log_params.results_path
            log_params.results_path = results_path + "fold_" + str(fold+1) + "/stage_1/"
            train_params.save_step = 10
            train_params.num_epochs = 10
            model, logs = initialize_model(1, model_params, train_params, log_params,
                                            logger, model_architecture, Model, PretrainedModel)
            run(trainLoader, validLoader, deepcopy(model), train_params, log_params, deepcopy(logs), logger)
            
            # Stages 2 + 3 + 4: 
            # Joint training, prototype projection and last layer optimization ----
            logger.info(f"{curr_time} | INFO | train_model | main | Stages 2, 3, 4: Joint training, Prototype projection, Last layer optimization")
            log_params.checkpoint_path = log_params.results_path
            log_params.results_path = results_path + "fold_" + str(fold+1) + "/stages_2_3_4/"
            train_params.save_step = 200
            train_params.num_epochs = 200
            model, logs = initialize_model(2, model_params, train_params, log_params,
                                            logger, model_architecture, Model, Model)
            run(trainLoader, validLoader, deepcopy(model), train_params, log_params, deepcopy(logs), logger)

if __name__ == "__main__":    
    main()