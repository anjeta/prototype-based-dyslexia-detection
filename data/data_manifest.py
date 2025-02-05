# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:54:28 2024

@author: aneta.kartali

Helper script for dividing data into train, valid and test set.

Tested by: Aneta

"""

import glob
import logging
import numpy as np
import os
import random

from params.data_params_Benfatto import data_params


def main():

    root = data_params.data_path  # Directory containing jpeg files to index
    dest = data_params.manifest_path  # Directory for saving generated .tsv files
    
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(filename=dest + "data_manifest_log.log", encoding='utf-8', level=logging.INFO, format=log_fmt)
    
    logger = logging.getLogger(__name__)
    logger.info('Making final data set from raw data')
    
    dataset = data_params.segmented_dataset
    
    seed = 42  # random seed
    
    assert data_params.train_percent >= 0 and data_params.train_percent <= 1.0
    assert data_params.valid_percent >= 0 and data_params.valid_percent <= 1.0
    assert data_params.test_percent >= 0 and data_params.test_percent <= 1.0
    assert data_params.test_percent <= data_params.valid_percent  # test data partition cannot be larger than validation partition
    assert data_params.train_percent >= (data_params.valid_percent + data_params.test_percent)
    
    if not os.path.exists(dest):
        os.makedirs(dest)
        
    dir_path = os.path.realpath(root)
    search_path = os.path.join(dir_path, dataset, "*\*\*\*." + data_params.segmented_data_ext)
    rand = random.Random(seed)
    
    if data_params.process_seq_names:
        split_char = '/'
    else:
        split_char = '\\'
    
    # -------------------------------------------------------------------------
    # Count the number of subjects, trials and examples for each dataset used
    # -------------------------------------------------------------------------
    dataset_info = {}
    
    for fname in glob.iglob(search_path, recursive=True):
        file_path = os.path.realpath(fname)
        
        dataset = fname.split(split_char)[-5]
        diagnosis = fname.split(split_char)[-4]
        subject = fname.split(split_char)[-3]
        task = fname.split(split_char)[-2]
        
        if not dataset in dataset_info:
            dataset_info[dataset] = {}
            dataset_info[dataset]['subject_count'] = 0
            dataset_info[dataset]['example_count'] = 0
            dataset_info[dataset]['subjects'] = {}
            
        if not subject in dataset_info[dataset]['subjects']:
            dataset_info[dataset]['subject_count'] += 1
            dataset_info[dataset]['subjects'][subject] = {}
            dataset_info[dataset]['subjects'][subject]['diagnosis'] = diagnosis
            dataset_info[dataset]['subjects'][subject]['example_count'] = 0
            dataset_info[dataset]['subjects'][subject]['files'] = []
            
        if not task in dataset_info[dataset]['subjects'][subject]:
            dataset_info[dataset]['subjects'][subject][task] = 0
            
        dataset_info[dataset]['subjects'][subject][task] += 1
        dataset_info[dataset]['subjects'][subject]['example_count'] += 1
        dataset_info[dataset]['subjects'][subject]['files'].append(file_path)
        dataset_info[dataset]['example_count'] += 1
        
    # -------------------------------------------------------------------------
    # Dataset statistics
    # -------------------------------------------------------------------------
    example_counts = []
    diagnosis_counts = {}
    diagnosis_example_counts = {}
    for subject in dataset_info[dataset]['subjects'].values():
        example_counts.append(subject['example_count'])
        if subject['diagnosis'] not in diagnosis_counts.keys():
            diagnosis_counts[subject['diagnosis']] = 1
            diagnosis_example_counts[subject['diagnosis']] = subject['example_count']
        else:
            diagnosis_counts[subject['diagnosis']] += 1
            diagnosis_example_counts[subject['diagnosis']] += subject['example_count']
        
    subject_ids = []
    for subject_id in dataset_info[dataset]['subjects'].keys():
        subject_ids.append(subject_id)
        
    logger.info(f'  Segmenting dataset: {dataset}')
    logger.info(f'  Number of subjects: {dataset_info[dataset]["subject_count"]}')
    logger.info(f'  Total number of data examples: {dataset_info[dataset]["example_count"]}')
    logger.info(f'  Median number of examples per subject: {np.median(example_counts)}')
    logger.info('  Number of subjects per diagnosis:')
    _ = [logger.info(f'  {key}: {value}') for key, value in diagnosis_counts.items()]
    logger.info('  Number of examples per diagnosis:')
    _ = [logger.info(f'  {key}: {value}') for key, value in diagnosis_example_counts.items()]
    
    # -------------------------------------------------------------------------
    # Generate train, valid and test sets
    # -------------------------------------------------------------------------
    with open(os.path.join(dest, "train.tsv"), "w") as train_f, open(
        os.path.join(dest, "valid.tsv"), "w") as valid_f, open(
        os.path.join(dest, "test.tsv"), "w") as test_f:
        
        print(dir_path, file=train_f)
        print(dir_path, file=valid_f)
        print(dir_path, file=test_f)
        
        frames = data_params.time_context * data_params.Fs
        
        if data_params.balance_data:
            bp_train, bp_valid, bp_test = 0, 0, 0
            dys_train, dys_valid, dys_test = 0, 0, 0
            
            for subject_id, subject_data in dataset_info[dataset]['subjects'].items():
                
                if subject_data['diagnosis'] != 'bp':
                    continue
                    
                if (data_params.train_valid_test_partition == 'leave-k-out'):
                    random_num = rand.random()
                    if random_num > (data_params.valid_percent + data_params.test_percent):
                        dest = train_f
                        bp_train += subject_data['example_count']
                    elif random_num <= (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                        dest = valid_f
                        bp_valid += subject_data['example_count']
                    else:
                        dest = test_f
                        bp_test += subject_data['example_count']
                
                for fname in subject_data['files']:
                            
                    dataset = fname.split(split_char)[-5]
                    diagnosis = fname.split(split_char)[-4]
                    subject = fname.split(split_char)[-3]
                    task = fname.split(split_char)[-2]
                    example = int(fname[-7:-5])
                    
                    file_path = os.path.realpath(fname)
                    
                    # -------------------------------------------------------------
                    # TESTED
                    if (data_params.train_valid_test_partition == 'random'):
                        random_num = rand.random()
                        if random_num > (data_params.valid_percent + data_params.test_percent):
                            dest = train_f
                            bp_train += 1
                        elif random_num < (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                            dest = valid_f
                            bp_valid += 1
                        else:
                            dest = test_f
                            bp_test += 1
                    # -------------------------------------------------------------
                    
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                    )
                    
            for subject_id, subject_data in dataset_info[dataset]['subjects'].items():
                
                if subject_data['diagnosis'] != 'dys':
                    continue
                
                if (data_params.train_valid_test_partition == 'leave-k-out'):
                    random_num = rand.random()
                    if random_num > (data_params.valid_percent + data_params.test_percent):
                        dest = train_f
                        if dys_train + subject_data['example_count'] > bp_train + 2:
                            continue
                        dys_train += subject_data['example_count']
                    elif random_num < (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                        dest = valid_f
                        if dys_valid + subject_data['example_count'] > bp_valid + 2:
                            continue
                        dys_valid += subject_data['example_count']
                    else:
                        dest = test_f
                        if dys_test + subject_data['example_count'] > bp_test + 2:
                            continue
                        dys_test += subject_data['example_count']
                
                for fname in subject_data['files']:
                            
                    dataset = fname.split(split_char)[-5]
                    diagnosis = fname.split(split_char)[-4]
                    subject = fname.split(split_char)[-3]
                    task = fname.split(split_char)[-2]
                    example = int(fname[-7:-5])
                    
                    file_path = os.path.realpath(fname)
                    
                    # -------------------------------------------------------------
                    # TESTED
                    if (data_params.train_valid_test_partition == 'random'):
                        random_num = rand.random()
                        if random_num > (data_params.valid_percent + data_params.test_percent):
                            dest = train_f
                            if dys_train + 1 > bp_train:
                                continue
                            dys_train += 1
                        elif random_num < (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                            dest = valid_f
                            if dys_valid + 1 > bp_valid:
                                continue
                            dys_valid += 1
                        else:
                            dest = test_f
                            if dys_test + 1 > bp_test:
                                continue
                            dys_test += 1
                    # -------------------------------------------------------------
                    
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                    )
                 
        else:
            for subject_id, subject_data in dataset_info[dataset]['subjects'].items():
                
                if data_params.objective == "reconstruction":
                    if subject_data['diagnosis'] != 'bp':
                        continue
                
                if (data_params.train_valid_test_partition == 'leave-k-out'):
                    random_num = rand.random()
                    if random_num > (data_params.valid_percent + data_params.test_percent):
                        dest = train_f
                    elif random_num < (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                        dest = valid_f
                    else:
                        dest = test_f
                
                for fname in subject_data['files']:
                            
                    dataset = fname.split(split_char)[-5]
                    diagnosis = fname.split(split_char)[-4]
                    subject = fname.split(split_char)[-3]
                    task = fname.split(split_char)[-2]
                    example = int(fname[-7:-5])
                    
                    file_path = os.path.realpath(fname)
                    
                    # -------------------------------------------------------------
                    # TESTED
                    if (data_params.train_valid_test_partition == 'random'):
                        random_num = rand.random()
                        if random_num > (data_params.valid_percent + data_params.test_percent):
                            dest = train_f
                        elif random_num < (data_params.valid_percent + data_params.test_percent) and random_num > data_params.test_percent:
                            dest = valid_f
                        else:
                            dest = test_f
                    # -------------------------------------------------------------
                    
                    print(
                        "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
                    )


if __name__ == "__main__":    
    main()