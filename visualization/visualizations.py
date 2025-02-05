# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 23:34:19 2021

@author: Aneta

Script for visualizating training logs.

"""

import matplotlib
matplotlib.use('Agg')

# Import other libraries
import numpy as np
from matplotlib import pyplot as plt

def visualize_logs(logs, log_params=None):

    # Select the file path where the results are stored     
    if log_params:
        results_dir = log_params.results_path
    else:
        results_dir = "../checkpoint/"
    
    train_losses = np.asarray(logs['epoch_train_loss'])
    val_losses = np.asarray(logs['epoch_valid_loss'])
    train_accs = np.asarray(logs['epoch_train_acc'])
    val_accs = np.asarray(logs['epoch_valid_acc'])
    
    num_epochs = len(logs['epoch'])
    
    FONT_SIZE = 15
    
    #   Plotting training and validation loss ---------------------------------
    epoch_axis = np.arange(num_epochs)+1
    
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, train_losses, label='Training loss')
    plt.plot(epoch_axis, val_losses, label='Validation loss')
    plt.xlabel('$epoch$')
    plt.ylabel('$loss$')
    plt.legend(frameon=False)
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
    # plt.show()
    plt.savefig(results_dir + 'Training_validation_loss.png', dpi=600)
    
    #   Plotting training and validation accuracy -----------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epoch_axis, train_accs, label='Training accuracy')
    plt.plot(epoch_axis, val_accs, label='Validation accuracy')
    plt.xlabel('$epoch$')
    plt.ylabel('$accuracy$')
    plt.legend(frameon=False)
    plt.rc('font', size=FONT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=FONT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONT_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=FONT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=FONT_SIZE)  # fontsize of the figure title
    # plt.show()
    plt.savefig(results_dir + 'Training_validation_accuracy.png', dpi=600)