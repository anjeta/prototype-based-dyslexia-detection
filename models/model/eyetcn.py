# -*- coding: utf-8 -*-
"""
Created on Sun May 26 23:08:48 2024

@author: aneta.kartali
"""

import contextlib
import datetime
import numpy as np
from model.tcn import TCN
import sklearn.metrics
import time
import torch
import torch.nn as nn
import sys


class Model:
    """
    EyeTCN Model
    """
    
    def __init__(self, device, params, criterion, regularizer, logger):
        
        self.params = params
        self.device = device

        # Create Encoder and Decoder module
        self.encoder = Encoder(device, params, logger)
        self.decoder = Decoder(device, params, logger)
        
        # Reset manual seed for repeatable initialization
        torch.manual_seed(42)
        self.encoder.init_weights()
        self.decoder.init_weights()

        self.encoder.to(device)
        self.decoder.to(device)
        
        # Create optimizer
        self.lr = self.params.lr
        self.scheduler = self.params.scheduler
        self.criterion = criterion
        self.regularizer = regularizer
        self.optimizer = self.create_optimizer()
        
        self.logger = logger
                
    def train(self):
        self.encoder.train()
        self.decoder.train()
    
    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.encoder.parameters()) + sum(p.numel() for p in self.decoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())
    
    def create_optimizer(self):
        g_params = self.get_parameters()
        if self.params.optimizer == 'adam':
            optimizer = torch.optim.Adam(g_params, lr=self.lr)
        elif self.params.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(g_params, lr=self.lr,
                                     weight_decay=self.params.L2)
        elif self.params.optimizer == 'sgd':
            optimizer = torch.optim.SGD(g_params, lr=self.lr, 
                                        momentum=self.params.momentum, weight_decay=self.params.L2)
        else:
            self.logger.info(f"Invalid optimizer type: {self.params.optimizer}")
            sys.exit()
        return optimizer
    
    def calc_conv_output_size(self, seq_len, kernel_size, stride, padding, dilation):
        return int(np.floor(((seq_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1))
    
    def calc_last_seq_len(self):
        # Calculate the output length of the last layer
        last_seq_len = self.params.input_seq_len
        """
        for i in range(len(self.params.kernel_sizes) - 1):
            last_seq_len = self.calc_conv_output_size(last_seq_len, self.params.kernel_sizes[i], 
                                                      self.params.strides[i], self.params.paddings[i], 
                                                      self.params.dilations[i])
        """
        return last_seq_len
    
    def save_model(self, checkpoint_path):
        
            state_dict = {"encoder": self.encoder.state_dict(), 
                          "decoder": self.decoder.state_dict(),
                          "optimizer": self.optimizer.state_dict()}
            torch.save(state_dict, checkpoint_path)
            
    def load_model(self, checkpoint_path):
        self.logger.info("Loading model from " + checkpoint_path)
        try:
            state_dict = torch.load(checkpoint_path, 'cpu')
            self.encoder.load_state_dict(state_dict["encoder"])
            self.decoder.load_state_dict(state_dict["decoder"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
        except:
            self.logger.info(f"ERROR: Unsuccessful loading model from checkpoint: {checkpoint_path}")
            sys.exit()
            
    def load_encoder(self, checkpoint_path):
        self.logger.info("Loading encoder from " + checkpoint_path)
        try:
            state_dict = torch.load(checkpoint_path, 'cpu')
            self.encoder.load_state_dict(state_dict["encoder"])
        except:
            self.logger.info(f"ERROR: Unsuccessful loading encoder from checkpoint: {checkpoint_path}")
            sys.exit()
            
    def calculate_acc(self, prediction, target):
        if self.params.criterion == 'BCE':
            prediction = (prediction[:,0] > 0.5) * 1.0
        elif self.params.criterion == 'BCE_with_logits':
            prediction = (prediction[:,0] > 0.0) * 1.0
        elif self.params.criterion == 'CE':
            prediction = torch.argmax(prediction, dim=-1)
        else:
            self.logger.info(f"Unsupported loss function: {self.params.criterion}")
            sys.exit()
        if len(target.size()) == 2:
            target = target.squeeze(-1)
        acc = torch.sum(torch.eq(prediction, target) * 1.0).detach().cpu().numpy() / torch.numel(target)
        return acc
    
    def forward(self, inputs, target, finetuning=False):
        if self.params.use_handcrafted_features:
            features = inputs[1]
            inputs = inputs[0]
        encoded_data = {}
        with torch.no_grad() if finetuning else contextlib.ExitStack():
            encoded_data = self.encoder(inputs)
        if finetuning:
            z = [zi for zi in encoded_data['z'] if zi is not None]
            latent = torch.cat(z, dim=1)
        else:
            latent = encoded_data['features']  # [batch_size, time_steps, embedding_size]
            if self.params.feature_aggregation == 'pooling':
                latent = torch.mean(latent, dim=1)
            else:
                latent = torch.flatten(latent, start_dim=1)
        if self.params.use_handcrafted_features:
            latent = torch.cat((latent, features.squeeze(1)), dim=1)
        prediction = self.decoder(latent)
        return prediction
            
    def train_batch(self, inputs, target, finetuning=False):
        prediction = self.forward(inputs, target, finetuning)
        loss = self.criterion(prediction, target)
        acc = self.calculate_acc(prediction, target)
        
        if (self.regularizer is not None):
            reg_loss = 0
            for param in self.encoder.parameters():
                reg_loss += self.regularizer(param, target=torch.zeros_like(param))
            for param in self.decoder.parameters():
                reg_loss += self.regularizer(param, target=torch.zeros_like(param))
            loss += self.params.L1 * reg_loss
        
        # Backward pass
        loss.backward()
        # Clip gradient norm of model's parameters
        if self.params.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.clip)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.clip)
        # Updating network parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), acc.item()
    
    def eval_batch(self, inputs, target, finetuning=False):
        with torch.no_grad():
            prediction = self.forward(inputs, target, finetuning)
            loss = self.criterion(prediction, target)
            acc = self.calculate_acc(prediction, target)
        return loss.item(), acc.item()      
    
    def train_epoch(self, epoch, data_loader, finetuning=False):
        if self.logger is not None:
            curr_time = datetime.datetime.now()
            self.logger.info(f"{curr_time} | INFO | train_model | Starting training at {epoch+1}")
        start_time = time.perf_counter()
        logs = {"train_loss": [], "train_acc": []}
        
        self.train()
        for batch_data in data_loader:
            if self.params.use_handcrafted_features:
                inputs = [batch_data[0].to(self.device), batch_data[1].to(self.device)]
                target = batch_data[2].to(self.device)
            elif self.params.data_info:
                inputs = batch_data[0]
                target = batch_data[2]
                inputs = inputs.to(self.device)
                target = target.to(self.device)
            else:
                inputs = batch_data[0]
                target = batch_data[1]
                inputs = inputs.to(self.device)
                target = target.to(self.device)
            batch_loss, batch_acc = self.train_batch(inputs, target, finetuning)
            logs["train_loss"].append(batch_loss)
            logs["train_acc"].append(batch_acc)
            
        epoch_loss = np.mean(logs["train_loss"])
        epoch_acc = np.mean(logs["train_acc"])
        if self.logger is not None:
            elapsed = time.perf_counter() - start_time
            curr_time = datetime.datetime.now()
            self.logger.info(f"{curr_time} | INFO | train_model | Epoch {epoch+1} | Training for {elapsed} seconds | loss: {epoch_loss} | acc: {epoch_acc}")
            self.logger.info("\n")
            
        return epoch_loss, epoch_acc, logs
    
    def eval_epoch(self, epoch, data_loader, finetuning=False):
        if self.logger is not None:
            curr_time = datetime.datetime.now()
            self.logger.info(f"{curr_time} | INFO | eval_model | Starting evaluation at epoch {epoch+1}")
        start_time = time.perf_counter()
        logs = {"eval_loss": [], "eval_acc": []}
        
        self.eval()
        for batch_data in data_loader:
            if self.params.use_handcrafted_features:
                inputs = [batch_data[0].to(self.device), batch_data[1].to(self.device)]
                target = batch_data[2].to(self.device)
            elif self.params.data_info:
                inputs = batch_data[0]
                target = batch_data[2]
                inputs = inputs.to(self.device)
                target = target.to(self.device)
            else:
                inputs = batch_data[0]
                target = batch_data[1]
                inputs = inputs.to(self.device)
                target = target.to(self.device)
            batch_loss, batch_acc = self.eval_batch(inputs, target, finetuning)
            logs["eval_loss"].append(batch_loss)
            logs["eval_acc"].append(batch_acc)
            
        epoch_loss = np.mean(logs["eval_loss"])
        epoch_acc = np.mean(logs["eval_acc"])
        if self.logger is not None:
            elapsed = time.perf_counter() - start_time
            curr_time = datetime.datetime.now()
            self.logger.info(f"{curr_time} | INFO | eval_model | Epoch {epoch+1} | One epoch evaluation for {elapsed} s | loss: {epoch_loss} | acc: {epoch_acc}")
            self.logger.info("\n")
            
        return epoch_loss, epoch_acc, logs
    
    def get_training_metrics(self, data_loader):
        
        if self.params.data_info:
            inputs, subjects, labels = zip(*[batch for batch in data_loader])
        else:
            inputs, labels = zip(*[batch for batch in data_loader])
        inputs = torch.cat(inputs, dim=0).to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)
        self.eval()
        with torch.no_grad():
            predictions = self.forward(inputs, labels)
        if self.params.criterion == 'BCE_with_logits':
            probabilities = nn.functional.sigmoid(predictions)
        elif self.params.criterion == 'CE':
            probabilities = nn.functional.softmax(predictions, dim=1)
        if self.params.criterion == 'BCE':
            predictions = (predictions[:,0] > 0.5) * 1.0
        elif self.params.criterion == 'BCE_with_logits':
            predictions = (predictions[:,0] > 0.0) * 1.0
        elif self.params.criterion == 'CE':
            predictions = torch.argmax(predictions, dim=-1)
        if len(labels.size()) == 2:
            targets = labels.squeeze(-1)
        else:
            targets = labels

        y_true = targets.detach().cpu().numpy().astype(int)
        y_pred = predictions.detach().cpu().numpy().astype(int)
        confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
        
        if self.params.criterion == 'BCE_with_logits':
            y_score = probabilities.detach().cpu().numpy()
        elif self.params.criterion == 'CE':
            y_score = probabilities.detach().cpu().numpy()[:,1]
        
        try:
            acc = sklearn.metrics.accuracy_score(y_true, y_pred)
            brier = sklearn.metrics.brier_score_loss(y_true, y_score)
            auc = sklearn.metrics.roc_auc_score(y_true, y_score)
            balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
            f1 = sklearn.metrics.f1_score(y_true, y_pred)
            precision = sklearn.metrics.precision_score(y_true, y_pred)
            recall = sklearn.metrics.recall_score(y_true, y_pred)
        except:
            acc, brier, auc, balanced_acc, f1, precision, recall = None, None, None, None, None, None, None
        
        logs = {'confusion_matrix': confusion_matrix, 'acc': acc, 'brier': brier,
                'auc': auc, 'balanced_acc': balanced_acc, 'f1': f1, 
                'precision': precision, 'recall': recall}
        return logs

# -----------------------------------------------------------------------------
# ENCODER - 1D Temporal Convolutional Network
# -----------------------------------------------------------------------------        
class Encoder(nn.Module):

    def __init__(self, device, params, logger):
        super(Encoder, self).__init__()
        self.params = params
        self.device = device
        self.logger = logger
        
        self.tcn = TCN(num_inputs=self.params.num_input_channels, 
                       num_channels=self.params.encoder_embedding_sizes, 
                       kernel_size=self.params.encoder_kernel_size, 
                       dilations=self.params.encoder_dilations, 
                       dilation_reset=None,
                       dropout=self.params.encoder_dropout, 
                       causal=self.params.encoder_causal, 
                       use_norm=self.params.encoder_norm, 
                       activation=self.params.encoder_activation, 
                       kernel_initializer=self.params.encoder_init, 
                       use_skip_connections=self.params.encoder_skip_connections)

    def init_weights(self):
        # Encoder is initialized in TCN class
        return
    
    def forward(self, inputs):
        # BxCxT = [batch_size, channels, time_samples]
        x, block_outputs = self.tcn(inputs)
        # [batch_size, embedidng_size, time_steps]
        x = x.transpose(1, 2)  # [batch_size, time_steps, embedding_size]
        for i in range(len(block_outputs)):
            block_outputs[i] = block_outputs[i].transpose(1, 2)
        # Return last layer output and intermediate block outputs
        return {"features": x, 'block_outputs': block_outputs}

# -----------------------------------------------------------------------------
# DECODER - Linear Fully-connected Network
# ----------------------------------------------------------------------------- 
class Decoder(nn.Module):

    def __init__(self, device, params, logger):
        super(Decoder, self).__init__()
        self.params = params
        self.device = device
        self.logger = logger
        
        if self.params.decoder_activation == "relu":
            activation = nn.ReLU()
        else:
            self.logger.info(f"ERROR: Unsupported activation function: {self.params.decoder_activation}")
            sys.exit()
            
        if self.params.encoder_init == 'xavier_uniform':
            self.initializer = nn.init.xavier_uniform_
        elif self.params.encoder_init == 'xavier_normal':
            self.initializer = nn.init.xavier_normal_
        elif self.params.encoder_init == 'kaiming_uniform':
            self.initializer = nn.init.kaiming_uniform_
        elif self.params.encoder_init == 'kaiming_normal':
            self.initializer = nn.init.kaiming_normal_
        elif self.params.encoder_init == 'standard_normal':
            self.initializer = nn.init.normal_
        else:
            self.logger.info(f"ERROR: Invalid normalization method: {self.params.encoder_init}")
            sys.exit()
        
        if self.params.feature_aggregation == "pooling":
            in_features = self.params.encoder_embedding_sizes[-1]
        elif self.params.feature_aggregation == None:
            in_features = 2 * self.params.encoder_embedding_sizes[-1]
        else:
            in_features = self.params.encoder_embedding_sizes[-1] * self.params.input_seq_len
        if self.params.use_handcrafted_features:
            in_features = in_features + self.params.num_handcrafted_features
        
        self.decoder_layers = None
        if self.params.num_decoder_layers > 0:
            self.decoder_layers = nn.ModuleList()
            for i in range(self.params.num_decoder_layers):
                self.decoder_layers.append(nn.Sequential(
                            nn.Linear(in_features, int(in_features / 2)),
                            nn.Dropout(p=self.params.decoder_dropout),
                            activation,
                            )
                )
                in_features = int(in_features / 2)
        self.last_linear = nn.Linear(in_features, self.params.output_size)
        if self.params.criterion == 'BCE':
            self.softmax_layer = nn.Sigmoid()
        else:
            self.softmax_layer = None   

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # self.initializer(m.weight, gain=nn.init.calculate_gain('linear'))
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, x):
        # x = [batch_size, time_steps, embedding_size]
        if self.decoder_layers is not None:
            for layer in self.decoder_layers:
                x = layer(x)  # [batch_size, time_steps, 1]
        out = self.last_linear(x)
        if self.softmax_layer is not None:
            out = self.softmax_layer(out)
        return out 