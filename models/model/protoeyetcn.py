# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:20:40 2024

@author: aneta.kartali
"""

import contextlib
import datetime
import numpy as np
from model.tcn import TCN
import os
import sklearn.metrics
import sys
import time
import torch
import torch.nn as nn

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from model.utils.training_utils import WarmupLRScheduler
from model.eyetcn import Model as EyeTCNClassifier

class Model:
    """
    ProtoEyeTCN Model
    """
    
    def __init__(self, device, params, criterion, regularizer, stage, logger):
        
        self.params = params
        self.device = device

        # Create Encoder and Decoder module
        self.encoder = Encoder(device, params, logger)
        self.prototype_layer = PrototypeLayer(device, params, logger)
        self.decoder = Decoder(device, params, logger)
        
        # Reset manual seed for repeatable initialization
        torch.manual_seed(42)
        self.encoder.init_weights()
        self.prototype_layer.init_weights()
        self.decoder.init_weights()

        self.encoder.to(device)
        self.prototype_layer.to(device)
        self.decoder.to(device)
        
        # Create optimizer
        self.lr = self.params.lr
        self.criterion = criterion
        self.regularizer = regularizer
        self.optimizer = self.create_optimizer()
        if self.params.scheduler is not None:
            self.scheduler = WarmupLRScheduler(self.optimizer, self.params.lr, self.params.warmup_epochs)
        else:
            self.scheduler = None
        
        self.projection_freq = self.params.projection_freq
        self.stage = stage
        self.logger = logger
                
    def train(self):
        self.encoder.train()
        self.prototype_layer.train()
        self.decoder.train()
    
    def eval(self):
        self.encoder.eval()
        self.prototype_layer.eval()
        self.decoder.eval()

    def get_parameter_count(self):
        total_params = sum(p.numel() for p in self.encoder.parameters())+ sum(p.numel() for p in self.prototype_layer.parameters()) + sum(p.numel() for p in self.decoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad) + sum(p.numel() for p in self.prototype_layer.parameters() if p.requires_grad) + sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def get_parameters(self):
        return list(self.encoder.parameters()) + list(self.prototype_layer.parameters()) + list(self.decoder.parameters())
    
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
        return last_seq_len
    
    def save_model(self, checkpoint_path):
        
            state_dict = {"encoder": self.encoder.state_dict(),
                          "prototype_layer": self.prototype_layer.state_dict(),
                          "decoder": self.decoder.state_dict(),
                          "optimizer": self.optimizer.state_dict()}
            torch.save(state_dict, checkpoint_path)
            
    def load_model(self, checkpoint_path, Model):
        self.logger.info("Loading model from " + checkpoint_path)
        if self.stage == 1:
            # For warm-up load the pretrained uninterpretable model
            pretrained = Model(self.device, self.params, self.criterion, self.regularizer)
            pretrained.load_encoder(checkpoint_path)
            self.encoder = pretrained.encoder
        else:
            try:
                state_dict = torch.load(checkpoint_path, 'cpu')
                self.encoder.load_state_dict(state_dict["encoder"])
                self.prototype_layer.load_state_dict(state_dict["prototype_layer"])
                self.decoder.load_state_dict(state_dict["decoder"])
                self.optimizer.load_state_dict(state_dict["optimizer"])
            except:
                self.logger.info(f"ERROR: Unsuccessful loading model from checkpoint: {checkpoint_path}")
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
        encoded_data = {}
        with torch.no_grad() if finetuning or self.stage == 2 else contextlib.ExitStack():
            encoded_data = self.encoder(inputs)
            latent = encoded_data['features']  # [batch_size, time_steps, embedding_size]
            if self.params.feature_aggregation == 'pooling':
                latent = torch.mean(latent, dim=1)
            else:
                latent = torch.flatten(latent, start_dim=1)
        with torch.no_grad() if self.stage == 2 else contextlib.ExitStack():
            latent, proto_loss = self.prototype_layer(latent)
        with torch.no_grad() if finetuning and self.params.prototype_learning else contextlib.ExitStack():
            prediction = self.decoder(latent, target)
        return prediction, proto_loss
            
    def train_batch(self, inputs, target, finetuning=False):
        prediction, proto_loss = self.forward(inputs, target, finetuning)
        loss = self.criterion(prediction, target)
        acc = self.calculate_acc(prediction, target)
        
        loss += proto_loss
        if self.regularizer is not None and self.stage == 2:
            reg_loss = 0
            # for param in self.encoder.parameters():
            #     reg_loss += self.regularizer(param, target=torch.zeros_like(param))
            for param in self.decoder.parameters():
                reg_loss += self.regularizer(param, target=torch.zeros_like(param))
            loss += self.params.L1 * reg_loss
        
        # Backward pass
        loss.backward()
        # Clip gradient norm of model's parameters
        if self.params.clip_gradients:
            if not finetuning:
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.params.clip)
            if self.params.prototype_learning:
                clamped = nn.Parameter(self.prototype_layer.get_weights().clamp(-1.,1.))
                self.prototype_layer.set_weights(clamped)
            torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.params.clip)
        # Updating network parameters
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), acc.item()
    
    def eval_batch(self, inputs, target, finetuning=False):
        with torch.no_grad():
            prediction, proto_loss = self.forward(inputs, target, finetuning)
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
            
        epoch_lr = self.scheduler.step() if self.scheduler else self.lr
        self.prototype_projection(data_loader, epoch+1)
        
        if epoch % self.projection_freq == 0:
            self.stage = 2
        else:
            self.stage = 1
            
        epoch_loss = np.mean(logs["train_loss"])
        epoch_acc = np.mean(logs["train_acc"])
        if self.logger is not None:
            elapsed = time.perf_counter() - start_time
            curr_time = datetime.datetime.now()
            self.logger.info(f"{curr_time} | INFO | train_model | Epoch {epoch+1} | Training for {elapsed} seconds | loss: {epoch_loss} | acc: {epoch_acc}")
            
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
            
        return epoch_loss, epoch_acc, logs
    
    def prototype_projection(self, data_loader, epoch):
        """
        Prototype projection computation + setting
        """
        if self.params.prototype_learning:
            if epoch % self.projection_freq == 0:
                
                if self.params.use_handcrafted_features:
                    inputs_raw, inputs_features, targets = zip(*[batch for batch in data_loader])
                    inputs = [inputs_raw, inputs_features]
                elif self.params.data_info:
                    inputs, _, targets = zip(*[batch for batch in data_loader])
                else:
                    inputs, targets = zip(*[batch for batch in data_loader])                
                inputs = torch.cat(inputs, dim=0).to(self.device)
                target = torch.cat(targets, dim=0).to(self.device)
    
                # get encodings of all train sequences
                with torch.no_grad():
                    encoded_data = self.encoder.forward(inputs)
                    latent = encoded_data['features']  # [batch_size, time_steps, embedding_size]
                    if self.params.feature_aggregation == 'pooling':
                        latent = torch.mean(latent, dim=1)
                    else:
                        latent = torch.flatten(latent, start_dim=1)
                    latent = torch.unsqueeze(latent, dim=-2)
    
                # distance matrix from prototypes
                protos = self.prototype_layer.get_weights()
                d2 = torch.norm(latent - protos, p=2, dim=-1)
                
                # Identify 
                # with torch.no_grad():
                #     predictions = self.decoder.forward(torch.exp(-d2), target)
                # if self.params.criterion == 'BCE':
                #     predictions = (predictions[:,0] > 0.5) * 1.0
                # elif self.params.criterion == 'BCE_with_logits':
                #     predictions = (predictions[:,0] > 0.0) * 1.0
                # elif self.params.criterion == 'CE':
                #     predictions = torch.argmax(predictions, dim=-1)
                # if len(target.size()) == 2:
                #     target = target.squeeze(-1)
                # matches = (torch.eq(predictions, target) * 1.0).int()
                if epoch >= 100:
                    protos_0 = protos[:, :protos.size(1)//2, :]                
                    latent_0 = latent[target == 0]
                    # matches_0 = matches[target == 0]
                    # latent_0 = latent_0[matches_0 == 1]
                    d2_0 = torch.norm(latent_0 - protos_0, p=2, dim=-1)
                    new_protos_0 = latent_0[torch.argmin(d2_0, dim=0)]
                    
                    protos_1 = protos[:, protos.size(1)//2:, :]
                    latent_1 = latent[target == 1]
                    # matches_1 = matches[target == 1]
                    # latent_1 = latent_1[matches_1 == 1]
                    d2_1 = torch.norm(latent_1 - protos_1, p=2, dim=-1)
                    new_protos_1 = latent_1[torch.argmin(d2_1, dim=0)]
                    
                    new_protos = torch.cat((new_protos_0, new_protos_1), dim=0)
                    new_protos = torch.reshape(new_protos, protos.shape) # need to swap axes
                else:
                    # reset prototypes to nearest neighbors
                    new_protos = latent[torch.argmin(d2, dim=0)]
                    new_protos = torch.reshape(new_protos, protos.shape) # need to swap axes
    
                self.prototype_layer.set_weights(nn.Parameter(new_protos))
                
    def get_training_metrics(self, data_loader):
        
        if self.params.use_handcrafted_features:
            inputs_raw, inputs_features, labels = zip(*[batch for batch in data_loader])
            inputs = [inputs_raw, inputs_features]
        elif self.params.data_info:
            inputs, _, labels = zip(*[batch for batch in data_loader])
        else:
            inputs, labels = zip(*[batch for batch in data_loader]) 
        inputs = torch.cat(inputs, dim=0).to(self.device)
        labels = torch.cat(labels, dim=0).to(self.device)
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(inputs, labels)
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
        
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        brier = sklearn.metrics.brier_score_loss(y_true, y_score)
        auc = sklearn.metrics.roc_auc_score(y_true, y_score)
        balanced_acc = sklearn.metrics.balanced_accuracy_score(y_true, y_pred)
        f1 = sklearn.metrics.f1_score(y_true, y_pred)
        precision = sklearn.metrics.precision_score(y_true, y_pred)
        recall = sklearn.metrics.recall_score(y_true, y_pred)
        
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
# PROTOTYPE MODULE
# -----------------------------------------------------------------------------
class PrototypeLayer(nn.Module):
    """
    PyTorch implemmentation of the "Prototype Layer" 
    adapted from: https://github.com/rgmyr/tf-ProSeNet/tree/master
    """
    def __init__(self, device, params, logger):
        """
        Parameters
        ----------
        k : int
            Number of prototype vectors to create.
        dmin : float, optional
            Threshold to determine whether two prototypes are close, default=1.0.
            For "diversity" regularization. See paper section 3.2 for details.
        Ld : float, optional
            Weight for "diversity" regularization loss, default=0.01.
        Lc : float, optional
            Weight for "clustering" regularization loss, default=0.01.
        Le : float, optional
            Weight for "evidence" regularization loss, default=0.1.
        **kwargs
            Additional arguments for base `Layer` constructor (name, etc.)
        """
        super(PrototypeLayer, self).__init__()
        self.params = params
        self.device = device
        self.logger = logger
        
        if self.params.prototype_init == 'xavier_uniform':
            self.initializer = nn.init.xavier_uniform_
        elif self.params.prototype_init == 'xavier_normal':
            self.initializer = nn.init.xavier_normal_
        elif self.params.prototype_init == 'kaiming_uniform':
            self.initializer = nn.init.kaiming_uniform_
        elif self.params.prototype_init == 'kaiming_normal':
            self.initializer = nn.init.kaiming_normal_
        elif self.params.prototype_init == 'standard_normal':
            self.initializer = nn.init.normal_
        else:
            self.logger.info(f"ERROR: Invalid normalization method: {self.params.decoder_init}")
            sys.exit()
        
        # Creating prototypes as weight variables
        if self.params.prototype_learning:
            self.prototypes = nn.Parameter(torch.zeros(1, self.params.k, self.params.prototype_channels))
        else:
            self.prototypes = nn.Identity()


    def forward(self, x, training=None):
        
        if self.params.prototype_learning:
            x = torch.unsqueeze(x, dim=-2)
            # L2 distances between encodings and prototypes
            d2 = torch.norm(x - self.prototypes, p=2, dim=-1)
    
            # Losses only computed if model is in training mode
            if self.training:
                dLoss = self.params.Ld * self._diversity_term()
                cLoss = self.params.Lc * torch.sum(torch.min(d2, dim=0).values)
                eLoss = self.params.Le * torch.sum(torch.min(d2, dim=1).values)
            else:
                dLoss, cLoss, eLoss = 0., 0., 0.
                
            protoLoss = dLoss + cLoss + eLoss
    
            # Return exponentially squashed distances
            return torch.exp(-d2), protoLoss
        else:
            return x, 0.0

    def _diversity_term(self):
        """Compute the "diversity" loss,
        which penalizes prototypes that are close to each other

        NOTE: Computes full distance matrix, which is redudant, but `prototypes`
              is usually a small-ish tensor and performance is acceptable,
              so I'm not going to worry about it.
        """
        D = self._distance_matrix(self.prototypes, self.prototypes)

        Rd = nn.functional.relu(-D + self.params.dmin)

        # Zero the diagonal elements
        zero_diag = torch.ones_like(Rd) - torch.eye(self.params.k).to(self.device)

        return torch.sum(torch.square(Rd * zero_diag)) / 2.0

    def init_weights(self):
        if self.params.prototype_learning:
            self.initializer(self.prototypes, gain=nn.init.calculate_gain('linear'))
    
    def set_weights(self, weights):
        if self.params.prototype_learning:
            self.prototypes = weights
    
    def get_weights(self):
        if self.params.prototype_learning:
            return self.prototypes
        else:
            return None
    
    def get_config(self):
        # implement to make serializable
        pass
    
    def _distance_matrix(self, a, b):
        """Return the distance matrix between rows of `a` and `b`
    
        They must both be squeezable or expand_dims-able to 2D,
        and have compatible shapes (same number of columns).
    
        Returns
        -------
        D : Tensor
            2D where D[i, j] == distance(a[i], b[j])
        """
        a_was_b = a is b
    
        #a = make2D(a)
        rA = torch.unsqueeze(torch.sum(a * a, dim=-1), dim=-1)
    
        if a_was_b:
            b, rB = a, rA
        else:
            #b = make2D(b)
            rB = torch.unsqueeze(torch.sum(b * b, dim=-1), dim=-1)
    
        D = rA - 2 * torch.matmul(a, b.mT) + rB.mT
        
        zero_diag = torch.ones_like(D) - torch.eye(self.params.k).to(self.device)
        D = torch.abs(D * zero_diag)
    
        return torch.sqrt(D)

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
            
        if self.params.decoder_init == 'xavier_uniform':
            self.initializer = nn.init.xavier_uniform_
        elif self.params.decoder_init == 'xavier_normal':
            self.initializer = nn.init.xavier_normal_
        elif self.params.decoder_init == 'kaiming_uniform':
            self.initializer = nn.init.kaiming_uniform_
        elif self.params.decoder_init == 'kaiming_normal':
            self.initializer = nn.init.kaiming_normal_
        elif self.params.decoder_init == 'standard_normal':
            self.initializer = nn.init.normal_
        else:
            self.logger.info(f"ERROR: Invalid normalization method: {self.params.decoder_init}")
            sys.exit()
        
        if self.params.prototype_learning:
            in_features = self.params.k
            self.decoder_layers = None
            self.last_linear = nn.Linear(in_features, self.params.output_size, bias=False)
            # self.last_linear = nn.Sequential(
            #             nn.Linear(in_features, self.params.output_size, bias=False),
            #             nn.Dropout(p=self.params.decoder_dropout),
            #             activation,
            #             )
            self.num_prototypes = self.params.k
            self.num_classes = self.params.output_size
            self.prototype_class_identity = torch.zeros(self.num_prototypes,
                                                            self.num_classes)
            num_prototypes_per_class = self.num_prototypes // self.num_classes
            for j in range(self.num_prototypes):
                self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        else:
            if self.params.feature_aggregation == "pooling":
                in_features = self.params.encoder_embedding_sizes[-1]
            elif self.params.feature_aggregation == None:
                in_features = 2 * self.params.encoder_embedding_sizes[-1]
            else:
                in_features = self.params.encoder_embedding_sizes[-1] * self.params.input_seq_len
                
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
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        if self.params.prototype_learning:
            self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        
    def forward(self, x, target):
        # x = [batch_size, time_steps, embedding_size]
        if self.decoder_layers is not None:
            for layer in self.decoder_layers:
                x = layer(x)  # [batch_size, time_steps, 1]
        out = self.last_linear(x)
        if self.softmax_layer is not None:
            out = self.softmax_layer(out)
        return out 
    
    def set_last_layer_incorrect_connection(self, incorrect_strength=-0.5):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_linear.weight.data.copy_(correct_class_connection * positive_one_weights_locations + incorrect_class_connection * negative_one_weights_locations)
    