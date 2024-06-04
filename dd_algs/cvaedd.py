from dd_algs.basedd import BaseDDAlg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from BPTT.me_bptt import MEBPTT
from synset.base_synset import BaseSynSet
from utils import get_model, get_optimizer
from dataset.baseset import ImageDataSet
from modules.basemodule import BaseModule
from modules.clfmodule import ClassifierModule
from typing import Callable
from modules.utils import get_module

class CVAEDDAlg(BaseDDAlg):

    def __init__(self,
                 batch_function:Callable,
                 inner_module_args:dict,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 external_module_args:dict,
                 data_per_loop:int,
                 device='cuda'):
        super().__init__(batch_function,
                         inner_module_args,
                         inner_model_args,
                         inner_opt_args,
                         inner_batch_size,
                         device)
        self.external_module_args = external_module_args
        self.external_module = get_module(**external_module_args)
        self.data_per_loop = data_per_loop
    
    def meta_loss_handle(self, backbone: nn.Module, **kwargs):
        loss = self.external_module.forward_loss(backbone=backbone, **kwargs)[0]
        return loss
    
    def compute_meta_loss(self, dataloader:DataLoader):
        num_data = 0
        meta_loss = 0.
        for images, targets in dataloader:
            if images.device != self.device:
                images = images.to(self.device)
            if targets.device!=self.device:
                targets = targets.to(self.device)
            if len(targets.shape)==1:
                targets = nn.functional.one_hot(targets).to(torch.float)
            batch_size = targets.shape[0]
            if num_data + batch_size > self.data_per_loop:
                batch_size = self.data_per_loop - num_data
                images = images[:batch_size]
                targets = targets[:batch_size]
            weight = float(batch_size)/float(self.data_per_loop)
            meta_loss += self.me_bptt.meta_loss(images=images, targets=targets, weight=weight)
            num_data += batch_size
            if num_data >= self.data_per_loop:
                break
        return meta_loss

    def forward_function_handle(self, step_idx:int, backbone:nn.Module, **forward_kwargs):
        '''
        by default, forward_args and forward_kwargs will be unrolled and passed into batch_function
        '''
        batch_kwargs = forward_kwargs
        batch_out = self.batch_function(batch_idx=step_idx, batch_size=self.inner_batch_size, soft_targets=False, **batch_kwargs)
        batch_out = self.inner_module.parse_batch(batch_out)
        loss = self.inner_module.forward_loss(backbone, **batch_out)[0]
        return loss