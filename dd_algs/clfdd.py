from dd_algs.basedd import BaseDDAlg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from BPTT.me_bptt import MEBPTT
from synset.base_synset import BaseSynSet
from utils import get_model, get_optimizer
from dataset.baseset import ImageDataSet
from modules.basemodule import BaseModule


class CLFDDAlg(BaseDDAlg):

    def __init__(self,
                 synset:BaseSynSet,
                 inner_module:BaseModule,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 real_loader:DataLoader,
                 data_per_loop:int):
        super().__init__(synset,
                         inner_module,
                         inner_model_args,
                         inner_opt_args,
                         inner_batch_size,
                         None)
        self.real_loader = real_loader
        self.data_per_loop = data_per_loop
    
    def meta_loss_handle(self, backbone: nn.Module, *args):
        loss = self.inner_module.forward_loss(backbone, *args)[0]
        return loss
    
    def compute_meta_loss(self):
        num_data = 0
        meta_loss = 0.
        for images, targets in self.real_loader:
            if images.device != self.device:
                images = images.to(self.device)
            if targets.device!=self.device:
                targets = targets.to(self.device)
            if len(targets.shape)==1:
                targets = nn.functional.one_hot(targets).to(torch.float)
            batch_size = targets.shape[0]
            if num_data + batch_size >= self.data_per_loop:
                batch_size = self.data_per_loop - num_data
                images = images[:batch_size]
                targets = targets[:batch_size]
            weight = float(batch_size)/float(self.data_per_loop)
            meta_loss += self.me_bptt.meta_loss(images, targets, weight=weight)
        return meta_loss

