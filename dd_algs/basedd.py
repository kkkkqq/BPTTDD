import torch
import torch.nn as nn
from BPTT.me_bptt import MEBPTT
from synset.base_synset import BaseSynSet
from utils import get_model, get_optimizer
from dataset.baseset import ImageDataSet
from modules.basemodule import BaseModule

class BaseDDAlg():

    def __init__(self,
                 synset:BaseSynSet,
                 inner_module:BaseModule,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 device=None):
        self.synset = synset
        self.device = device
        if device is None:
            self.device = self.synset.device
        else:
            if self.device != self.synset.device:
                raise AssertionError("DDAlg must be on the same device as synset!")
        self.inner_module = inner_module
        self.inner_model_args = inner_model_args
        self.inner_opt_args = inner_opt_args
        self.inner_batch_size = inner_batch_size
        self.me_bptt = MEBPTT(self.forward_function_handle, self.meta_loss_handle)
        self.me_bptt.register_meta_params(**self.synset.trainables)

    def forward_function_handle(self, step_idx:int, backbone:nn.Module):
        '''
        template, override to change inner forward behavior
        '''
        synset_out = self.synset.batch(step_idx, self.inner_batch_size)
        loss = self.inner_module.forward_loss(backbone, *synset_out)[0]
        return loss
    
    def meta_loss_handle(self, backbone:nn.Module):
        '''
        the meta loss handle registered to self.me_bptt.
        override to customize.
        '''
        raise NotImplementedError
    
    def compute_meta_loss(self):
        '''
        called in self.step(). It calls self.me_bptt.meta_loss.
        override to customize.
        '''
        raise NotImplementedError
    
    def step(self, num_forward:int, num_backward:int):
        '''
        template, override to customize.
        '''
        self.synset.train()
        backbone = get_model(**self.inner_model_args)
        opt = get_optimizer(backbone.parameters(), **self.inner_opt_args)
        self.me_bptt.register_backbone_and_optimizer(backbone, opt)
        self.me_bptt.forward(num_forward)
        meta_loss = self.compute_meta_loss()
        self.me_bptt.backprop(num_backward)
        return meta_loss



        
