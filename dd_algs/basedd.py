import torch
import torch.nn as nn
from BPTT.me_bptt import MEBPTT
from synset.base_synset import BaseSynSet
from utils import get_model, get_optimizer
from modules.basemodule import BaseModule
from modules.utils import get_module
from typing import Callable

class BaseDDAlg():

    def __init__(self,
                 batch_function:Callable,
                 inner_module_args:BaseModule,
                 inner_model_args:dict,
                 inner_opt_args:dict,
                 inner_batch_size:int,
                 device='cuda'):
        
        self.batch_function = batch_function # takes (step_idx, batch_size, *args, **kwargs)
        self.inner_module_args = inner_module_args
        self.device = device
        self.inner_module:BaseModule = get_module(**self.inner_module_args)
        self.inner_model_args = inner_model_args
        self.inner_opt_args = inner_opt_args
        self.inner_batch_size = inner_batch_size
        self.me_bptt = MEBPTT(self.forward_function_handle, self.meta_loss_handle)
    
    def register_meta_params(self, *args, **kwargs):
        return self.me_bptt.register_meta_params(*args, **kwargs)

    def forward_function_handle(self, step_idx:int, backbone:nn.Module, **forward_kwargs):
        '''
        by default, forward_args and forward_kwargs will be unrolled and passed into batch_function
        '''
        batch_kwargs = forward_kwargs
        batch_out = self.batch_function(batch_idx=step_idx, batch_size=self.inner_batch_size, **batch_kwargs)
        batch_out = self.inner_module.parse_batch(batch_out)
        loss = self.inner_module.forward_loss(backbone, **batch_out)[0]
        return loss
    
    def meta_loss_handle(self, backbone:nn.Module, **kwargs):
        '''
        the meta loss handle that will be called in self.me_bptt.meta_loss(*args,
        **kwargs, weight=1.)
        '''
        raise NotImplementedError
    
    def compute_meta_loss(self, **meta_loss_kwargs):
        '''
        called in self.step(). It calls self.me_bptt.meta_loss.
        override to customize.
        '''
        raise NotImplementedError
    
    def step(self, 
             num_forward:int, 
             num_backward:int, 
             forward_kwargs:dict=dict(),
             meta_loss_kwargs:dict=dict(),
             meta_params_lst:list=None,
             meta_params_dict:dict=None
             ):
        '''
        `forward_kwargs`: kwargs that will be unrolled and passed into `self.forward_function_handle`
        as & `forward_kwargs`. By default, these will then be passed into `self.batch_function`.\\
        `meta_loss_kwargs`: args and kwargs that will be unrolled and passed into `self.compute_meta_loss`.\\
        `meta_params_lst`&`meta_params_dict`: meta params to compute meta grads for. If both are None, meta 
        params will be computed for whatever registered with `self.register_meta_params` earlier.
        '''
        if meta_params_lst is not None or meta_params_dict is not None:
            if meta_params_lst is None:
                meta_params_lst = []
            if meta_params_dict is None:
                meta_params_dict = dict()
            self.register_meta_params(*meta_params_lst, **meta_params_dict)
        backbone = get_model(**self.inner_model_args)
        backbone.to(self.device)
        opt = get_optimizer(backbone.parameters(), **self.inner_opt_args)
        self.me_bptt.register_backbone_and_optimizer(backbone, opt)
        self.me_bptt.forward(num_steps=num_forward, num_taped=num_backward, **forward_kwargs)
        meta_loss = self.compute_meta_loss(**meta_loss_kwargs)
        #mem_before_step = torch.cuda.memory_allocated(0)
        self.me_bptt.backprop(num_backward)
        #mem_after_step = torch.cuda.memory_allocated(0)
        #print('mem before and after backprop: ', mem_before_step, mem_after_step)
        return meta_loss



        
