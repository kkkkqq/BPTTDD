import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer, Adam, SGD
from BPTT.diff_adam import DiffAdam
from BPTT.diff_sgd import DiffSGD
from typing import List, Dict, Union, Tuple, Callable

def diff_optimizer(opt:Optimizer)->Union[DiffSGD, DiffAdam]:
    if isinstance(opt, Adam):
        return DiffAdam(opt)
    elif isinstance(opt, SGD):
        return DiffSGD(opt)
    else:
        raise NotImplementedError("Only implemented Differentiable SGD and Adam")

    
class MEBPTT():

    def __init__(self,
                 forward_function_handle:Callable,
                 meta_loss_handle:Callable,
                 backbone:nn.Module=None,
                 optimizer:Optimizer=None,
                 meta_params_lst:List[Union[Tensor, List[Tensor], Dict[str,Tensor]]]=None,
                 meta_params_dict:Dict[str, Union[Tensor, List[Tensor], Dict[str,Tensor]]]=None) -> None:
        """
        A simple memory-efficient BPTT wrapper.\\
        `forward_function_handle`: any forward function that takes `step_idx:int`
        as its first positional arg, `backbone:nn.Module` as the second, and 
        returns `loss:Tensor`\\
        `meta_loss_handle`: any function computing meta loss on backbone. It
        takes `backbone:nn.Module` as the first positional arg and returns 
        `meta_loss:Tensor`.\\
        `backbone`: the model to be optimized in the inner loop\\
        `optimizer`: inner-loop optimizer linked to `backbone`\\
        """
        
        self.forward_function_handle = forward_function_handle
        self.meta_loss_handle = meta_loss_handle
        self.backbone = backbone
        if optimizer is not None:
            self.diff_optimizer = diff_optimizer(optimizer)
        else:
            self.diff_optimizer = None
        self.forward_args:list = None
        self.forward_kwargs:dict = None
        self.meta_params:List[Tensor] = None
        self.meta_params_memo_lst:List[Tuple[int,int,str,str]] = None
        self.meta_params_memo_dict:Dict[str, Tuple[int,int,str,List[str]]] = None
        if meta_params_lst is None:
            meta_params_lst = []
        if meta_params_dict is None:
            meta_params_dict = dict()
        self.register_meta_params(*meta_params_lst, **meta_params_dict)
        return None
    
    def register_backbone_and_optimizer(self, backbone:nn.Module, optimizer:Optimizer):
        """
        reads backbone and optimizer and replace the current ones.
        """
        self.backbone = backbone
        self.diff_optimizer = diff_optimizer(optimizer)
        return None
    
    def register_meta_params(self, *args, **kwargs):
        """
        reads args and kwargs, flatten them into a list and replace the current registered meta_params.
        """
        self.meta_params = []
        start_idx = 0
        for arg in args:
            if isinstance(arg, Tensor):
                end_idx = start_idx + 1
                memo = (start_idx, end_idx, 't', None)
                self.meta_params.append(arg)
                start_idx = end_idx
            elif isinstance(arg, list):
                end_idx = start_idx + len(arg)
                memo = (start_idx, end_idx, 'l', None)
                self.meta_params.extend(arg)
                start_idx = end_idx
            elif isinstance(arg, dict):
                key_lst = []
                end_idx = start_idx
                for key, val in arg.items():
                    end_idx = end_idx + 1
                    self.meta_params.append(val)
                    key_lst.append(key)
                memo = (start_idx, end_idx, 'd', key_lst)
            else:
                raise ValueError("args passed in can only be Tensor|List[Tensor]|Dict[str,Tensor]")
            self.meta_params_memo_lst.append(memo)
        for k, arg in kwargs.items():
            if isinstance(arg, Tensor):
                end_idx = start_idx + 1
                memo = (start_idx, end_idx, 't', None)
                self.meta_params.append(arg)
                start_idx = end_idx
            elif isinstance(arg, list):
                end_idx = start_idx + len(arg)
                memo = (start_idx, end_idx, 'l', None)
                self.meta_params.extend(arg)
                start_idx = end_idx
            elif isinstance(arg, dict):
                key_lst = []
                end_idx = start_idx
                for key, val in arg.items():
                    end_idx = end_idx + 1
                    self.meta_params.append(val)
                    key_lst.append(key)
                memo = (start_idx, end_idx, 'd', key_lst)
            else:
                raise ValueError("args passed in can only be Tensor|List[Tensor]|Dict[str,Tensor]")
            self.meta_params_memo_dict[k] = memo
        return None
            

    def _check_model_opt(self):
        if self.backbone is None or self.diff_optimizer is None:
            raise AssertionError("haven't registered backbone and optimizer!")
        return None
    
    def forward(self, num_steps:int, *args, **kwargs):
        """
        Perform the inner_loop for `num_steps` steps.\\
        Args and kwargs are those passed into forward_function_handle
        apart from step_idx and backbone.\\
        """
        self._check_model_opt()
        self.forward_args = args
        self.forward_kwargs = kwargs
        for _ in range(num_steps):
            self.diff_optimizer.zero_grad()
            step_idx = self.diff_optimizer.cur_idx
            loss = self.forward_function_handle(step_idx = step_idx, backbone = self.backbone, *args, **kwargs)
            self.diff_optimizer.backward(loss)
            self.diff_optimizer.step()
        return None
    
    def meta_loss(self, *args, weight=1, **kwargs):
        """
        Compute meta loss, args and kwargs are passed into meta_loss_handle apart from backbone.\\
        This may be called multiple times and the weighted meta_loss will be accumulated.
        """
        self._check_model_opt()
        self.diff_optimizer.zero_grad()
        meta_loss:Tensor = self.meta_loss_handle(backbone = self.backbone, *args, **kwargs)
        self.diff_optimizer.backward(meta_loss)
        self.diff_optimizer.post_meta_loss_backprop(weight=weight)
        return None
    
    def backprop(self, num_steps:int):
        """
        Backpropagation through the forward process for `num_steps` steps.
        """
        for _ in range(num_steps):
            self.diff_optimizer.roll_back()
            step_idx = self.diff_optimizer.cur_idx
            loss = self.forward_function_handle(step_idx = step_idx, 
                                                backbone = self.backbone, 
                                                *self.forward_args,
                                                **self.forward_kwargs)
            self.diff_optimizer.backward(loss, True, True)
            _ = self.diff_optimizer.backprop_step(self.meta_params, True, True)
        return None


            

    

    



