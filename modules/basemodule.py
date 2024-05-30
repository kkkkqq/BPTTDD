import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Any
from torch.optim import Optimizer
from torch.utils.data import DataLoader

class BaseModule():

    def __init__(self):
        self._device=None

    def forward_loss(self, backbone:nn.Module, *args, **kwargs)->tuple:
        '''
        returns loss, aux1, aux2, ...
        '''
        raise NotImplementedError
    
    def parse_batch(self, batch_out:tuple):
        '''
        parse batch into named dict
        '''
        raise NotImplementedError
    
    def post_loss(self, loss, *args)->Tuple[int, dict]:
        '''
        pass in whatever comes out of forward_loss and compute auxiliary metrics.
        Returns batch_size and dict of metrics including 'loss'
        '''
        raise NotImplementedError
    
    def batch(self, backbone:nn.Module, opt:Optimizer, *args, train:bool=False, record_metric:bool=False, **kwargs):
        if train:
            forward_out = self.forward_loss(backbone, *args, **kwargs)
            loss:Tensor = forward_out[0]
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                forward_out = self.forward_loss(backbone, *args, **kwargs)
        if record_metric:
            return self.post_loss(*forward_out)
        else:
            return None
    
    def epoch(self, backbone:nn.Module, opt:Optimizer, loader:DataLoader, record_metric:bool=False, train:bool=False):
        if train:
            backbone.train()
        else:
            backbone.eval()
        metric_dict = dict()
        num_data = 0
        for item in loader:
            outs = self.batch(backbone, opt, *item, train=train, record_metric=record_metric)
            if record_metric:
                num_data += outs[0]
                out_dict = outs[1]
                for key, val in out_dict.items():
                    if key in metric_dict:
                        metric_dict[key] += val
                    else:
                        metric_dict[key] = val
        if record_metric:
            for key in metric_dict.keys():
                metric_dict[key] /= num_data
            return metric_dict
        else:
            return None







        



