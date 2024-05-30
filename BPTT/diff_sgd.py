import torch
import numpy as np
from torch.optim import SGD
from BPTT.diff_opt import DiffOptimizer
from torch import Tensor
from typing import List, Dict, Union, Tuple

class DiffSGD(DiffOptimizer, SGD):

    def __init__(self, opt:SGD):
        DiffOptimizer.__init__(self)
        init_args, added_groups, state_dict = self.read_optimizer(opt)
        SGD.__init__(self, **init_args)
        self.tape_state = False
        for pa_grp in added_groups:
            self.add_param_group(pa_grp)
        self.load_state_dict(state_dict)
        self.dLdv_groups:List[Tensor] = None
        
    
    def step(self, taped:bool=True):
        """
        step function for DiffAdam.
        """
        self.pre_step(taped=taped)
        SGD.step(self)
        self.post_step(taped=taped)
        return None
    
    def update_backprop_state(self):
        if self.dLdv_groups is None:
            self.dLdv_groups = []
            for dLdw in self.dLdw_groups:
                self.dLdv_groups.append(torch.zeros_like(dLdw))
        self.dLdgrad_groups = []
        with torch.no_grad():
            for idx, group in enumerate(self.param_groups):
                lr = group['lr']
                momentum = group['momentum']
                dampening = group['dampening']
                weight_decay = group['weight_decay']
                nesterov = group['nesterov']
                maximize = group['maximize']
                dLdw = self.dLdw_groups[idx]
                dLdv = self.dLdv_groups[idx]
                if maximize:
                    lr *= -1
                if momentum == 0:
                    self.dLdgrad_groups.append(dLdw.mul(-lr))
                    if weight_decay!=0:
                        dLdw.mul_(1.-weight_decay*lr)
                else:
                    if nesterov:
                        dLdv.mul_(momentum).sub_(dLdw.mul(momentum*lr))
                        self.dLdgrad_groups.append(dLdw.mul(-lr).add(dLdv.mul(1-dampening)))
                        if weight_decay != 0:
                            dLdw.mul_(1.-weight_decay*lr).add_(dLdv.mul((1.-dampening)*weight_decay))
                    else:
                        dLdv.mul_(momentum).sub_(dLdw.mul(lr))
                        self.dLdgrad_groups.append(dLdv.mul(1.-dampening))
                        if weight_decay != 0:
                            dLdw.add_(dLdv.mul(weight_decay*(1.-dampening)))
                return None


                
    
