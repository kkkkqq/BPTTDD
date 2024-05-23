import torch
import numpy as np
from torch.optim import Adam
from BPTT.diff_opt import DiffOptimizer
from torch import Tensor
from typing import List, Dict, Union, Tuple

class DiffAdam(DiffOptimizer, Adam):

    def __init__(self, opt:Adam):
        super(DiffOptimizer, self).__init__()
        init_args, added_groups, state_dict = self.read_optimizer(opt)
        super(Adam, self).__init__(**init_args)
        self.tape_state = True
        for pa_grp in added_groups:
            self.add_param_group(pa_grp)
        self.load_state_dict(state_dict)
        self.dLdv_groups:List[Tensor] = None
        self.dLdm_groups:List[Tensor] = None
        for group in self.param_groups:
            if group['amsgrad']:
                raise NotImplementedError("Haven't implemented amsgrad!")
    
    def step(self, taped:bool=True):
        """
        step function for DiffAdam.
        """
        super(Adam, self).step()
        self.post_step(self, taped=taped)
        return None

    def update_backprop_state(self):
        if self.dLdv_groups is None:
            self.dLdv_groups = []
            for dLdw in self.dLdw_groups:
                self.dLdv_groups.append(torch.zeros_like(dLdw))
        if self.dLdm_groups is None:
            self.dLdm_groups = []
            for dLdw in self.dLdw_groups:
                self.dLdm_groups.append(torch.zeros_like(dLdw))
        self.dLdgrad_groups = []
        states = self.states_tape[self.cur_idx]
        with torch.no_grad():
            for idx, group in enumerate(self.param_groups):
                gt = self.flatten([ele.grad.detach() for ele in group['params']]).clone()
                lr = group['lr']
                beta1 = group['betas'][0]
                beta2 = group['betas'][1]
                eps = group['eps']
                weight_decay = group['weight_decay']
                maximize = group['maximize']
                dLdw = self.dLdw_groups[idx]
                dLdm = self.dLdm_groups[idx]
                dLdv = self.dLdv_groups[idx]
                state = states[idx]
                m = self.flatten([dct['exp_avg'] for dct in state])
                v = self.flatten([dct['exp_avg_sq'] for dct in state])
                t = self.state[0]['step'].item()
                omb1 = 1.-beta1
                omb1t = 1.- np.power(beta1, t)
                omb2 = 1.-beta2
                omb2t = 1. - np.power(beta2, t)
                if maximize:
                    gt.mul_(-1.)
                if weight_decay!=0:
                    w = self.flatten([ele.detach() for ele in group['params']])
                    gt.add_(w.mul(weight_decay))
                sqrt_vdivomb2t = v.div_(omb2t).pow_(0.5)#v no longer used therefore in-place
                dLdm.mul_(beta1).sub_(dLdw.mul(lr/omb1t).div(sqrt_vdivomb2t.add(eps)))
                dLdv.mul_(beta2).add_(dLdw.mul(lr/omb1t/omb2t/2.0).mul(m).div(sqrt_vdivomb2t).div(sqrt_vdivomb2t.add(eps).pow(2)))
                gt.mul_(2.*omb2).mul_(dLdv).add_(dLdm.mul(omb1))
                if weight_decay!=0:
                    dLdw.add_(gt.mul(weight_decay))
                if maximize:
                    gt.mul_(-1.)
                self.dLdgrad_groups.append(gt)
        return None
            



            
        

