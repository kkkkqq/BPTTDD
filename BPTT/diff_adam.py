import torch
import numpy as np
from torch.optim import Adam
from BPTT.diff_opt import DiffOptimizer
from torch import Tensor
from typing import List, Dict, Union, Tuple

class DiffAdam(DiffOptimizer, Adam):

    def __init__(self, opt:Adam):
        DiffOptimizer.__init__(self)
        init_args, added_groups, state_dict = self.read_optimizer(opt)
        Adam.__init__(self, **init_args)
        self.tape_state = True
        for pa_grp in added_groups:
            self.add_param_group(pa_grp)
        self.load_state_dict(state_dict)
        self.dLdv_groups:List[Tensor] = None
        self.dLdm_groups:List[Tensor] = None
        for group in self.param_groups:
            if group['amsgrad']:
                raise NotImplementedError("Haven't implemented amsgrad!")
        self.use_grad_in_backprop = True
    
    def step(self, taped:bool=True):
        """
        step function for DiffAdam.
        """
        self.pre_step(taped=taped)
        Adam.step(self)
        self.post_step(taped=taped)
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
                # gt = self.flatten([ele.grad.detach() for ele in group['params']]).detach()
                gt = torch.cat([ele.grad.detach().flatten() for ele in group['params']]).detach()
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
                m = torch.cat([dct['exp_avg'].flatten() for dct in state])
                v = torch.cat([dct['exp_avg_sq'].flatten() for dct in state])
                m.add_(1e-8)# offset m and v by a very small amount, keep dLdv from exploding
                v.add_(1e-16)
                t = state[0]['step'].item()
                omb1 = 1.-beta1
                omb1t = 1.- np.power(beta1, t)
                omb2 = 1.-beta2
                omb2t = 1. - np.power(beta2, t)
                if maximize:
                    gt.mul_(-1.)
                if weight_decay!=0:
                    w = torch.cat([ele.detach().flatten() for ele in group['params']])
                    gt.add_(w.mul(weight_decay))
                sqrt_vdivomb2t = v.div_(omb2t).pow_(0.5)
                # print('max m', torch.max(torch.abs(m)).item())
                # print('min m', torch.min(torch.abs(m)).item())
                # print('max v', torch.max(torch.abs(v)).item())
                # print('min v', torch.min(torch.abs(v)).item())
                # print('max sqrtvdivomb2t', torch.max(torch.abs(sqrt_vdivomb2t)).item())
                # print('min sqrtvdivomb2t', torch.min(torch.abs(sqrt_vdivomb2t)).item())
                dLdm.mul_(beta1).sub_(dLdw.mul(lr/omb1t).div(sqrt_vdivomb2t.add(eps)))
                dLdv.mul_(beta2)
                # print('max dLdv before adding', torch.max(torch.abs(dLdv)).item())
                dLdv_add = m.mul_(lr/omb1t/omb2t/2.0).mul_(dLdw)
                # print('max additive before division', torch.max(torch.abs(dLdv_add)).item())
                dLdv_add.div_(sqrt_vdivomb2t)
                # print('max additive after first division', torch.max(torch.abs(dLdv_add)).item())
                dLdv_add.div_(sqrt_vdivomb2t.add_(eps).pow_(2))# this is where dLdv explodes if v and m are not slighted offset.
                # print('epsilon: ', eps)
                # print('max second division:', torch.max(torch.abs(sqrt_vdivomb2t.add(eps).pow(2))).item())
                # print('min second division:', torch.min(torch.abs(sqrt_vdivomb2t.add(eps).pow(2))).item())
                # print('max additive after division', torch.max(torch.abs(dLdv_add)).item())
                #dLdv.add_(dLdw.mul(m.mul(lr/omb1t/omb2t/2.0)).div(sqrt_vdivomb2t).div(sqrt_vdivomb2t.add(eps).pow(2)))
                dLdv.add_(dLdv_add)
                # print('max gt', torch.max(torch.abs(gt)).item())
                # print('max dLdv', torch.max(torch.abs(dLdv)).item())
                # print('max dLdm', torch.max(torch.abs(dLdm)).item())
                # print('omb2', omb2)
                # print('omb1', omb1)
                gt.mul_(2.*omb2).mul_(dLdv).add_(dLdm.mul(omb1))
                if weight_decay!=0:
                    dLdw.add_(gt.mul(weight_decay))
                if maximize:
                    gt.mul_(-1.)
                self.dLdgrad_groups.append(gt)
                # print('max dLdw', torch.max(torch.abs(dLdw)).item())
                # print('max dLdgt', torch.max(torch.abs(gt)).item())
        return None
            



            
        

