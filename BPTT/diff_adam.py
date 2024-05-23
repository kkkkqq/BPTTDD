import torch
from torch.optim import Adam
from BPTT.diff_opt import DiffOptimizer
from torch import Tensor
from typing import List, Dict, Union, Tuple

class DiffAdam(DiffOptimizer, Adam):

    def __init__(self, opt:Adam):
        super(DiffOptimizer, self).__init__()
        init_args, added_groups, state_dict = self.read_optimizer(opt)
        super(Adam, self).__init__(**init_args)
        for pa_grp in added_groups:
            self.add_param_group(pa_grp)
        self.load_state_dict(state_dict)
        self.dLdv_groups:List[Tensor] = None
        self.dLdm_groups:List[Tensor] = None
    
    def update_backprop_state(self):
        if self.dLdv_groups is None:
            self.dLdv_groups = []
            for dLdw in self.dLdw_groups:
                self.dLdv_groups.append(torch.zeros_like(dLdw))
        if self.dLdm_groups is None:
            self.dLdm_groups = []
            for dLdw in self.dLdw_groups:
                self.dLdm_groups.append(torch.zeros_like(dLdw))
        
        

