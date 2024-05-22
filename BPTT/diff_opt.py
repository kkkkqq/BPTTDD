import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Dict, Union, Tuple
import copy

class DiffOptimizer():

    def __init__(self)->None:
        self.params_tape:List[List[Tensor]] = None
        self.states_tape:List[List[Dict[str,Tensor]]] = None

    @staticmethod
    def read_optimizer(opt:Optimizer)->Tuple[dict, List[dict], dict]:
        """
        reads an optimizer and returns its (initialization args, additional
        param groups and state dict).
        """
        params_0 = opt.param_groups[0]
        if len(opt.param_groups)>1:
            groups = opt.param_groups[1:]
        else:
            groups = []
        state_dict = opt.state_dict()
        init_args = {params_0}
        added_groups = groups
        return init_args, added_groups, state_dict
    
    @staticmethod
    def flatten(tsr_lst:List[Tensor], detached:bool=False):
        """
        flattens (and detach if detached) a list of tensors
        """
        if not detached:
            flat = torch.cat([tsr.flatten() for tsr in tsr_lst], dim=0)
        else:
            flat = torch.cat([tsr.flatten().detach() for tsr in tsr_lst], dim=0)
        return flat
    
    @staticmethod
    def read_state(opt:Optimizer, copy_state:bool=True)->List[List[Dict[str, Tensor]]]:
        """
        returns a list of list of (copied if copy_state) current states of each params in each param_groups in opt.
        """
        state = opt.state
        grouped_states_lst = []
        for pa_group in opt.param_groups:
            pas = pa_group['params']
            pas_states_lst = []
            for pa in pas:
                if copy_state:
                    sts = copy.deepcopy(state[pa])
                else:
                    sts = state[pa]
                pas_states_lst.append(sts)
            grouped_states_lst.append(pas_states_lst)
        return grouped_states_lst
    
    @staticmethod
    def read_params(opt:Optimizer, copy_params:bool=True)->List[List[Tensor]]:
        """
        returns a list of list of (copied if copy_params) current params in each param_groups in opt.
        """
        pa_groups = opt.param_groups
        grouped_params_lst = []
        for pa_group in pa_groups:
            pas:List[Tensor] = pa_group['params']
            pas_lst = []
            for pa in pas:
                if copy_params:
                    savepa = pa.detach().clone()
                else:
                    savepa = pa
                pas_lst.append(savepa)
            grouped_params_lst.append(pas_lst)
        return grouped_params_lst
                
        
                



