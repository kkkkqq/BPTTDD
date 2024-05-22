import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Dict, Union, Tuple
import copy

class DiffOptimizer():

    def __init__(self)->None:
        self.params_tape:List[List[List[Tensor]]] = None
        self.states_tape:List[List[List[Dict[str,Tensor]]]] = None
        self.cur_idx:int = 0
        self.dLdw_groups:List[Tensor] = None

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
    def flatten(tsr_lst:List[Tensor]):
        """
        flattens a list of tensors
        """
        return torch.cat([tsr.flatten() for tsr in tsr_lst], dim=0)
        
    
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
    
    @staticmethod
    def replace_params(opt:Optimizer, param_groups:List[List[Tensor]]):
        """
        in-place replacing of params in opt.param_groups by param_groups
        """
        pa_groups = opt.param_groups
        with torch.no_grad():
            for pag_idx, pa_group in enumerate(pa_groups):
                pas:List[Tensor] = pa_group['params']
                for pa_idx, pa in enumerate(pas):
                    source = param_groups[pag_idx][pa_idx]
                    pa[:] = source[:]
        return None

    def backward_step(self, 
                      meta_params:List[Tensor], 
                      accum_grad:bool=True, 
                      roll_back:bool=True):
        """
        Take one step backward and compute the meta gradients for meta_params.\\
        If accum_grad, the meta gradients will be added to meta_params[:].grad;
        if not accum_grad, the meta gradients will replace meta_params[:].grad.\\
        If roll_back, the optimizer will roll the backbone model back to its state
        at the previous forward step before computing meta gradients.\\
        The backbone model must have is tracked gradients ready before calling 
        backward_step.
        """
        if roll_back:
            self.roll_back()
        meta_grads = self._backward_meta_grads(meta_params, roll_back)
        if accum_grad:
            for idx, mepa in enumerate(meta_params):
                if mepa.grad is None:
                    mepa.grad = meta_grads[idx]
                else:
                    mepa.grad.add_(meta_grads[idx])
        else:
            for idx, mepa in enumerate(meta_params):
                mepa.grad = meta_grads[idx]
        return meta_grads
    
    def roll_back(self):
        """
        Roll the backbone model back to its state at the previous forward step.
        """
        self.cur_idx -= 1
        param_groups = self.params_tape[self.cur_idx]
        self.replace_params(self, param_groups)
        
        return None
    
    def post_step(self, taped:bool=True):
        """
        All subclasses must call this function in its step() after calling step() of its Optimzer
        father class.\\
        tape determines whether to append current state and params.
        """
        if taped:
            self.states_tape.append(self.read_state(self, True))
            self.params_tape.append(self.read_params(self, True))
        else:
            self.states_tape.append(None)
            self.params_tape.append(None)
        self.cur_idx += 1
        return None
    
    def read_meta_grads(self):
        """
        Reads the meta_grads from backbone after backwarding from meta loss
        at the end of forward process.
        """
        opt:Optimizer = self
        param_groups = opt.param_groups
        with torch.no_grad():
            if self.dLdw_groups is None:
                self.dLdw_groups = []
                for param_group in param_groups:
                    dLdw = torch.cat([ele.grad.flatten() for ele in param_group], dim=0)
                    self.dLdw_groups.append(dLdw)
            else:
                for idx, param_group in enumerate(param_groups):
                    dLdw = torch.cat([ele.grad.flatten() for ele in param_group], dim=0)
                    self.dLdw_groups[idx].add_(dLdw)
        return None



    def _backward_meta_grads(meta_params:List[Tensor], roll_back:bool=True)->List[Tensor]:
        """
        This is the optimizer-specific function for computing the meta gradient.\\
        If roll_back, the computations on dLdw and other backward states will be in-place,
        otherwise the computations are performed on copies.
        """
        raise NotImplementedError

                



