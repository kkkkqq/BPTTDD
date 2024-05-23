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
        self.dLdgrad_groups:List[Tensor] = None
        self.tape_state:bool = None
        # self.add_dLdw_groups:List[Tensor] = None

    @staticmethod
    def read_optimizer(opt:Optimizer)->Tuple[dict, List[dict], dict]:
        """
        reads an optimizer and returns its (initialization args, additional
        param groups and state dict).
        """
        params_0 = opt.param_groups[0]
        groups = opt.param_groups[1:]
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

    def backprop_step(self, 
                      meta_params:List[Tensor], 
                      accum_grad:bool=True, 
                      roll_back:bool=True,
                      update_bp_states:bool=True):
        """
        Take one step backward and compute the meta gradients for meta_params.\\
        If accum_grad, the meta gradients will be added to meta_params[:].grad;
        if not accum_grad, the meta gradients will replace meta_params[:].grad.\\
        If roll_back, the optimizer will roll the backbone model back to its state
        at the previous forward step before computing meta gradients.\\
        The backbone model must have its param.grad ready before calling backprop_step.
        """
        if roll_back:
            self.roll_back()
        if update_bp_states:
            self.update_backprop_state()
            meta_grads = self.backprop_meta_params(meta_params, True)
        else:
            meta_grads = self.backprop_meta_params(meta_params)
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
        taped determines whether to append current state and params.
        """
        if taped:
            if self.tape_state:
                self.states_tape.append(self.read_state(self, True))
            else:
                self.states_tape.append(None)
            self.params_tape.append(self.read_params(self, True))
        else:
            self.states_tape.append(None)
            self.params_tape.append(None)
        self.cur_idx += 1
        return None
    
    def post_meta_loss_backprop(self):
        """
        call after self.backward(meta_loss)
        Reads the grads on backbone from meta_loss and set backprop states.
        """
        opt:Optimizer = self
        param_groups = opt.param_groups
        with torch.no_grad():
            if self.dLdw_groups is None:
                self.dLdw_groups = []
                for param_group in param_groups:
                    dLdw = torch.cat([ele.grad.flatten() for ele in param_group['params']], dim=0)
                    self.dLdw_groups.append(dLdw)
            else:
                for idx, param_group in enumerate(param_groups):
                    dLdw = torch.cat([ele.grad.flatten() for ele in param_group['params']], dim=0)
                    self.dLdw_groups[idx].add_(dLdw)
        return None


    def update_backprop_state(self):
        """
        Optimizer-specific backprop update function, in-place modify all backprop states.
        The dLdw_groups is only partially computed, it requires a further dLdgrad*dgraddw
        to be later computed and added to itself.
        """
        raise NotImplementedError
    
    def backprop_meta_params(self, meta_params:List[Tensor], update_dLdw:bool=True):
        """
        Compute meta gradients for params in meta_params.\\
        Must be precedented by update_backprop_state.\\
        If update_dLdw, 
        """
        params = meta_params
        opt:Optimizer = self
        if update_dLdw:
            start_idx = len(params)
            opt_group_memos:List[Tuple[int,int]] = []
            for group in opt.param_groups:
                pas = group['params']
                end_idx = start_idx + len(pas)
                opt_group_memos.append((start_idx, end_idx))
                params.extend(pas)
                start_idx = end_idx
        grads = []
        for group in opt.param_groups:
            pas = group['params']
            grads.extend([ele.grad for ele in pas])
        grads = self.flatten(grads)
        dLdgrad = self.flatten(self.dLdgrad_groups)
        meta_grads = torch.autograd.grad(outputs=grads,
                                         inputs=meta_params,
                                         grad_outputs=dLdgrad)
        with torch.no_grad():
            if update_dLdw:
                for group_idx, memo in enumerate(opt_group_memos):
                    dLdw = self.flatten(meta_grads[memo[0]:memo[1]])
                    self.dLdw_groups[group_idx].add_(dLdw)
                meta_grads = meta_grads[:opt_group_memos[0][0]]
        
        return meta_grads
    
    def backward(self, loss:Tensor, retain_graph:bool=False, create_graph:bool=False):
        opt:Optimizer = self
        params = []
        for group in opt.param_groups:
            pas = group['params']
            params.extend(pas)
        grads = torch.autograd.grad(outputs=loss,
                                    inputs=params,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)
        for idx, pa in enumerate(params):
            pa.grad = grads[idx]
        return None



            






                



