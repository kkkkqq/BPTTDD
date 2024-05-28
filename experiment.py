import os
import copy
import torch
import random
import torch.utils
import tqdm
import numpy as np
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from utils import get_dataset, get_model, get_optimizer
from modules.basemodule import BaseModule
from modules.clfmodule import ClassifierModule
from dataset.baseset import ImageDataSet
from synset.image_lab_synset import ImageLabSynSet
from dd_algs.clfdd import CLFDDAlg
from typing import Dict
from synset.synset_loader import SynSetLoader

def seed_everything(seed:int):
	#  下面两个常规设置了，用来np和random的话要设置 
    np.random.seed(seed) 
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # 多GPU训练需要设置这个
    torch.manual_seed(seed)

def get_module(module_name:str, module_args:dict):
    if module_name.lower() in ['clfmodule', 'classifiermodule', 'classifier']:
        return ClassifierModule(**module_args)
    else:
        raise NotImplementedError
    
class Experiment():

    def __init__(self, config:dict):
        self.config:dict = config
        # experiment settings
        self.exp_config:dict = self.parse_exp_config()
        self.seed:int = self.exp_config['seed']
        self.project:str = self.exp_config['project']
        self.exp_name:str = self.exp_config['exp_name'] + 'seed' + str(self.seed)
        self.save_dir:str = './results/'+self.exp_name
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        seed_everything(self.seed)
        self.use_wandb:bool = self.exp_config['use_wandb']
        self.wandb_api_key:str = self.exp_config['wandb_api_key']
        self.num_steps:int = self.exp_config['num_steps']
        self.device = self.exp_config['device']

        #dataset settings
        self.dataset_config:dict = self.parse_dataset_config()
        self.dataset:ImageDataSet = get_dataset(**self.dataset_config)
        self.dataset_aux_args = {'channel': self.dataset.channel,
                                 'num_classes': self.dataset.num_classes,
                                 'image_size': self.dataset.image_size}
        self.test_loader = DataLoader(self.dataset.dst_test, 512, False)

        #evaluation settings
        self.eval_config:dict = self.parse_eval_config()
        

        self.synset_config = self.parse_synset_config()
        #init synset and their optimizers

        self.dd_config = self.parse_dd_config()
        #set dd args and initiate a dd



    def parse_dataset_config(self)->dict:
        dataset_config = copy.deepcopy(self.config['dataset_config'])
        return dataset_config


    def parse_exp_config(self)->dict:
        return copy.deepcopy(self.config['exp_config'])

    def parse_synset_config(self)->dict:
        synset_config = copy.deepcopy(self.config['synset_config'])
        synset_type:str = synset_config['synset_type']
        if synset_type.lower() in ['imagelab', 'imagelabsynset']:
            synset_args:dict = synset_config['synset_args']
            synset_args.update(self.dataset_aux_args)
            synset_args['zca'] = self.dataset.zca_trans
            synset_args['device'] = self.device
            synset_args['real_loader'] = DataLoader(self.dataset.dst_train, 256, False)
            self.synset = ImageLabSynSet(**synset_args)
            return synset_config
        else:
            raise NotImplementedError
        

    def parse_dd_config(self)->dict:
        dd_config:dict = copy.deepcopy(self.config['dd_config'])
        self.bptt_type:str = dd_config['bptt_type']
        self.num_forward:int = dd_config['num_forward']
        self.num_backward:int = dd_config['num_backward']
        self.ema_grad_clip:bool = dd_config['ema_grad_clip']
        self.ema_coef:float = dd_config['ema_coef']
        self.inner_module:BaseModule = get_module(**dd_config['inner_module_args'])
        ddalg_config:dict = dd_config['ddalg_config']
        ddalg_type:str = ddalg_config['ddalg_type']
        if ddalg_type in ['clfdd', 'classifier', 'classifierdd', 'clf']:
            ddalg_args = ddalg_config['ddalg_args']
            ddalg_args['synset'] = self.synset
            ddalg_args['inner_module'] = self.inner_module
            ddalg_args['real_dataset'] = self.dataset
            if 'channel' not in ddalg_args['inner_model_args']:
                ddalg_args['inner_model_args'].update(self.dataset_aux_args)
            self.ddalg = CLFDDAlg(**ddalg_args)
            return ddalg_config
        else:
            raise NotImplementedError
    
    def parse_eval_config(self)->dict:
        self.eval_config = copy.deepcopy(self.config['eval_config'])
        self.eval_interval:int = self.eval_config['eval_interval']
        self.num_eval:int = self.eval_config['num_eval']
        self.eval_models_dict:dict = self.eval_config['eval_models']
        for args in self.eval_models_dict.values():
            if 'channel' not in args['model_args']:
                args['model_args'].update(self.dataset_aux_args)
        self.eval_steps:int = self.eval_config['eval_steps']
        self.test_module:BaseModule = get_module(**self.eval_config['test_module_args'])
        self.upload_visualize:bool = self.eval_config['upload_visualize']
        self.upload_visualize_interval:int = self.eval_config['upload_visualize_interval']
        self.save_visualize:bool = self.eval_config['save_visualize']
        self.save_visualize_interval:bool = self.eval_config['save_visualize_interval']
        return self.eval_config

    def evaluate_synset(self)->Dict[str,float]:
        synset_loader = SynSetLoader(self.synset, self.synset.num_items, self.synset.num_items)
        metrics = dict()
        for name, args in self.eval_models_dict.items():
            print('evaluating synset on', name,':')
            model_args = args['model_args']
            opt_args = args['opt_args']
            mean_train_metric = dict()
            mean_test_metric = dict()
            for _ in range(self.num_eval):
                model = get_model(**model_args)
                model.to(self.device)
                opt = get_optimizer(**opt_args)
                for _ in tqdm.tqdm(range(self.eval_steps-1)):
                    self.test_module.epoch(model, opt, synset_loader, False, True)
                train_metric = self.test_module.epoch(model, opt, synset_loader, True, True)
                for key, val in train_metric.items():
                    print('train', key, ':', val)
                    if key not in mean_train_metric:
                        mean_train_metric[key] = val/self.num_eval
                    else:
                        mean_train_metric[key] += val/self.num_eval
                test_metric = self.test_module.epoch(model, opt, self.test_loader, True, False)
                for key, val in test_metric.items():
                    print('test', key, ':', val)
                    if key not in mean_test_metric:
                        mean_test_metric[key] = val/self.num_eval
                    else:
                        mean_test_metric[key] += val/self.num_eval
            for key, val in mean_train_metric.items():
                print('mean train', key, 'for', name, ':', val)
            for key, val in mean_test_metric.items():
                print('mean test', key, 'for', name, ':', val)
            metrics.update({'eval/'+name+'_train_'+ key:val for key,val in mean_train_metric.items()})
            metrics.update({'eval/'+name+'_test_'+ key:val for key,val in mean_test_metric.items()})    
        return metrics         

    def run(self):
        if self.use_wandb:
            wandb.login(key=self.wandb_api_key)
            wandb.init(project=self.project,
                       reinit=True,
                       name=self.exp_name,
                       config=self.config)
        trainables = self.synset.trainables
        optimizers = self.synset.make_optimizers()
        #prepare for ema grad clipping
        if self.ema_grad_clip:
            ema_dict = dict()
            for key, val in trainables.items():
                if isinstance(val, torch.Tensor):
                    ema_dict[key] = -1e5
                elif isinstance(val, list):
                    ema_dict[key] = [-1e5 for _ in val]
                elif isinstance(val, dict):
                    ema_dict[key] = {k:-1e5 for k in val.keys()}
        for it in range(self.num_steps):
            if it%self.eval_interval == 0:
                metrics = self.evaluate_synset()
                if self.use_wandb:
                    wandb.log(metrics, step=it)
            upload_vis = self.use_wandb and self.upload_visualize and it%self.upload_visualize_interval==0
            save_vis = self.save_visualize and it%self.save_visualize_interval==0
            if upload_vis or save_vis:
                disp_imgs = self.synset.image_for_display(10)
            if upload_vis:
                wandb.log({'Images': wandb.Image(disp_imgs)}, step=it)
                wandb.log({'Pixels': wandb.Histogram(disp_imgs)}, step=it)
            if save_vis:
                disp_imgs.div_(torch.max(torch.abs(disp_imgs))*2).add_(0.5)
                save_image(disp_imgs, self.save_dir+'/'+str(it)+'.jpg')
                torch.save(copy.deepcopy(self.synset, self.save_dir+'/'+str(it)+'.pt'))
            for opt in optimizers.values():
                opt.zero_grad()
            
            if self.bptt_type.lower() in ['bptt', 'tbptt']:
                self.ddalg.step(self.num_forward, self.num_backward)
            elif self.bptt_type.lower() in ['ratbptt', 'rat_bptt']:
                num_forward = np.random.randint(self.num_backward, self.num_forward)
                self.ddalg.step(num_forward, self.num_backward)

            #ema
            if self.ema_grad_clip:
                for key, val in trainables.items():
                    if isinstance(val, torch.Tensor):
                        if val.grad is not None:
                            if not torch.all(val.grad==0):
                                shadow = ema_dict[key]
                                if shadow == -1e5:
                                    ema_dict[key] = torch.norm(val.grad).item()
                                else:
                                    shadow -= (1 - self.ema_coef) * (shadow - torch.norm(val.grad).item())
                                torch.nn.utils.clip_grad_norm_(val, max_norm = 2*ema_dict[key])
                    elif isinstance(val, list):
                        for idx, tsr in enumerate(val):
                            if tsr.grad is not None:
                                if not torch.all(tsr.grad==0):
                                    shadow = ema_dict[key][idx]
                                    if shadow == -1e5:
                                        ema_dict[key][idx] = torch.norm(tsr.grad)
                                    else:
                                        shadow -= (1 - self.ema_coef) * (shadow - torch.norm(tsr.grad))
                                    torch.nn.utils.clip_grad_norm_(tsr, max_norm = 2*ema_dict[key][idx])
                    elif isinstance(val, dict):
                        for k,v in val.items():
                            if v.grad is not None:
                                if not torch.all(v.grad==0):
                                    shadow = ema_dict[key][k]
                                    if shadow == -1e5:
                                        ema_dict[key][k] = torch.norm(v.grad)
                                    else:
                                        shadow -= (1 - self.ema_coef) * (shadow - torch.norm(v.grad))
                                    torch.nn.utils.clip_grad_norm_(v, max_norm = 2*ema_dict[key][k])
            for opt in optimizers.values():
                opt.step()
                
            
        
        


        