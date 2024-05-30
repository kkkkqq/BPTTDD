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
from modules.utils import get_module
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
        self.meta_loss_batchsize:int = dd_config['meta_loss_batchsize']
        ddalg_config:dict = dd_config['ddalg_config']
        ddalg_type:str = ddalg_config['ddalg_type']
        if ddalg_type in ['clfdd', 'classifier', 'classifierdd', 'clf']:
            ddalg_args = ddalg_config['ddalg_args']
            if 'channel' not in ddalg_args['inner_model_args']:
                ddalg_args['inner_model_args'].update(self.dataset_aux_args)
            if 'inner_batch_size' in ddalg_args:
                if ddalg_args['inner_batch_size'] is None:
                    ddalg_args['inner_batch_size'] = self.synset.num_items
            else:
                ddalg_args['inner_batch_size'] = self.synset.num_items
            if 'device' in ddalg_args:
                if ddalg_args['device'] is None:
                    ddalg_args['device'] = self.device
            else:
                ddalg_args['device'] = self.device
            ddalg_args['batch_function'] = self.synset.batch
            self.real_loader = DataLoader(self.dataset.dst_train, self.meta_loss_batchsize, True)
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
                opt = get_optimizer(model.parameters(), **opt_args)
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
                print('mean train', key, 'for', name, ':', round(val, 4))
            for key, val in mean_test_metric.items():
                print('mean test', key, 'for', name, ':', round(val,4))
            metrics.update({'eval/'+name+'_train_'+ key:val for key,val in mean_train_metric.items()})
            metrics.update({'eval/'+name+'_test_'+ key:val for key,val in mean_test_metric.items()})    
        return metrics         

    def save_synset(self, path):
        synsetcopy = copy.deepcopy(self.synset)
        synsetcopy.to('cpu')
        torch.save(synsetcopy, path)
        return None



    def run(self):
        if self.use_wandb:
            wandb.login(key=self.wandb_api_key)
            wandb.init(project=self.project,
                       reinit=True,
                       name=self.exp_name,
                       config=self.config)
        trainables = self.synset.trainables
        optimizers = self.synset.make_optimizers()
        self.synset.train()
        self.ddalg.register_meta_params(**trainables)
        #prepare for ema grad clipping
        if self.ema_grad_clip:
            ema_dict = dict()
            norm_dict = dict()
            for key, val in trainables.items():
                if isinstance(val, torch.Tensor):
                    ema_dict[key] = -1e5
                    norm_dict[key] = -1e5
                elif isinstance(val, list):
                    ema_dict[key] = [-1e5 for _ in val]
                    norm_dict[key] = [-1e5 for _ in val]
                elif isinstance(val, dict):
                    ema_dict[key] = {k:-1e5 for k in val.keys()}
                    norm_dict[key] = {k:-1e5 for k in val.keys()}
                else:
                    raise TypeError("trainables can only be Dict[str, Union[Tensor, List[Tensor], Dict[Tensor]]]!")
        for it in range(self.num_steps):
            self.synset.shuffle()
            if (it+1)%self.eval_interval == 0:
                metrics = self.evaluate_synset()
                if self.use_wandb:
                    wandb.log(metrics, step=it)
            upload_vis = self.use_wandb and self.upload_visualize and (it+1)%self.upload_visualize_interval==0
            save_vis = self.save_visualize and (it+1)%self.save_visualize_interval==0
            if upload_vis or save_vis:
                disp_imgs = self.synset.image_for_display(10)
            if upload_vis:
                wandb.log({'Images': wandb.Image(disp_imgs)}, step=it)
                wandb.log({'Pixels': wandb.Histogram(disp_imgs)}, step=it)
            if save_vis:
                disp_imgs.div_(torch.max(torch.abs(disp_imgs))*2).add_(0.5)
                save_image(disp_imgs, self.save_dir+'/'+str(it)+'.jpg')
                self.save_synset(self.save_dir+'/current_synset.pt')


            for opt in optimizers.values():
                opt.zero_grad()
            
            
            if self.bptt_type.lower() in ['bptt', 'tbptt']:
                meta_loss = self.ddalg_step(self.num_forward, self.num_backward)
            elif self.bptt_type.lower() in ['ratbptt', 'rat_bptt']:
                num_forward = np.random.randint(self.num_backward, self.num_forward)
                meta_loss = self.ddalg_step(num_forward, self.num_backward)
            
            print('meta_loss at it {}: {}'.format(it, meta_loss))
            if self.use_wandb:
                wandb.log({'train/meta_loss': meta_loss}, step=it)

            #ema
            if self.ema_grad_clip:
                for key, val in trainables.items():
                    if isinstance(val, torch.Tensor):
                        # if val.grad is not None:
                        #     if not torch.all(val.grad==0):
                        shadow = ema_dict[key]
                        norm_dict[key] = torch.norm(val.grad).item()
                        if shadow == -1e5:
                            ema_dict[key] = norm_dict[key]
                        else:
                            ema_dict[key] -= (1 - self.ema_coef) * (shadow - norm_dict[key])
                        torch.nn.utils.clip_grad_norm_(val, max_norm = 2*ema_dict[key])
                    elif isinstance(val, list):
                        for idx, tsr in enumerate(val):
                            # if tsr.grad is not None:
                            #     if not torch.all(tsr.grad==0):
                            shadow = ema_dict[key][idx]
                            norm_dict[key][idx] = torch.norm(tsr.grad).item()
                            if shadow == -1e5:
                                ema_dict[key][idx] = norm_dict[key][idx]
                            else:
                                ema_dict[key][idx] -= (1 - self.ema_coef) * (shadow - norm_dict[key][idx])
                            torch.nn.utils.clip_grad_norm_(tsr, max_norm = 2*ema_dict[key][idx])
                    elif isinstance(val, dict):
                        for k,v in val.items():
                            # if v.grad is not None:
                            #     if not torch.all(v.grad==0):
                            shadow = ema_dict[key][k]
                            norm_dict[key][k] = torch.norm(v.grad).item()
                            if shadow == -1e5:
                                ema_dict[key][k] = norm_dict[key][k]
                            else:
                                ema_dict[key][k] -= (1 - self.ema_coef) * (shadow - norm_dict[key][k])
                            torch.nn.utils.clip_grad_norm_(v, max_norm = 2*ema_dict[key][k])
                    else:
                        raise TypeError("values of trainables can only be Tensor|List[Tensor]|Dict[str,Tensor]")
                
                upload_gradema_dct = dict()
                for key, val in ema_dict.items():
                    name = key
                    if isinstance(val, list):
                        for idx, ema in enumerate(val):
                            newname = name + str(idx)
                            upload_gradema_dct[newname] = ema
                    elif isinstance(val, dict):
                        for k,v in val.items():
                            newname = name + k
                            upload_gradema_dct[k] = v
                    elif isinstance(val, float):
                        upload_gradema_dct[name] = val
                upload_gradema_dct = {'GradEMA/'+k:v for k,v in upload_gradema_dct.items()}
                if self.use_wandb:
                    wandb.log(upload_gradema_dct, step=it)
            
                upload_gradnorm_dct = dict()
                for key, val in norm_dict.items():
                    name = key
                    if isinstance(val, list):
                        for idx, nm in enumerate(val):
                            newname = name + str(idx)
                            upload_gradnorm_dct[newname] = nm
                    elif isinstance(val, dict):
                        for k,nm in val.items():
                            newname = name + k
                            upload_gradnorm_dct[k] = nm
                    elif isinstance(val, float):
                        upload_gradnorm_dct[name] = val
                upload_gradnorm_dct = {'GradNorm/'+k:v for k,v in upload_gradnorm_dct.items()}
                if self.use_wandb:
                    wandb.log(upload_gradnorm_dct, step=it)
                
            for opt in optimizers.values():
                opt.step()
                
    def ddalg_step(self, num_forward, num_backward):
        if isinstance(self.ddalg, CLFDDAlg):
            return self.ddalg.step(num_forward, num_backward, meta_loss_kwargs={'dataloader': self.real_loader})
        else:
            raise NotImplementedError
        
        


        