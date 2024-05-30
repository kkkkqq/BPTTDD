import os
os.environ['PYTHONHASHSEED'] = str(0)  # 禁止hash随机化
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 在cuda 10.2及以上的版本中，需要设置以下环境变量来保证cuda的结果可复现
import numpy as np
import random
import torch
import torch.backends
torch.use_deterministic_algorithms(True) # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
torch.backends.cudnn.benchmark = False  # 与上面一条代码配套使用，True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现。
import yaml
import argparse
from experiments.experiment import Experiment


    
    
parser = argparse.ArgumentParser(description='Generic runner for Training Synset')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/clfdd.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

exp = Experiment(config)
exp.run()
