from modules.basemodule import BaseModule
from modules.clfmodule import ClassifierModule

def get_module(module_name:str, module_args:dict):
    if module_name.lower() in ['clfmodule', 'classifiermodule', 'classifier']:
        return ClassifierModule(**module_args)
    else:
        raise NotImplementedError