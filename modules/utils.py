from modules.basemodule import BaseModule
from modules.clfmodule import ClassifierModule
from modules.vaemodule import CVAEModule

def get_module(module_name:str, module_args:dict):
    if module_name.lower() in ['clfmodule', 'classifiermodule', 'classifier']:
        return ClassifierModule(**module_args)
    elif module_name.lower() in ['cvae', 'cvaemodule']:
        return CVAEModule(**module_args)
    else:
        raise NotImplementedError