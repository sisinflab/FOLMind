"""
Module description:
    models call
"""

__version__ = '0.0'
__email__ = ''

import re
from .BaseModelConfig import BaseModelConfig
from .MF_Elliot import MFElliotConfig, MatrixFactorizationElliot


# Config registry - add here the class for the config of the model
MODEL_CONFIG_REGISTRY = {
    "MF_Elliot": MFElliotConfig,
}

# Model registry - add here the class name of the model
MODEL_CLASS_REGISTRY = {
    "MF_Elliot": MatrixFactorizationElliot,
}


def get_model_config_class(name: str):
    if name in MODEL_CONFIG_REGISTRY:
        return MODEL_CONFIG_REGISTRY[name]

    base_name = re.match(r"^([A-Za-z0-9_]+)", name).group(1)
    if base_name in MODEL_CONFIG_REGISTRY:
        return MODEL_CONFIG_REGISTRY[base_name]

    parts = base_name.split('_')
    for i in range(len(parts) - 1, 0, -1):
        shorter_name = '_'.join(parts[:i])
        if shorter_name in MODEL_CONFIG_REGISTRY:
            return MODEL_CONFIG_REGISTRY[shorter_name]

    raise ValueError(f"No config class registered for '{name}' or its derivatives.")


def get_model_class(name: str):
    if name in MODEL_CLASS_REGISTRY:
        return MODEL_CLASS_REGISTRY[name]

    base_name = re.match(r"^([A-Za-z0-9_]+)", name).group(1)
    if base_name in MODEL_CLASS_REGISTRY:
        return MODEL_CLASS_REGISTRY[base_name]

    parts = base_name.split('_')
    for i in range(len(parts) - 1, 0, -1):
        shorter_name = '_'.join(parts[:i])
        if shorter_name in MODEL_CLASS_REGISTRY:
            return MODEL_CLASS_REGISTRY[shorter_name]

    raise ValueError(f"No model class registered for '{name}' or its derivatives.")


__all__ = [
    'MF_Elliot', 'MFElliotConfig',
    'BaseModelConfig',
    'MODEL_CONFIG_REGISTRY',
    'MODEL_CLASS_REGISTRY',
    'get_model_config_class',
    'get_model_class'
]





