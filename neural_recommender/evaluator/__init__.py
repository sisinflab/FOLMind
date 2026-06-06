"""
Module description:
    evaluator call
"""

__version__ = '0.0'
__email__ = ''

from .Evaluator_neural import Evaluator, EvaluatorConfig
from .metrics import *

__all__ = ['Evaluator', 'EvaluatorConfig']
