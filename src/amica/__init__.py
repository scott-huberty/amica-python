from . import datasets, utils
from ._sklearn_interface import AMICA
from .core import fit_amica

__all__ = ['fit_amica', 'AMICA', 'datasets', 'utils']
