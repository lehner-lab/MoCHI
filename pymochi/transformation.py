
"""
MoCHI report module
"""

import math
import torch
import numpy as np

def get_transformation(
    name):
    """
    Get a transformation function by name.

    :param name: transformation function name (required).
    :returns: transformation function.
    """     
    return eval(name)   

def Linear(
    X):
    """
    1-dimensional linear transformation function.

    :param X: list of tensors (required).
    :returns: first tensor in the input tensor list.
    """   
    return X[0]

def TwoStateFractionFolded(
    X):
    """
    1-dimensional nonlinear transformation relating Gibbs free energy of folding to fraction of molecules folded.

    :param X: list of tensors (required).
    :returns: fraction of molecules folded tensor.
    """  
    return torch.pow(1+torch.exp(X[0]), -1)

def ThreeStateFractionBound(
    X):
    """
    2-dimensional nonlinear transformation relating Gibbs free energy of folding and binding to fraction of molecules folded and bound.

    :param X: list of tensors (required).
    :returns: fraction of molecules folded and bound tensor.
    """  
    return torch.pow(1+torch.mul(torch.exp(X[1]), 1+torch.exp(X[0])), -1)

