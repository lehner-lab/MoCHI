
"""
MoCHI transormation module
"""

import math
import torch
import numpy as np

def get_transformation(
    name,
    custom = {}):
    """
    Get a transformation function by name.

    :param name: transformation function name (required).
    :param custom: dictionary of custom transformations where keys are function names and values are functions.
    :returns: transformation function.
    """
    if name in custom.keys():
        return custom[name]
    else:
        return eval(name)   

def Linear(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional linear transformation function.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: first tensor in the input tensor list.
    """   
    if X is None:
        return {}
    else:
        return X[0]

def ReLU(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional rectified linear unit (ReLU) function.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: ReLU applied to first tensor in the input tensor list.
    """  
    if X is None:
        return {}
    else:
        m = torch.nn.ReLU()
        return m(X[0])

def SiLU(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional sigmoid Linear Unit (SiLU) or swish function.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: SiLU applied to first tensor in the input tensor list.
    """  
    if X is None:
        return {}
    else:
        m = torch.nn.SiLU()
        return m(X[0])

def Sigmoid(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional sigmoid function.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: Sigmoid function applied to first tensor in the input tensor list.
    """  
    if X is None:
        return {}
    else:
        return torch.sigmoid(X[0])

def SumOfSigmoids(
    X = None,
    trainable_parameters = {}):
    """
    Dummy function.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: first tensor in the input tensor list.
    """   
    if X is None:
        return {}
    else:
        return X[0]

def TwoStateFractionFolded(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional nonlinear transformation relating Gibbs free energy of folding to fraction of molecules folded.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded tensor.
    """  
    if X is None:
        return {}
    else:
        return torch.pow(1+torch.exp(X[0]), -1)
        
def ThreeStateFractionBound(
    X = None,
    trainable_parameters = {}):
    """
    2-dimensional nonlinear transformation relating Gibbs free energy of folding and binding to fraction of molecules folded and bound.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded and bound tensor.
    """  
    if X is None:
        return {}
    else:
        return torch.pow(1+torch.mul(torch.exp(X[1]), 1+torch.exp(X[0])), -1)

def FourStateFractionBound(
    X = None,
    trainable_parameters = {}):
    """
    3-dimensional nonlinear transformation relating Gibbs free energy of folding and binding and binding2 to fraction of molecules folded and bound.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded and bound tensor.
    """  
    if X is None:
        return {}
    else:
        return torch.div(torch.exp(-X[0]-X[1]), 1+torch.exp(-X[0])+torch.exp(-X[0]-X[1])+torch.exp(-X[0]-X[2]))

