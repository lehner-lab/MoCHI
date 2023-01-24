
"""
Custom function
"""

import torch

def TwoStateFractionFoldedMod(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional nonlinear transformation relating modified Gibbs free energy of folding to fraction of molecules folded.

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded tensor.
    """  
    if X is None:
        return {}
    else:
        return torch.pow(1+torch.exp(X[0]*torch.abs(X[1])), -1)

def TwoStateFractionFoldedGlobal(
    X = None,
    trainable_parameters = {}):
    """
    1-dimensional nonlinear transformation relating Gibbs free energy of folding to fraction of molecules folded.
    Global effects on folding (e.g. chaperone) concentration is a trained parameter (c).

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded tensor.
    """  
    if X is None:
        return {'c': None}
    else:
        return torch.pow(1+torch.exp(X[0]+trainable_parameters['c']), -1)

def ThreeStateFractionBoundLig(
    X = None,
    trainable_parameters = {}):
    """
    2-dimensional nonlinear transformation relating Gibbs free energy of folding and binding to fraction of molecules folded and bound.
    Ligand concentration is a trained parameter (c).

    :param X: list of tensors (required).
    :param trainable_parameters: dictionary of global parameter names (optional).
    :returns: fraction of molecules folded and bound tensor.
    """  
    if X is None:
        return {'c': None}
    else:
        return torch.pow(1+torch.mul(torch.exp(X[1] - torch.log(trainable_parameters['c'])), 1+torch.exp(X[0])), -1)

