
"""
MoCHI data module
"""

import os
import json
import hashlib
import re
import copy
import random
import math
import tempfile
import time
import gc
import csv
import linecache
import queue
import threading
from os.path import exists
import pyreadr
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pathlib
from pathlib import Path
from pymochi.transformation import get_transformation
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import itertools
import collections, functools, operator

import types
from inspect import getmembers, isfunction

def current_process_rss_gb():
    """
    Return current process resident memory in GiB when available.

    :returns: Float or None.
    """
    try:
        with open("/proc/self/status", "r", encoding = "utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)
    except OSError:
        return None
    return None

def log_process_memory(
    label):
    """
    Print a simple memory checkpoint message.

    :param label: Human-readable stage label (required).
    :returns: Nothing.
    """
    rss_gb = current_process_rss_gb()
    if rss_gb is None:
        print(f"{label} | RSS unavailable")
    else:
        print(f"{label} | RSS_GiB={rss_gb:.2f}")

def compact_feature_tensors():
    """
    Decide whether binary feature tensors should be kept as uint8 on host.

    Set MOCHI_FEATURES_UINT8=0|1 to override.

    :returns: Boolean.
    """
    compact_override = os.environ.get("MOCHI_FEATURES_UINT8", "0").lower()
    return compact_override in ["1", "true", "yes", "on"]

def get_feature_store_backend():
    """
    Select the retained-feature storage backend.

    Set MOCHI_FEATURE_STORE to "sparse" or "lazy". Invalid values fall back to
    the sparse backend.

    :returns: Backend name.
    """
    backend = os.environ.get("MOCHI_FEATURE_STORE", "sparse").lower()
    if backend not in ["sparse", "lazy"]:
        return "sparse"
    return backend

class FeatureMatrixMetadata:
    """
    Lightweight metadata wrapper for a lazily materialized feature matrix.
    """
    def __init__(
        self,
        index,
        columns):
        self.index = index
        self.columns = pd.Index(columns)
        self.shape = (len(index), len(self.columns))

    def __len__(
        self):
        return self.shape[0]

class CustomTransformations:
    """
    A class for custom transformations.
    """
    def __init__(
        self, 
        code_string):
        """
        Initialize a CustomTransformations object.

        :param code_string: code string (required).
        :returns: CustomTransformations object.
        """
        #Import code as module
        temp_module = self.import_code(code_string)
        #Save functions in dictionary
        self.transformations = {i[0]:i[1] for i in getmembers(temp_module) if isfunction(i[1])}
        #Check functions valid
        invalid_functions = self.check_transformations()
        if invalid_functions!=[]:
            print("Error: Invalid custom transformations: "+",".join(invalid_functions))
            raise ValueError                  

    def import_code(
        self, 
        code, 
        name = "temp_module"):
        """
        Import code as module.

        :param code: string code to import (required).
        :param name: module name (default:'temp_module').
        :returns: imported module.
        """
        #Create blank module
        module = types.ModuleType(name)
        #Populate the module with code
        exec(code, module.__dict__)
        return module

    def check_transformations(
        self):
        """
        Check custom transformations are valid functions.

        :returns: list of transformation names not satisfying mochi requirements.
        """
        #Check dictionary returned when no input tensor supplied
        bad_trans = [tname for tname in self.transformations.keys() if type(self.transformations[tname]())!=dict]
        #Check tensor returned when list of tensors supplied
        for i in self.transformations.keys():
            if i not in bad_trans:
                param_dict = {p:torch.tensor(np.asarray([1.0]), dtype=torch.float32) for p in self.transformations[i]().keys()}
                bad_trans += [tname for tname in [i] if type(self.transformations[i](
                    X = [torch.tensor(np.asarray([1.0, 1.0]), dtype=torch.float32)]*100, 
                    trainable_parameters = param_dict))!=torch.Tensor]
        return bad_trans

class FitnessData:
    """
    A class for the storage of fitness data from a single DMS experiment.
    """
    def __init__(
        self, 
        file_path, 
        name = None, 
        order_subset = None,
        downsample_observations = None,
        seed = 1):
        """
        Initialize a FitnessData object.

        :param file_path: path to input file (required).
        :param name: name of fitness dataset (optional).
        :param order_subset: list of mutation orders corresponding to retained variants (optional).
        :param downsample_observations: number (if integer) or proportion (if float) of observations to retain including WT (optional).
        :param seed: Random seed for downsampling observations (default:1).
        :returns: FitnessData object.
        """
        #Initialize attributes
        self.vtable = pd.DataFrame()
        self.sequenceType = None #"nucleotide" or "aminoacid"
        self.variantCol = None #"nt_seq" or "aa_seq"
        self.mutationOrderCol = None #"Nham_nt" or "Nham_aa"
        self.wildtype = None #string
        self.wildtype_split = None #list of characters

        #Read the indicated file
        self.read_fitness(
            file_path = file_path, 
            name = name, 
            order_subset = order_subset,
            downsample_observations = downsample_observations,
            seed = seed)

    def read_fitness_r(
        self, 
        file_path):
        """
        Read fitness from DiMSum RData file.

        :param file_path: path to RData file (required).
        :returns: pandas DataFrame.
        """
        vtable = pd.DataFrame()
        #Read file
        try:
            vtable = pyreadr.read_r(file_path)
        except:
            print("Error: Invalid RData fitness file: cannot read file.")
            raise ValueError
        #Check variant fitness object exists
        if 'all_variants' not in vtable.keys():
            print("Error: Invalid RData fitness file: variant data not found.")
            raise ValueError
        return vtable['all_variants']

    def read_fitness_txt(
        self, 
        file_path):
        """
        Read fitness from plain text file.

        :param file_path: path to plain text file (required).
        :returns: pandas DataFrame.
        """
        vtable = pd.DataFrame()
        #Read file
        try:
            vtable = pd.read_csv(file_path, sep = None, engine='python', na_values = [''], keep_default_na = False)
        except:
            print("Error: Invalid plain text fitness file: cannot read file.")
            raise ValueError
        return vtable

    def read_fitness(
        self, 
        file_path, 
        name = None, 
        order_subset = None,
        downsample_observations = None,
        seed = 1):
        """
        Read fitness from DiMSum file.

        :param file_path: path to file (required).
        :param name: name of fitness dataset (optional).
        :param order_subset: list of mutation orders corresponding to retained variants (optional).
        :param downsample_observations: number (if integer) or proportion (if float) of observations to retain including WT (optional).
        :param seed: Random seed for downsampling observations (default:1).
        :returns: Nothing.
        """

        #Check file exists
        if not exists(file_path):
            print("Error: Fitness file not found.")
            raise ValueError

        #Automatically detect file type
        file_type = "text"
        try:
            pd.read_csv(file_path, nrows = 1, sep = None, engine='python')
        except:
            file_type = "RData"

        #Convert downsample_observations to integer if round number
        if downsample_observations!=None:
            if downsample_observations%1 == 0:
                downsample_observations = int(downsample_observations)

        #Read file
        if file_type == "RData":
            self.vtable = self.read_fitness_r(file_path)
        if file_type == "text":
            self.vtable = self.read_fitness_txt(file_path)

        #Check if data exists
        if self.vtable.size == 0:
            return

        #Nucleotide or peptide sequence?
        self.sequenceType = "nucleotide"
        self.variantCol = "nt_seq"
        self.mutationOrderCol = "Nham_nt"
        if 'aa_seq' in self.vtable.columns:
            self.sequenceType = "aminoacid"
            self.variantCol = "aa_seq"
            self.mutationOrderCol = "Nham_aa"

        #Check required columns exist
        if sum([i not in self.vtable.columns for i in [
            self.variantCol, 
            self.mutationOrderCol, 
            'WT', 
            'fitness',
            'sigma']])!=0:
            print("Error: Invalid fitness data: required columns not found.")
            raise ValueError

        #Remove variants with STOP or STOP_readthrough (if columns present)
        if self.sequenceType == "aminoacid":
            if 'STOP' in self.vtable.columns:
                self.vtable = self.vtable[self.vtable['STOP']==False]
            if 'STOP_readthrough' in self.vtable.columns:
                self.vtable = self.vtable[self.vtable['STOP_readthrough']==False]

        #Remove variants of undesired mutation order
        if order_subset!=None and isinstance(order_subset, list):
            self.vtable = self.vtable.iloc[[i in order_subset for i in self.vtable[self.mutationOrderCol]],:]

        #Check single WT variant present
        if sum(self.vtable['WT']==True)!=1:
            print("Error: Invalid fitness data: WT variant missing or ambiguous.")
            raise ValueError

        #Remove synonymous variants (if present)
        self.vtable = self.vtable[(self.vtable['WT']==True) | (self.vtable[self.mutationOrderCol]>0)]

        #Wild-type sequence
        self.wildtype = str(np.array(self.vtable.loc[self.vtable['WT'] == True, self.variantCol])[0])
        self.wildtype_split = [c for c in self.wildtype]

        #Name
        if name!=None:
            self.vtable['phenotype'] = name

        #Randomly downsample observations
        if downsample_observations!=None:
            if type(downsample_observations) == float:
                #Downsample observations by proportion
                if downsample_observations < 1 and downsample_observations > 0:
                    vtable_wt = copy.deepcopy(self.vtable.loc[self.vtable['WT']==True,:])
                    self.vtable = pd.concat([
                        vtable_wt,
                        self.vtable.loc[self.vtable['WT']!=True,:].sample(frac = downsample_observations, random_state = seed)])
                    self.vtable.reset_index(drop = True, inplace = True)
                else:
                    print("Error: downsample_observations argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                    raise ValueError
            elif type(downsample_observations) == int:
                #Downsample observations by number
                if downsample_observations >= 1:
                    vtable_wt = copy.deepcopy(self.vtable.loc[self.vtable['WT']==True,:])
                    self.vtable = pd.concat([
                        vtable_wt,
                        self.vtable.loc[self.vtable['WT']!=True,:].sample(n = downsample_observations, random_state = seed)])
                    self.vtable.reset_index(drop = True, inplace = True)
                else:
                    print("Error: downsample_observations argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                    raise ValueError

    def __len__(self):
        """
        Length of object.

        :returns: Number of variants (length of vtable).
        """
        return len(self.vtable)

class MochiData:
    """
    A class for the storage of genotype-phenotype data from a collection of DMS experiments.
    An object of this class is required for model inference with MoCHI.
    """
    def __init__(
        self, 
        # directory,
        model_design,
        order_subset = None,
        downsample_observations = None,
        downsample_interactions = None,
        max_interaction_order = 1,
        min_observed = 2,
        k_folds = 10,
        seed = 1,
        validation_factor = 2, 
        holdout_minobs = 0, 
        holdout_orders = [], 
        holdout_WT = False,
        features = {},
        ensemble = False,
        custom_transformations = None):
        """
        Initialize a MochiData object.

        # :param directory: Path to directory where data should be saved/loaded (required).
        :param model_design: Model design DataFrame with phenotype, transformation, trait and file columns (required).
        :param order_subset: list of mutation orders corresponding to retained variants (optional).
        :param downsample_observations: number (if integer) or proportion (if float) of observations to retain including WT (optional).
        :param downsample_interactions: number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (optional).
        :param max_interaction_order: Maximum interaction order (default:1).
        :param min_observed: Minimum number of observations required to include interaction term (default:2).
        :param k_folds: Numbef of cross-validation folds (default:10).
        :param seed: Random seed for downsampling (observations and interactions) and defining cross-validation groups (default:1).
        :param validation_factor: Relative size of validation set with respect to test set (default:2).
        :param holdout_minobs: Minimum number of observations of additive trait weights to be held out (default:0).
        :param holdout_orders: list of mutation orders corresponding to retained variants (default:[] i.e. variants of all mutation orders can be held out).
        :param holdout_WT: Whether or not to WT variant can be held out (default:False).
        :param features: dictionary of trait-specific feature names to fit (default:{} i.e. all features fit).
        :param ensemble: Ensemble encode features. (default:False).
        :param custom_transformations: Path to custom transformations file (optional).
        :returns: MochiData object.
        """
        #Save attributes
        # self.directory = directory
        self.model_design = model_design
        self.order_subset = order_subset
        self.downsample_observations = downsample_observations
        self.downsample_interactions = downsample_interactions
        self.max_interaction_order = max_interaction_order
        self.min_observed = min_observed
        self.k_folds = k_folds
        self.seed = seed
        self.validation_factor = validation_factor
        self.holdout_minobs = holdout_minobs
        self.holdout_orders = holdout_orders
        self.holdout_WT = holdout_WT
        self.features = features
        self.features_trait = {}
        self.ensemble = ensemble
        self.custom_transformations = custom_transformations
        #Initialize attributes
        self.fdata = None
        self.additive_trait_names = None
        self.phenotype_names = None
        self.fitness = None
        self.phenotypes = None
        self.X = None
        self.Xoh = None
        self.Xohi = None
        self.Xohi_memmap = None
        self.Xohi_memmap_path = None
        self.feature_matrix_mode = "dense"
        self.feature_sparse_matrix = None
        self.feature_source_indices = None
        self.feature_component_indices = None
        self.cvgroups = None
        self.coefficients = None
        self.coefficients_userspec = None
        self.custom_transformations_code = None
        self.feature_names = None

        # #Create data directory
        # try:
        #     os.makedirs(self.directory)
        # except FileExistsError:
        #     print("Error: Data directory already exists.")
        #     raise ValueError

        #Check and save custom transformations
        self.custom_transformations = self.check_custom_transformations(self.custom_transformations) 
        #Check and save model design
        self.model_design = self.check_model_design(self.model_design)
        #Check features
        self.features, self.features_trait = self.check_features(self.features)
        #Load all datasets
        print("Loading fitness data")
        filepath_list = list(self.model_design['file'])
        fdatalist = [FitnessData(
            file_path = filepath_list[i], 
            name = str(i+1),
            order_subset = self.order_subset,
            downsample_observations = self.downsample_observations,
            seed = self.seed) for i in range(len(filepath_list))]
        #Merge all datasets
        self.fdata = self.merge_datasets(fdatalist)
        #Fitness
        self.fitness = self.fdata.vtable.loc[:,['fitness', 'sigma']]
        #Phenotypes
        self.phenotypes = self.one_hot_encode_phenotypes()
        #Fitness weights (1/sigma)
        for i in range(len(self.model_design)):
            self.fitness.loc[self.phenotypes['phenotype_'+str(self.model_design.loc[i,'phenotype'])]==1,'weight'] = self.model_design.loc[i,'weight']
        self.fitness['weight'] = 1/np.power(np.asarray(self.fitness['sigma']), 1) * self.fitness['weight']
        #Sequence features
        self.X = self.fdata.vtable[self.fdata.variantCol].str.split('', expand=True).iloc[:, 1:-1]
        #One hot encode sequence features
        print("One-hot encoding sequence features")
        self.Xoh = self.one_hot_encode_features()
        #One-hot encode interaction features
        print("One-hot encoding interaction features")
        self.one_hot_encode_interactions(
            max_order = self.max_interaction_order,
            min_observed = self.min_observed,
            features = self.features,
            downsample_interactions = self.downsample_interactions,
            seed = self.seed)
        log_process_memory("After interaction features")
        # Release split sequence tokens before later wide-matrix passes.
        self.X = None
        gc.collect()
        log_process_memory("After releasing sequence token matrix")
        # self.one_hot_encode_interactions_todisk(
        #     max_order = self.max_interaction_order,
        #     min_observed = self.min_observed,
        #     features = self.features,
        #     downsample_interactions = self.downsample_interactions,
        #     seed = self.seed,
        #     holdout_minobs = self.holdout_minobs, 
        #     holdout_orders = self.holdout_orders, 
        #     holdout_WT = self.holdout_WT)
        #Split into training, validation and test sets
        print("Defining cross-validation groups")
        self.define_cross_validation_groups()
        log_process_memory("After defining cross-validation groups")
        #Define coefficients to fit (for each phenotype and trait)
        print("Defining coefficient groups")
        self.define_coefficient_groups(
            k_folds = self.k_folds)
        log_process_memory("After defining coefficient groups")
        #Ensemble encode features
        if self.ensemble:
            print("Ensemble encoding features")
            self.Xohi = self.ensemble_encode_features()
            log_process_memory("After ensemble encoding features")
        print("Done!")

    def check_custom_transformations(
        self, 
        input_obj):
        """
        Check custom transformations valid and reformat.

        :param input_obj: Path to custom transformations file or None (required).
        :returns: A reformatted custom transformations dictionary.
        """
        #No argument supplied
        if input_obj is None:
            return {}
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            print("Error: custom_transformations argument invalid.")
            raise ValueError
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            print("Error: Custom transformations file not found.")
            raise ValueError
        else:
            #Read input file
            with open(input_obj, "rb") as source_file:
                self.custom_transformations_code = source_file.read()
            return CustomTransformations(self.custom_transformations_code).transformations

    def restore_custom_transformations(
        self):
        """
        Restore custom transformations and check valid and reformat.

        :returns: A reformatted custom transformations dictionary.
        """
        #For backwards compatibility supply all previous custom transformations code
        if "custom_transformations_code" not in dir(self):
            input_obj = str(Path(__file__).parent / "data/custom_transformations.py")
            with open(input_obj, "rb") as source_file:
                self.custom_transformations_code = source_file.read()
        #Restore custom transformation from code
        if self.custom_transformations_code is None:
            self.custom_transformations = {}
        else:
            self.custom_transformations = CustomTransformations(self.custom_transformations_code).transformations

    def check_model_design(
        self, 
        model_design):
        """
        Check model design valid and reformat.

        :param model_design: Model design DataFrame with phenotype, transformation, trait and file columns (required).
        :returns: A reformatted model design DataFrame.
        """
        #Model design keys
        model_design_keys = [
            'phenotype',
            'transformation',
            'trait',
            'file',
            'weight']
        #Check if model_design is a pandas DataFrame
        if not isinstance(model_design, pd.DataFrame):
            print("Error: Model design is not a pandas DataFrame.")
            raise ValueError
        #Add unity weights if not supplied
        if 'weight' not in model_design.keys():
            model_design['weight'] = 1
        #Check if all keys present
        if sum([i not in model_design.keys() for i in model_design_keys])!=0:
            print("Error: Model design missing required keys.")
            raise ValueError
        #Split trait column items into list if necessary
        model_design['trait'] = [i.split(',') if type(i)==str else i for i in model_design['trait']]
        #Unique traits
        all_traits = [item for sublist in list(model_design['trait']) for item in sublist]
        all_traits_unique = []
        [all_traits_unique.append(i) for i in all_traits if i not in all_traits_unique]
        all_traits_unique_dict = dict(zip(all_traits_unique, range(1, len(all_traits_unique)+1)))
        #Translate traits to integers
        self.additive_trait_names = all_traits_unique
        model_design['trait'] = [[all_traits_unique_dict[j] for j in i] for i in model_design['trait']]
        #Unique phenotypes
        all_phenotypes = list(model_design['phenotype'])
        all_phenotypes_unique = []
        [all_phenotypes_unique.append(i) for i in all_phenotypes if i not in all_phenotypes_unique]
        if len(all_phenotypes_unique)!=len(all_phenotypes):
            print("Error: Duplicated phenotype names.")
            raise ValueError
        #Translate phenotypes to integers
        self.phenotype_names = all_phenotypes_unique
        model_design['phenotype'] = range(1, len(model_design)+1)
        #Check phenotype names
        forbidden_pnames = ['nt_seq', 'aa_seq', 'Nham_nt', 'Nham_aa', 'WT', 'fitness', 'sigma', 'phenotype', 'mean', 'std', 'ci95', 'Fold']
        if sum([(i in forbidden_pnames) or (i.startswith('fold_')) for i in self.phenotype_names])!=0:
            print("Error: Forbidden phenotype names.")
            raise ValueError
        #Check files not duplicated
        all_files = list(model_design['file'])
        if len(all_files)!=len(list(set(all_files))):
            print("Error: Duplicated fitness files.")
            raise ValueError
        return model_design

    def check_features(
        self, 
        features):
        """
        Check features and reformat.

        :param features: features dictionary (required).
        :returns: A tuple of reformatted features (list) and features_trait (dictionary).
        """
        #Check dictionary
        if type(features) != dict:
            print("Error: 'features' argument is not a dictionary.")
            raise ValueError
        #Filter applied to all traits (original input = list or single column of feature identifiers without header)
        if len(features)==1:
            key1 = list(features.keys())[0]
            if key1 is None:
                #Filter applied to all traits
                features = features[key1]
                return (features, {})
            elif key1 not in self.additive_trait_names:
                #Filter applied to all traits
                features = [key1]+features[key1]
                return (features, {})

        #Check all dictionary keys are trait names
        if sum([1 for i in features.keys() if i in self.additive_trait_names])!=len(features):
            print("Error: One or more invalid trait names in 'features' argument.")
            raise ValueError

        #Check all dictionary values include WT
        if sum([1 for i in features.keys() if 'WT' in features[i]])!=len(features):
            print("Error: 'WT' missing for one or more traits in 'features' argument.")
            raise ValueError        

        #Copy features dictionary
        features_trait = copy.deepcopy(features)
        #List of unique features
        features = [item for sublist in list(features.values()) for item in sublist if type(item)==str]
        return (features, features_trait)

    def merge_datasets(
        self,
        data_list):
        """
        Merge variant tables from a list of FitnessData objects with same WT and sequence type.

        :param data_list: list of FitnessData objects to merge (required).
        :returns: A merged FitnessData object.
        """
        #Check if data to merge
        if sum([len(i) for i in data_list]) == 0:
            print("Error: No Fitness datasets to merge.")
            raise ValueError
        #Check if same WT and sequence type
        if len(set([i.wildtype for i in data_list]))==1 & len(set([i.sequenceType for i in data_list]))==1:
            fdata = copy.deepcopy(data_list[0])
            fdata.vtable = pd.concat([i.vtable for i in data_list])
            fdata.vtable.reset_index(drop = True, inplace = True)
            return fdata
        else:
            print("Error: Fitness datasets cannot be merged: WT variants do not match.")
            raise ValueError

    def one_hot_encode_phenotypes(self):
        """
        1-hot encode phenotypes.

        :returns: A DataFrame with 1-hot phenotypes.
        """
        all_phenotypes = [str(i) for i in list(self.model_design['phenotype'])]
        phenotypes_df = pd.DataFrame()
        for i in all_phenotypes:
            phenotypes_df['phenotype_'+i] = (self.fdata.vtable['phenotype']==i).astype(np.uint8)
        return phenotypes_df

    def one_hot_encode_features(
        self,
        include_WT = True):
        """
        1-hot encode sequences.

        :param include_WT: Whether or not to include WT feature (default:True).
        :returns: A DataFrame with 1-hot sequences.
        """
        enc = OneHotEncoder(
            handle_unknown='ignore', 
            drop = np.array(self.fdata.wildtype_split), 
            dtype = np.uint8)
        enc.fit(self.X)
        one_hot_names = [self.fdata.wildtype_split[int(i[1:-2])]+str(int(i[1:-2])+1)+i[-1] for i in enc.get_feature_names_out()]
        one_hot_df = pd.DataFrame(enc.transform(self.X).toarray(), columns = one_hot_names)
        if include_WT:
            one_hot_df = pd.concat([pd.DataFrame({'WT': np.ones(len(one_hot_df), dtype = np.uint8)}), one_hot_df], axis=1)
        one_hot_df = one_hot_df.astype(np.uint8)
        return one_hot_df

    def get_theoretical_interactions_phenotype(
        self, 
        phenotype = 1,
        max_order = 2):
        """
        Get theoretical interaction features for variants corresponding to a specific phenotype.

        :param phenotype: Phenotype number (default:1).
        :param max_order: Maximum interaction order (default:2).
        :returns: dictionary of interaction features.
        """
        #Mutations observed for this phenotype
        mut_count = list(self.Xoh.loc[self.phenotypes['phenotype_'+str(phenotype)]==1,:].sum(axis = 0))
        pheno_mut = [self.Xoh.columns[i] for i in range(len(self.Xoh.columns)) if mut_count[i]!=0]
        #All possible combinations of mutations
        all_pos = list(set([i[1:-1] for i in pheno_mut if i!="WT"]))
        all_pos_mut = {int(i):[j for j in pheno_mut if j[1:-1]==i] for i in all_pos}
        all_features = {}
        for n in range(max_order + 1):
            #Order at least 2
            if n>1:
                all_features[n] = []
                #All position combinations
                pos_comb = list(itertools.combinations(sorted(all_pos_mut.keys()), n))
                for p in pos_comb:
                    #All mutation combinations for these positions
                    all_features[n] += ["_".join(c) for c in itertools.product(*[all_pos_mut[j] for j in p])]
        return all_features

    def merge_dictionaries(
        self,
        dict1,
        dict2):
        """
        Merge two dictionaries by keys (and unique values).

        :param dict1: Dictionary 1.
        :param dict2: Dictionary 2.
        :returns: merged dictionary.
        """
        for k in dict2:
            if k in dict1:
                dict1[k] = sorted(set(dict1[k] + dict2[k]))
            else:
                dict1[k] = sorted(dict2[k])
        return dict1

    def get_theoretical_interactions(
        self, 
        max_order = 2):
        """
        Get theoretical interaction features.

        :param max_order: Maximum interaction order (default:2).
        :returns: tuple with dictionary of interaction features and dictionary of order counts.
        """
        #All features
        all_features = {}
        for i in list(self.model_design.phenotype):
            all_features = self.merge_dictionaries(
                dict1 = all_features,
                dict2 = self.get_theoretical_interactions_phenotype(phenotype = i, max_order = max_order))
        #All feature orders
        int_order_dict = {k:len(all_features[k]) for k in all_features}
        return (all_features, int_order_dict)

    def get_xohi_cache_paths(
        self,
        max_order,
        min_observed,
        features,
        downsample_interactions,
        seed):
        """
        Get cache paths for interaction features if caching is enabled.

        Set MOCHI_XOHI_CACHE_DIR to enable the cache.

        :returns: Tuple of metadata path and memmap path, or (None, None).
        """
        cache_dir = os.environ.get("MOCHI_XOHI_CACHE_DIR")
        if cache_dir in [None, ""]:
            return (None, None)
        os.makedirs(cache_dir, exist_ok = True)
        file_stats = [
            {
                'path': str(pathlib.Path(i).resolve()),
                'size': os.path.getsize(i),
                'mtime_ns': os.stat(i).st_mtime_ns}
            for i in list(self.model_design['file'])]
        cache_input = {
            'files': file_stats,
            'max_order': max_order,
            'min_observed': min_observed,
            'features': features,
            'downsample_interactions': downsample_interactions,
            'seed': seed,
            'order_subset': self.order_subset,
            'downsample_observations': self.downsample_observations,
            'xoh_columns': list(self.Xoh.columns),
            'rows': len(self.Xoh)}
        cache_key = hashlib.sha256(
            json.dumps(cache_input, sort_keys = True, default = str).encode('utf-8')).hexdigest()
        metadata_path = os.path.join(cache_dir, f"{cache_key}.json")
        memmap_path = os.path.join(cache_dir, f"{cache_key}.dat")
        return (metadata_path, memmap_path)

    def load_cached_interactions(
        self,
        metadata_path,
        memmap_path):
        """
        Load cached interaction features if present and valid.

        :returns: True if cache hit, False otherwise.
        """
        if metadata_path is None:
            return False
        if not os.path.exists(metadata_path):
            return False
        try:
            with open(metadata_path, "r") as handle:
                metadata = json.load(handle)
        except Exception:
            return False
        if metadata.get('rows') != len(self.Xoh):
            return False
        if metadata.get('base_columns') != list(self.Xoh.columns):
            return False
        print("Loading cached interaction features")
        columns = metadata.get('columns', list(self.Xoh.columns))
        interaction_columns = [i for i in columns if i not in self.Xoh.columns]
        if len(interaction_columns) == 0:
            self.Xohi = self.Xoh.loc[:, columns].copy()
            self.feature_matrix_mode = "dense"
            self.feature_sparse_matrix = None
            self.feature_source_indices = None
            self.feature_component_indices = None
            self.Xohi_memmap = None
            self.Xohi_memmap_path = None
            self.feature_names = self.Xohi.columns
        else:
            if get_feature_store_backend() == "sparse":
                self.build_sparse_feature_matrix_from_columns(columns)
            else:
                self.set_lazy_feature_matrix(columns)
        return True

    def save_cached_interactions(
        self,
        metadata_path,
        columns):
        """
        Save interaction-feature cache metadata.

        :returns: Nothing.
        """
        if metadata_path is None:
            return
        metadata = {
            'rows': len(self.Xoh),
            'base_columns': list(self.Xoh.columns),
            'columns': list(columns)}
        if self.Xohi_memmap is not None:
            metadata['shape'] = list(self.Xohi_memmap.shape)
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle)

    def get_feature_names(
        self):
        """
        Return feature names regardless of whether features are dense or lazy.

        :returns: pandas Index.
        """
        if self.feature_names is not None:
            return pd.Index(self.feature_names)
        if self.Xohi is not None and hasattr(self.Xohi, "columns"):
            return pd.Index(self.Xohi.columns)
        return pd.Index([])

    def is_lazy_feature_matrix(
        self):
        """
        Check whether the feature matrix is materialized on demand.

        :returns: Boolean.
        """
        return getattr(self, "feature_matrix_mode", "dense") == "lazy"

    def is_sparse_feature_matrix(
        self):
        """
        Check whether the feature matrix is stored as a sparse matrix.

        :returns: Boolean.
        """
        return getattr(self, "feature_matrix_mode", "dense") == "sparse"

    def activate_sparse_feature_matrix(
        self,
        columns,
        sparse_matrix):
        """
        Activate a sparse retained-feature matrix backend.

        :param columns: Ordered feature names (required).
        :param sparse_matrix: Sparse feature matrix (required).
        :returns: Nothing.
        """
        columns = pd.Index(columns)
        self.feature_matrix_mode = "sparse"
        self.feature_names = columns
        self.feature_sparse_matrix = sparse_matrix.tocsr().astype(np.uint8, copy = False)
        self.feature_source_indices = None
        self.feature_component_indices = None
        self.Xohi_memmap = None
        self.Xohi_memmap_path = None
        self.Xohi = FeatureMatrixMetadata(
            index = self.Xoh.index,
            columns = columns)

    def build_sparse_feature_matrix(
        self,
        columns,
        interaction_names,
        interaction_row_indices):
        """
        Build a sparse retained-feature matrix from base one-hot columns and
        retained interaction row indices.

        :param columns: Ordered feature names to expose (required).
        :param interaction_names: Canonical interaction names (required).
        :param interaction_row_indices: Active-row arrays for interactions (required).
        :returns: Nothing.
        """
        base_sparse = sp.csr_matrix(self.Xoh.to_numpy(dtype = np.uint8, copy = False))
        if len(interaction_names) == 0:
            combined = base_sparse
        else:
            nnz_per_col = np.asarray([len(i) for i in interaction_row_indices], dtype = np.int64)
            if int(np.sum(nnz_per_col)) == 0:
                interaction_sparse = sp.csr_matrix(
                    (len(self.Xoh), len(interaction_names)),
                    dtype = np.uint8)
            else:
                row_ind = np.concatenate(interaction_row_indices).astype(np.int32, copy = False)
                col_ind = np.repeat(
                    np.arange(len(interaction_names), dtype = np.int32),
                    nnz_per_col)
                data = np.ones(len(row_ind), dtype = np.uint8)
                interaction_sparse = sp.csr_matrix(
                    (data, (row_ind, col_ind)),
                    shape = (len(self.Xoh), len(interaction_names)),
                    dtype = np.uint8)
            combined = sp.hstack([base_sparse, interaction_sparse], format = "csr", dtype = np.uint8)
        self.activate_sparse_feature_matrix(
            columns = list(self.Xoh.columns) + list(interaction_names),
            sparse_matrix = combined)
        if list(self.get_feature_names()) != list(columns):
            self.reorder_feature_columns(columns)

    def build_sparse_feature_matrix_from_columns(
        self,
        columns):
        """
        Build a sparse retained-feature matrix by recomputing only the retained
        interaction columns named in `columns`.

        :param columns: Ordered feature names (required).
        :returns: Nothing.
        """
        xoh_column_index = {name:i for i, name in enumerate(self.Xoh.columns)}
        xoh_values = self.Xoh.to_numpy(dtype = np.uint8, copy = False)
        interaction_names = [i for i in columns if i not in xoh_column_index]
        interaction_row_indices = []
        for name in interaction_names:
            feature_idx = [xoh_column_index[i] for i in name.split("_")]
            interaction_row_indices.append(
                np.flatnonzero(
                    np.all(xoh_values[:, feature_idx] == 1, axis = 1)
                ).astype(np.int32, copy = False))
        self.build_sparse_feature_matrix(
            columns = columns,
            interaction_names = interaction_names,
            interaction_row_indices = interaction_row_indices)

    def set_lazy_feature_matrix(
        self,
        columns):
        """
        Configure retained feature metadata without materializing a dense matrix.

        :param columns: Ordered feature names (required).
        :returns: Nothing.
        """
        xoh_column_index = {name:i for i,name in enumerate(self.Xoh.columns)}
        columns = pd.Index(columns)
        source_indices = np.full(len(columns), -1, dtype = np.int32)
        component_indices = [None] * len(columns)
        for i, name in enumerate(columns):
            if name in xoh_column_index:
                source_indices[i] = xoh_column_index[name]
            else:
                component_indices[i] = np.asarray(
                    [xoh_column_index[j] for j in name.split("_")],
                    dtype = np.int32)
        self.feature_matrix_mode = "lazy"
        self.feature_names = columns
        self.feature_sparse_matrix = None
        self.feature_source_indices = source_indices
        self.feature_component_indices = component_indices
        self.Xohi_memmap = None
        self.Xohi_memmap_path = None
        self.Xohi = FeatureMatrixMetadata(
            index = self.Xoh.index,
            columns = columns)

    def reorder_feature_columns(
        self,
        columns):
        """
        Reorder retained feature columns while preserving the active storage mode.

        :param columns: Ordered feature names (required).
        :returns: Nothing.
        """
        columns = list(columns)
        missing = [i for i in columns if i not in self.get_feature_names()]
        if len(missing) != 0:
            print("Error: Invalid feature names.")
            raise ValueError
        if self.is_lazy_feature_matrix():
            self.set_lazy_feature_matrix(columns)
        elif self.is_sparse_feature_matrix():
            feature_index = {name:i for i, name in enumerate(self.get_feature_names())}
            self.activate_sparse_feature_matrix(
                columns = columns,
                sparse_matrix = self.feature_sparse_matrix[:, [feature_index[i] for i in columns]])
        else:
            self.Xohi = self.Xohi.loc[:, columns]
            self.feature_names = self.Xohi.columns

    def get_feature_chunk_size(
        self):
        """
        Return feature chunk size used for on-demand interaction assembly.

        :returns: Integer chunk size.
        """
        return int(os.environ.get("MOCHI_FEATURE_CHUNK_SIZE", "256"))

    def check_materialization_memory(
        self,
        n_rows,
        n_features,
        dtype):
        """
        Fail fast when an on-demand dense materialization exceeds a configured cap.

        Set MOCHI_MAX_XOHI_GB to a positive number to enable the guard.

        :returns: Nothing.
        """
        max_gb = os.environ.get("MOCHI_MAX_XOHI_GB")
        if max_gb in [None, ""]:
            return
        projected_bytes = int(n_rows) * int(n_features) * np.dtype(dtype).itemsize
        if projected_bytes > (float(max_gb) * (1024 ** 3)):
            print("Error: On-demand feature matrix exceeds MOCHI_MAX_XOHI_GB.")
            raise MemoryError

    def iterate_feature_chunks(
        self,
        row_indices,
        feature_indices = None,
        dtype = np.uint8,
        chunk_size = None):
        """
        Yield dense feature chunks for a selected row subset.

        :param row_indices: Row indices to materialize (required).
        :param feature_indices: Feature indices to materialize (default:None i.e. all).
        :param dtype: Output dtype (default:uint8).
        :param chunk_size: Feature chunk size (default:env or 256).
        :returns: iterator of (start, stop, ndarray).
        """
        if chunk_size is None:
            chunk_size = self.get_feature_chunk_size()
        row_indices = np.asarray(row_indices, dtype = np.int64)
        if feature_indices is None:
            feature_indices = np.arange(len(self.get_feature_names()), dtype = np.int64)
        else:
            feature_indices = np.asarray(feature_indices, dtype = np.int64)
        if len(feature_indices) == 0:
            return
        if self.is_sparse_feature_matrix():
            sparse_rows = self.feature_sparse_matrix[row_indices]
            for start in range(0, len(feature_indices), chunk_size):
                stop = min(start + chunk_size, len(feature_indices))
                chunk_feature_indices = feature_indices[start:stop]
                chunk = sparse_rows[:, chunk_feature_indices].toarray()
                if dtype != chunk.dtype:
                    chunk = chunk.astype(dtype, copy = False)
                yield (start, stop, chunk)
            return
        if not self.is_lazy_feature_matrix():
            feature_values = self.Xohi.iloc[row_indices, feature_indices].to_numpy(
                dtype = dtype,
                copy = True)
            yield (0, len(feature_indices), feature_values)
            return

        xoh_values = self.Xoh.iloc[row_indices,:].to_numpy(dtype = np.uint8, copy = False)
        feature_source_indices = self.feature_source_indices
        feature_component_indices = self.feature_component_indices
        for start in range(0, len(feature_indices), chunk_size):
            stop = min(start + chunk_size, len(feature_indices))
            chunk_feature_indices = feature_indices[start:stop]
            chunk = np.empty((len(row_indices), len(chunk_feature_indices)), dtype = np.uint8)
            chunk_sources = feature_source_indices[chunk_feature_indices]
            base_local_idx = np.flatnonzero(chunk_sources >= 0)
            if len(base_local_idx) != 0:
                chunk[:, base_local_idx] = xoh_values[:, chunk_sources[base_local_idx]]
            interaction_local_idx = np.flatnonzero(chunk_sources < 0)
            for local_i in interaction_local_idx:
                feature_i = chunk_feature_indices[local_i]
                chunk[:, local_i] = np.all(
                    xoh_values[:, feature_component_indices[feature_i]] == 1,
                    axis = 1).astype(np.uint8, copy = False)
            if dtype != np.uint8:
                chunk = chunk.astype(dtype, copy = False)
            yield (start, stop, chunk)

    def materialize_feature_matrix(
        self,
        row_indices,
        feature_indices = None,
        dtype = np.uint8):
        """
        Materialize a dense feature matrix for a selected row subset.

        :param row_indices: Row indices to materialize (required).
        :param feature_indices: Feature indices to materialize (default:None i.e. all).
        :param dtype: Output dtype (default:uint8).
        :returns: ndarray.
        """
        row_indices = np.asarray(row_indices, dtype = np.int64)
        if feature_indices is None:
            feature_indices = np.arange(len(self.get_feature_names()), dtype = np.int64)
        else:
            feature_indices = np.asarray(feature_indices, dtype = np.int64)
        self.check_materialization_memory(
            n_rows = len(row_indices),
            n_features = len(feature_indices),
            dtype = dtype)
        if len(feature_indices) == 0:
            return np.empty((len(row_indices), 0), dtype = dtype)
        if self.is_sparse_feature_matrix():
            matrix = self.feature_sparse_matrix[row_indices][:, feature_indices].toarray()
            if matrix.dtype != dtype:
                matrix = matrix.astype(dtype, copy = False)
            return matrix
        matrix = np.empty((len(row_indices), len(feature_indices)), dtype = dtype)
        for start, stop, chunk in self.iterate_feature_chunks(
            row_indices = row_indices,
            feature_indices = feature_indices,
            dtype = dtype):
            matrix[:, start:stop] = chunk
        return matrix

    def sum_features_for_rows(
        self,
        row_indices):
        """
        Sum feature columns across a selected row subset.

        :param row_indices: Row indices to aggregate (required).
        :returns: ndarray of per-feature sums.
        """
        if self.is_sparse_feature_matrix():
            return np.asarray(
                self.feature_sparse_matrix[np.asarray(row_indices, dtype = np.int64)].sum(axis = 0),
                dtype = np.int64).ravel()
        feature_sums = np.zeros(len(self.get_feature_names()), dtype = np.int64)
        for start, stop, chunk in self.iterate_feature_chunks(row_indices = row_indices):
            feature_sums[start:stop] = np.sum(chunk, axis = 0, dtype = np.int64)
        return feature_sums

    def sum_selected_features_per_row(
        self,
        row_indices,
        feature_indices):
        """
        Sum a selected feature subset across each requested row.

        :param row_indices: Row indices to aggregate (required).
        :param feature_indices: Feature indices to include (required).
        :returns: ndarray of per-row sums.
        """
        row_sums = np.zeros(len(row_indices), dtype = np.int64)
        feature_indices = np.asarray(feature_indices, dtype = np.int64)
        if len(feature_indices) == 0:
            return row_sums
        if self.is_sparse_feature_matrix():
            return np.asarray(
                self.feature_sparse_matrix[np.asarray(row_indices, dtype = np.int64)][:, feature_indices].sum(axis = 1),
                dtype = np.int64).ravel()
        for _, _, chunk in self.iterate_feature_chunks(
            row_indices = row_indices,
            feature_indices = feature_indices):
            row_sums += np.sum(chunk, axis = 1, dtype = np.int64)
        return row_sums

    # def write_features(
    #     self,
    #     feature_list, 
    #     feature_chunk_size,
    #     samples_chunk_size = 100,
    #     initial_chunk = False,
    #     final_chunk = False):
    #     """
    #     Write features to disk.

    #     :param feature_list: List of features.
    #     :param feature_chunk_size: Features chunk size in number of features.
    #     :param samples_chunk_size: Samples chunk size in number of samples (default:100).
    #     :param initial_chunk: Whether or not the supplied list is the initial chunk (default:False).
    #     :param final_chunk: Whether or not the supplied list is the final chunk (default:False).
    #     :returns: feature list.
    #     """
    #     #Check if anything to write
    #     if len(feature_list)==feature_chunk_size or final_chunk or initial_chunk:
    #         write_df = pd.concat(feature_list, axis = 1)
    #         self.update_holdout_observations(write_df)
    #         write_df = write_df.transpose()
    #         write_count = 0
    #         #Write feature data in chunks of rows (transposed)
    #         while write_count < write_df.shape[1]:
    #             write_file = os.path.join(self.directory, 'data_chunk'+str(write_count)+".csv")
    #             write_df.iloc[:,list(range(write_count, min([write_count+samples_chunk_size, write_df.shape[1]])))].to_csv(
    #                 write_file, mode='a', index=False, header=False)
    #             write_count += samples_chunk_size
    #         #Reset list and index
    #         feature_list = []
    #     return feature_list

    # def transpose_features(
    #     self):
    #     """
    #     Transpose features on disk.

    #     :returns: nothing.
    #     """
    #     files = os.listdir(self.directory)
    #     for f in files:
    #         if f.startswith("data_chunk"):
    #             pd.read_csv(
    #                 os.path.join(self.directory, f), header=None).transpose().to_csv(
    #                 os.path.join(self.directory, f), index=False, header=False)

    # def one_hot_encode_interactions_todisk(
    #     self, 
    #     max_order = 2,
    #     max_cells = 1e9,
    #     min_observed = 2,
    #     features = [],
    #     downsample_interactions = None,
    #     seed = 1,
    #     chunk_size = 100,
    #     holdout_minobs = 0,
    #     holdout_orders = [],
    #     holdout_WT = False):
    #     """
    #     Add interaction terms to 1-hot encoding DataFrame.

    #     :param max_order: Maximum interaction order (default:2).
    #     :param max_cells: Maximum matrix cells permitted (default:1billion).
    #     :param min_observed: Minimum number of observations required to include interaction term (default:2).
    #     :param features: list of feature names to filter (default:[] i.e. all  features retained).
    #     :param downsample_interactions: number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (optional).
    #     :param seed: Random seed for downsampling interactions (default:1).
    #     :param chunk_size: Number of features per file (default:100).
    #     :param holdout_minobs: Minimum number of observations of additive trait weights to be held out (default:0).
    #     :param holdout_orders: list of mutation orders corresponding to retained variants (default:[] i.e. variants of all mutation orders can be held out).
    #     :param holdout_WT: list of mutation orders corresponding to retained variants (default:False).
    #     :returns: Nothing.
    #     """

    #     #First order interaction features
    #     #Filter features
    #     if features!=[]:
    #         print("Filtering features")
    #         self.Xoh = self.filter_features(
    #             input_df = self.Xoh,
    #             features = features)
    #     #Write to disk
    #     int_list = self.write_features(
    #         feature_list = [self.Xoh[i] for i in self.Xoh.columns], 
    #         feature_chunk_size = chunk_size,
    #         initial_chunk = True)
    #     #Save feature names
    #     self.feature_names = list(self.Xoh.columns)

    #     #Check if no interactions to add
    #     if max_order<2:
    #         #Transpose disk data in place
    #         print("Transposing features")
    #         self.transpose_features()
    #         return

    #     #Check downsample_interactions argument valid
    #     if downsample_interactions!=None:
    #         if type(downsample_interactions) == float:
    #             #Downsample observations by proportion
    #             if downsample_interactions >= 1 or downsample_interactions <= 0:
    #                 print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
    #                 raise ValueError
    #         elif type(downsample_interactions) == int:
    #             #Downsample observations by number
    #             if downsample_interactions < 1:
    #                 print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
    #                 raise ValueError
    #         elif type(downsample_interactions) == str:
    #             try:
    #                 downsample_interactions = {(i+2):int(d) for i,d in enumerate(str(downsample_interactions).split(","))}
    #             except:
    #                 print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
    #                 raise ValueError
    #         else:
    #             print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
    #             raise ValueError

    #     #Get all theoretical interactions
    #     all_features,int_order_dict = self.get_theoretical_interactions(max_order = max_order)
    #     print("... Total theoretical features (order:count): "+", ".join([str(i)+":"+str(int_order_dict[i]) for i in sorted(int_order_dict.keys())]))
    #     #Flatten
    #     all_features_flat = list(itertools.chain(*list(all_features.values())))

    #     #Check if all interaction features exist (i.e. with mutation order>1)
    #     if len([i for i in features if (i not in all_features_flat) and (len(i.split('_'))>1)]) != 0:
    #         print("Error: Invalid feature names.")
    #         raise ValueError

    #     #Select interactions
    #     int_list = []
    #     int_order_dict_retained = {}
    #     int_list_names = []
    #     #No shuffle if not downsampling
    #     if downsample_interactions is None:
    #         all_features_loop = {0: all_features_flat}
    #     #Shuffle flattened features
    #     elif type(downsample_interactions) in [float, int]:
    #         random.seed(seed)
    #         all_features_loop = {0: random.sample(all_features_flat, len(all_features_flat))}
    #     #Shuffle features separately per order
    #     else:
    #         all_features_loop = {k: random.sample(all_features[k], len(all_features[k])) for k in all_features}

    #     #Loop over all orders
    #     for n in all_features_loop.keys():
    #         #Loop over all features of this order
    #         for c in all_features_loop[n]:
    #             c_split = c.split("_")
    #             #Check if feature desired
    #             if (c in features) or features==[]:
    #                 int_col = (self.Xoh.loc[:,c_split].sum(axis = 1)==len(c_split)).astype(int)
    #                 #Check if minimum number of observations satisfied
    #                 if sum(int_col) >= min_observed:
    #                     int_list += [int_col]
    #                     int_list_names += [c]
    #                     if len(c_split) not in int_order_dict_retained.keys():
    #                         int_order_dict_retained[len(c_split)] = 1
    #                     else:
    #                         int_order_dict_retained[len(c_split)] += 1
    #                 # else:
    #                 #     if len(c_split)==3 and sum(int_col)==1:
    #                 #         print(c)
    #                 #Check memory footprint
    #                 if len(int_list_names)*len(self.Xoh) > max_cells:
    #                     print(f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}")
    #                     raise ValueError
    #                 #Check if sufficient features obtained
    #                 if type(downsample_interactions) == float:
    #                     if len(int_list_names) == int(len(all_features_flat)*downsample_interactions):
    #                         break
    #                 elif type(downsample_interactions) == int:
    #                     if len(int_list_names) == downsample_interactions:
    #                         break
    #                 elif type(downsample_interactions) == dict:
    #                     if len(c_split) in int_order_dict_retained.keys():
    #                         if int_order_dict_retained[len(c_split)] > downsample_interactions[len(c_split)] and downsample_interactions[len(c_split)]!=(-1):
    #                             int_list.pop()
    #                             int_list_names.pop()
    #                             int_order_dict_retained[len(c_split)] -= 1
    #                             break
    #                         elif int_order_dict_retained == downsample_interactions:
    #                             break
    #                 #Write chunk to disk
    #                 int_list = self.write_features(
    #                     feature_list = int_list, 
    #                     feature_chunk_size = chunk_size)
    #     #Write final chunk to disk
    #     int_list = self.write_features(
    #         feature_list = int_list, 
    #         feature_chunk_size = chunk_size,
    #         final_chunk = True)

    #     print("... Total retained features (order:count): "+", ".join([str(i)+":"+str(int_order_dict_retained[i])+" ("+str(round(int_order_dict_retained[i]/int_order_dict[i]*100, 1))+"%)" for i in sorted(int_order_dict_retained.keys())]))

    #     #Transpose disk data in place
    #     print("Transposing features")
    #     self.transpose_features()

    #     #Save interaction feature names
    #     self.feature_names += int_list_names

    def one_hot_encode_interactions(
        self, 
        max_order = 2,
        max_cells = 1e12,
        min_observed = 2,
        features = [],
        downsample_interactions = None,
        seed = 1):
        """
        Add interaction terms to 1-hot encoding DataFrame.

        :param max_order: Maximum interaction order (default:2).
        :param max_cells: Maximum matrix cells permitted (default:1trillion).
        :param min_observed: Minimum number of observations required to include interaction term (default:2).
        :param features: list of feature names to filter (default:[] i.e. all  features retained).
        :param downsample_interactions: number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (optional).
        :param seed: Random seed for downsampling interactions (default:1).
        :returns: Nothing.
        """

        #Check if no interactions to add
        if max_order<2:
            self.Xohi = copy.deepcopy(self.Xoh)
            #Filter features
            if features!=[]:
                print("Filtering features")
                self.Xohi = self.filter_features(
                    input_df = self.Xohi,
                    features = features)
            return

        #Check downsample_interactions argument valid
        if downsample_interactions!=None:
            if type(downsample_interactions) == float:
                #Downsample observations by proportion
                if downsample_interactions >= 1 or downsample_interactions <= 0:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                    raise ValueError
            elif type(downsample_interactions) == int:
                #Downsample observations by number
                if downsample_interactions < 1:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                    raise ValueError
            elif type(downsample_interactions) == str:
                try:
                    downsample_interactions = {(i+2):int(d) for i,d in enumerate(str(downsample_interactions).split(","))}
                except:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                    raise ValueError
            else:
                print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
                raise ValueError

        metadata_path, memmap_path = self.get_xohi_cache_paths(
            max_order = max_order,
            min_observed = min_observed,
            features = features,
            downsample_interactions = downsample_interactions,
            seed = seed)
        if self.load_cached_interactions(
            metadata_path = metadata_path,
            memmap_path = memmap_path):
            if features!=[]:
                print("Filtering features")
                if self.is_lazy_feature_matrix() or self.is_sparse_feature_matrix():
                    self.reorder_feature_columns(
                        [i for i in self.get_feature_names() if i in features])
                else:
                    self.Xohi = self.filter_features(
                        input_df = self.Xohi,
                        features = features)
            self.feature_names = self.get_feature_names()
            return

        #Get all theoretical interactions
        all_features,int_order_dict = self.get_theoretical_interactions(max_order = max_order)
        print("... Total theoretical features (order:count): "+", ".join([str(i)+":"+str(int_order_dict[i]) for i in sorted(int_order_dict.keys())]))
        #Flatten
        all_features_flat = list(itertools.chain(*list(all_features.values())))
        xoh_values = self.Xoh.to_numpy(dtype = np.uint8, copy = False)
        xoh_column_index = {name:i for i,name in enumerate(self.Xoh.columns)}

        #Check if all interaction features exist (i.e. with mutation order>1)
        invalid_features = [i for i in features if (i not in all_features_flat) and (len(i.split('_'))>1)]
        if len(invalid_features) != 0:
            # print("Error: Invalid feature names.")
            print("Warning: Invalid feature names: "+",".join(invalid_features))
            # raise ValueError

        #Select interactions
        int_set = set()
        int_order_dict_retained = {}
        int_list_names = []
        interaction_row_index_map = {}
        #No shuffle if not downsampling
        if downsample_interactions is None:
            all_features_loop = {0: all_features_flat}
        #Shuffle flattened features
        elif type(downsample_interactions) in [float, int]:
            random.seed(seed)
            all_features_loop = {0: random.sample(all_features_flat, len(all_features_flat))}
        #Shuffle features separately per order
        else:
            all_features_loop = {k:random.sample(all_features[k], len(all_features[k])) for k in all_features}

        #Loop over all orders
        for n in all_features_loop.keys():
            #Loop over all features of this order
            for c in all_features_loop[n]:
                c_split = c.split("_")
                #Check if feature desired
                if (c in features) or features==[]:
                    feature_idx = [xoh_column_index[i] for i in c_split]
                    int_col = np.all(xoh_values[:,feature_idx] == 1, axis = 1).astype(np.uint8, copy = False)
                    #Check if minimum number of observations satisfied
                    if int(np.sum(int_col)) >= min_observed:
                        int_list_names += [c]
                        int_set.add(c)
                        interaction_row_index_map[c] = np.flatnonzero(int_col).astype(np.int32, copy = False)
                        if len(c_split) not in int_order_dict_retained.keys():
                            int_order_dict_retained[len(c_split)] = 1
                        else:
                            int_order_dict_retained[len(c_split)] += 1
                    # else:
                    #     if len(c_split)==3 and sum(int_col)==1:
                    #         print(c)
                    #Check memory footprint
                    if len(int_list_names)*len(self.Xoh) > max_cells:
                        print(f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}")
                        raise ValueError
                    #Check if sufficient features obtained
                    if type(downsample_interactions) == float:
                        if len(int_list_names) == int(len(all_features_flat)*downsample_interactions):
                            break
                    elif type(downsample_interactions) == int:
                        if len(int_list_names) == downsample_interactions:
                            break
                    elif type(downsample_interactions) == dict:
                        if len(c_split) in int_order_dict_retained.keys():
                            if int_order_dict_retained[len(c_split)] > downsample_interactions[len(c_split)] and downsample_interactions[len(c_split)]!=(-1):
                                removed_name = int_list_names.pop()
                                int_set.remove(removed_name)
                                del interaction_row_index_map[removed_name]
                                int_order_dict_retained[len(c_split)] -= 1
                                break
                            elif int_order_dict_retained == downsample_interactions:
                                break

        print("... Total retained features (order:count): "+", ".join([str(i)+":"+str(int_order_dict_retained[i])+" ("+str(round(int_order_dict_retained[i]/int_order_dict[i]*100, 1))+"%)" for i in sorted(int_order_dict_retained.keys())]))

        #Concatenate into dataframe
        if len(int_list_names)>0:
            ordered_interactions = [i for i in all_features_flat if i in int_set]
            if get_feature_store_backend() == "sparse":
                self.build_sparse_feature_matrix(
                    columns = list(self.Xoh.columns) + ordered_interactions,
                    interaction_names = ordered_interactions,
                    interaction_row_indices = [interaction_row_index_map[i] for i in ordered_interactions])
            else:
                self.set_lazy_feature_matrix(list(self.Xoh.columns) + ordered_interactions)
            self.save_cached_interactions(
                metadata_path = metadata_path,
                columns = list(self.get_feature_names()))
        else:
            self.Xohi = copy.deepcopy(self.Xoh)
            self.feature_matrix_mode = "dense"
            self.feature_sparse_matrix = None
            self.feature_source_indices = None
            self.feature_component_indices = None

        #Filter features
        if features!=[]:
            print("Filtering features")
            if self.is_lazy_feature_matrix() or self.is_sparse_feature_matrix():
                self.reorder_feature_columns(
                    [i for i in self.get_feature_names() if i in features])
            else:
                self.Xohi = self.filter_features(
                    input_df = self.Xohi,
                    features = features)

        #Save interaction feature names
        self.feature_names = self.get_feature_names()

        # Drop temporary builders before later preprocessing stages allocate
        # additional wide matrices over the same feature set.
        del all_features
        del all_features_flat
        del int_set
        del int_list_names
        del interaction_row_index_map
        del xoh_values
        if 'ordered_interactions' in locals():
            del ordered_interactions
        gc.collect()

    def get_xohi_values(
        self):
        """
        Return an array-like feature matrix view without forcing DataFrame copies.

        :returns: numpy-compatible 2D array.
        """
        if self.is_lazy_feature_matrix():
            print("Error: Lazy feature matrix cannot be returned as one dense array.")
            raise RuntimeError
        if self.is_sparse_feature_matrix():
            return self.feature_sparse_matrix
        if self.Xohi_memmap is not None:
            return self.Xohi_memmap
        return self.Xohi.to_numpy(copy = False)

    def filter_features(
        self, 
        input_df,
        features):
        """
        Filter features by name.

        :param input_df: DataFrame of features to filter.
        :param features: list of feature names to filter (default:[] i.e. all features retained).
        :returns: filtered DataFrame.
        """
        #Check if all features exist 
        invalid_features = [i for i in features if i not in input_df.columns]
        if len(invalid_features) != 0:
            # print("Error: Invalid feature names.")
            print("Warning: Invalid feature names: "+",".join(invalid_features))
            # raise ValueError
        #Filter features
        features_order = [i for i in input_df.columns if i in features]
        return input_df.loc[:,features_order]

    def H_matrix(
        self,
        str_geno,
        str_coef,
        num_states = 2,
        invert = False):
        """
        Construct Walsh-Hadamard matrix.

        :param str_geno: list of genotype strings where '0' indicates WT state.
        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :returns: Walsh-Hadamard matrix as a numpy matrix.
        """
        #Genotype string length
        string_length = len(str_geno[0])
        #Number of states per position in genotype string (float)
        if type(num_states) == int:
            num_states = [float(num_states) for i in range(string_length)]
        else:
            num_states = [float(i) for i in num_states]
        #Convert reference characters to "." and binary encode
        str_coef = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
        str_geno = [[ord(j) for j in i] for i in str_geno]
        #Matrix representations
        num_statesi = np.repeat([num_states], len(str_geno)*len(str_coef), axis = 0)
        str_genobi = np.repeat(str_geno, len(str_coef), axis = 0)
        str_coefbi = np.transpose(np.tile(np.transpose(np.asarray(str_coef)), len(str_geno)))
        str_genobi_eq_str_coefbi = (str_genobi == str_coefbi)
        #Factors
        row_factor2 = str_genobi_eq_str_coefbi.sum(axis = 1)
        if invert:
            row_factor1 = np.prod(str_genobi_eq_str_coefbi * (num_statesi-2) + 1, axis = 1)       
            return ((row_factor1 * np.power(-1, row_factor2))/np.prod(num_states)).reshape((len(str_geno),-1))
        else:
            row_factor1 = (np.logical_or(np.logical_or(str_genobi_eq_str_coefbi, str_genobi==ord('0')), str_coefbi==ord('.')).sum(axis = 1) == string_length).astype(float)            
            return ((row_factor1 * np.power(-1, row_factor2))).reshape((len(str_geno),-1))

    def H_matrix_chunker(
        self,
        str_geno,
        str_coef,
        num_states = 2,
        invert = False,
        chunk_size = 1000):
        """
        Construct Walsh-Hadamard matrix in chunks.

        :param str_geno: list of genotype strings where '0' indicates WT state.
        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :param chunk_size: chunk size in number of genotypes/variants (default:1000).
        :returns: Walsh-Hadamard matrix as a numpy matrix.
        """

        #Check if chunking not necessary
        if len(str_geno) < chunk_size:
            return self.H_matrix(
                str_geno = str_geno, 
                str_coef = str_coef, 
                num_states = num_states, 
                invert = invert)

        #Chunk
        hmat_list = []
        for i in range(math.ceil(len(str_geno)/chunk_size)):
            from_i = (i*chunk_size)
            to_i = (i+1)*chunk_size
            if to_i > len(str_geno):
                to_i = len(str_geno)
            hmat_list += [self.H_matrix(
                str_geno = str_geno[from_i:to_i], 
                str_coef = str_coef, 
                num_states = num_states, 
                invert = invert)]
        return np.concatenate(hmat_list, axis = 0)

    def V_matrix(
        self,
        str_coef,
        num_states = 2,
        invert = False):
        """
        Construct diagonal weighting matrix.

        :param str_coef: list of coefficient strings where '0' indicates WT state.
        :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
        :param invert: invert the matrix.
        :returns: diagonal weighting matrix as a numpy matrix.
        """
        #Genotype subset
        str_geno = str_coef
        #Genotype string length
        string_length = len(str_geno[0])
        #Number of states per position in genotype string
        if type(num_states) == int:
            num_states = [float(num_states) for i in range(string_length)]
        else:
            num_states = [float(i) for i in num_states]
        #Convert reference characters to "."
        str_coef_ = [i.replace("0", ".") for i in str_coef]
        #initialize V matrix
        V = np.array([[0.0]*len(str_coef)]*len(str_geno))
        #Fill matrix
        for i in range(len(str_geno)):
            factor1 = int(np.prod([c for a,b,c in zip(str_coef_[i], str_geno[i], num_states) if ord(a) != ord(b)]))
            factor2 = sum([1 for a,b in zip(str_coef_[i], str_geno[i]) if ord(a) == ord(b)])
            if invert:
                V[i,i] = factor1 * np.power(-1, factor2)
            else:
                V[i,i] = 1/(factor1 * np.power(-1, factor2))
        return V

    def coefficient_to_sequence(
        self,
        coefficient,
        length):
        """
        Get sequence representation of a coefficient string.

        :param coefficient: coefficient string.
        :param length: integer sequence length.
        :returns: sequence string.
        """
        #initialize sequence string
        coefficient_seq = ['0']*length
        #Wild-type sequence string
        if coefficient == "WT":
            return ''.join(coefficient_seq)
        #Variant sequence string
        for i in coefficient.split("_"):
            coefficient_seq[int(i[1:-1])-1] = i[-1]
        return ''.join(coefficient_seq)

    def ensemble_encode_features(
        self):
        """
        Ensemble encode features.

        :returns: Nothing.
        """
        #Wild-type mask variant sequences
        geno_list = list(self.fdata.vtable.apply(lambda row : "".join(x if x!=y else '0' for x,y in zip(str(row[self.fdata.variantCol]),self.fdata.wildtype)),
            axis = 1))
        #Sequence representation of 1-hot encoded coefficients/features
        ceof_list = [self.coefficient_to_sequence(coef, len(self.fdata.wildtype)) for coef in self.get_feature_names()]
        #Number of states per position
        state_list = (self.X.apply(lambda column: column.value_counts(), axis = 0)>0).apply(lambda column: column.value_counts(), axis = 0)
        state_list = list(np.asarray(state_list)[0])
        #Ensemble encode features
        start = time.time()
        hmat_inv = self.H_matrix_chunker(
            str_geno = geno_list, 
            str_coef = ceof_list, 
            num_states = state_list, 
            invert = True)
        end = time.time()
        print("Construction time for H_matrix :", end-start)
        vmat_inv = self.V_matrix(
            str_coef = ceof_list, 
            num_states = state_list, 
            invert = True)
        return pd.DataFrame(np.matmul(hmat_inv, vmat_inv), columns = self.get_feature_names())

    def update_holdout_observations(
        self, 
        input_df):
        """
        Update holdout status for observations.

        :param input_df: DataFrame of features to use for update.
        :returns: Nothing.
        """
        #Default: hold out all orders
        if self.holdout_orders == []:
            self.holdout_orders = list(set(self.fdata.vtable[self.fdata.mutationOrderCol]))

        #Variants that can be held out (determined separately for each additive trait)
        all_traits_unique = list(set([item for sublist in list(self.model_design['trait']) for item in sublist]))
        #Initialize holdout status (all variants can be held out)
        if self.cvgroups is None:
            self.cvgroups = pd.DataFrame({
                "holdout" : np.array([1]*len(input_df))
                })
        #Consider each additive trait separately
        for t in all_traits_unique:
            #Phenotypes reporting on this trait
            relevant_phenotype_columns = ["phenotype_"+str(self.model_design.loc[i,'phenotype']) for i in range(len(self.model_design)) if t in self.model_design.loc[i,'trait']]
            #Number of observations per coefficient
            Xohp_colsum = pd.DataFrame(input_df.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,:].sum(axis=0))
            #Indices of coefficients that do not meet required threshold
            Xohp_noholdout = list(Xohp_colsum.loc[Xohp_colsum.iloc[:,0]<self.holdout_minobs,:].index)
            #Observations of coefficients that do not meet the required threshold
            Xohp_noholdout_rowsum = np.array(input_df.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,Xohp_noholdout].sum(axis=1))
            #WT variants for these phenotypes
            Xohp_WT = np.array(self.fdata.vtable.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'WT'])
            #Mutation orders for these phenotypes
            Xohp_mutationOrder = np.array(self.fdata.vtable.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,self.fdata.mutationOrderCol])
            #Current holdout status
            current_status = list(self.cvgroups.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'holdout'])
            #Holdout status for this additive trait
            noholdout_minobs = [Xohp_noholdout_rowsum[i]!=0 for i in range(len(Xohp_noholdout_rowsum))]
            noholdout_orders = [Xohp_mutationOrder[i] not in self.holdout_orders for i in range(len(Xohp_noholdout_rowsum))]
            noholdout_WT = [((Xohp_WT[i]==True) & (self.holdout_WT==False)) for i in range(len(Xohp_noholdout_rowsum))]
            noholdout = [(noholdout_minobs[i] | noholdout_orders[i] | noholdout_WT[i]) for i in range(len(Xohp_noholdout_rowsum))]
            #New holdout status
            self.cvgroups.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'holdout'] = np.asarray([int((current_status[i]==1) & (noholdout[i]==False)) for i in range(len(Xohp_noholdout_rowsum))])
        
    def define_cross_validation_groups(
        self):
        """
        Define cross-validation groups.

        :returns: Nothing.
        """

        #Define holdout observations
        if self.cvgroups is None:
            #Default: hold out all orders
            if self.holdout_orders == []:
                self.holdout_orders = list(set(self.fdata.vtable[self.fdata.mutationOrderCol]))

            #Variants that can be held out (determined separately for each additive trait)
            all_traits_unique = list(set([item for sublist in list(self.model_design['trait']) for item in sublist]))
            #Initialize holdout status (all variants can be held out)
            self.cvgroups = pd.DataFrame({
                "holdout" : np.array([1]*len(self))
                })
            #Consider each additive trait separately
            for t in all_traits_unique:
                #Phenotypes reporting on this trait
                relevant_phenotype_columns = ["phenotype_"+str(self.model_design.loc[i,'phenotype']) for i in range(len(self.model_design)) if t in self.model_design.loc[i,'trait']]
                phenotype_mask = np.asarray(self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1)
                phenotype_indices = np.flatnonzero(phenotype_mask)
                #Number of observations per coefficient
                Xohp_colsum = self.sum_features_for_rows(phenotype_indices)
                #Indices of coefficients that do not meet required threshold
                Xohp_noholdout = np.flatnonzero(Xohp_colsum < self.holdout_minobs)
                #Observations of coefficients that do not meet the required threshold
                Xohp_noholdout_rowsum = self.sum_selected_features_per_row(
                    row_indices = phenotype_indices,
                    feature_indices = Xohp_noholdout)
                #WT variants for these phenotypes
                Xohp_WT = np.array(self.fdata.vtable.loc[phenotype_mask,'WT'])
                #Mutation orders for these phenotypes
                Xohp_mutationOrder = np.array(self.fdata.vtable.loc[phenotype_mask,self.fdata.mutationOrderCol])
                #Current holdout status
                current_status = list(self.cvgroups.loc[phenotype_mask,'holdout'])
                #Holdout status for this additive trait
                noholdout_minobs = [Xohp_noholdout_rowsum[i]!=0 for i in range(len(Xohp_noholdout_rowsum))]
                noholdout_orders = [Xohp_mutationOrder[i] not in self.holdout_orders for i in range(len(Xohp_noholdout_rowsum))]
                noholdout_WT = [((Xohp_WT[i]==True) & (self.holdout_WT==False)) for i in range(len(Xohp_noholdout_rowsum))]
                noholdout = [(noholdout_minobs[i] | noholdout_orders[i] | noholdout_WT[i]) for i in range(len(Xohp_noholdout_rowsum))]
                #New holdout status
                self.cvgroups.loc[phenotype_mask,'holdout'] = np.asarray([int((current_status[i]==1) & (noholdout[i]==False)) for i in range(len(Xohp_noholdout_rowsum))])
        
        #Total number of variants that can be held out
        n_holdout = sum(self.cvgroups.holdout)

        #Hold out folds
        holdout_fold = (list(range(1, self.k_folds+1))*(int(n_holdout/self.k_folds)+1))[:n_holdout]
        random.seed(self.seed)
        random.shuffle(holdout_fold)

        #Add to cvgroups DataFrame
        self.cvgroups['fold'] = None
        self.cvgroups.loc[self.cvgroups.holdout==1,'fold'] = holdout_fold

        #Add cross validation groups
        for i in range(self.k_folds):
            self.cvgroups['fold_'+str(i+1)] = "training"
            self.cvgroups.loc[self.cvgroups['fold']==(i+1),'fold_'+str(i+1)] = "test"
            val_groups = [(j%self.k_folds)+1 for j in list(range(i+1, i+1+self.validation_factor))]
            for j in val_groups:
                self.cvgroups.loc[self.cvgroups['fold']==j,'fold_'+str(i+1)] = "validation"

    def define_coefficient_groups(
        self,
        k_folds = 10):
        """
        Define coefficient groups.

        :param k_folds: Number of cross-validation folds (default:10).
        :returns: Nothing.
        """
        # Coefficients that can be fit (for each phenotype and fold). Work in
        # manageable column chunks to avoid creating another full-width pandas
        # temporary over the memmap-backed feature matrix.
        n_features = len(self.get_feature_names())
        phenotype_values = self.phenotypes.to_numpy(dtype = np.uint8, copy = False)
        fold_values = {
            'fold_'+str(i+1): self.cvgroups['fold_'+str(i+1)].to_numpy(copy = False)
            for i in range(k_folds)}
        coefficient_chunk_size = int(os.environ.get("MOCHI_COEFFICIENT_CHUNK_SIZE", "1024"))
        self.coefficients = {}
        for p_index, p in enumerate(self.phenotypes.columns):
            self.coefficients[p] = np.empty((n_features, k_folds), dtype = np.uint8)
            phenotype_mask = phenotype_values[:, p_index] == 1
            for i in range(k_folds):
                row_indices = np.flatnonzero(np.logical_and(phenotype_mask, fold_values['fold_'+str(i+1)] == "training"))
                for chunk_start, chunk_end, chunk in self.iterate_feature_chunks(
                    row_indices = row_indices,
                    chunk_size = coefficient_chunk_size):
                    self.coefficients[p][chunk_start:chunk_end, i] = np.any(
                        chunk != 0,
                        axis = 0).astype(np.uint8, copy = False)

        # Coefficients specified to be fit (for each additive trait)
        self.coefficients_userspec = np.ones(
            (n_features, len(self.additive_trait_names)),
            dtype = np.uint8)
        xohi_columns = np.asarray(self.get_feature_names())
        for t in range(len(self.additive_trait_names)):
            if self.additive_trait_names[t] in self.features_trait.keys():
                allowed_features = set(self.features_trait[self.additive_trait_names[t]])
                self.coefficients_userspec[:, t] = np.asarray(
                    [int(i in allowed_features) for i in xohi_columns],
                    dtype = np.uint8)
        gc.collect()

    def is_valid_instance(
        self):
        """
        Check object is valid instance.

        :returns: bool.
        """
        not_none = [
            self.fdata,
            self.additive_trait_names,
            self.phenotype_names,
            self.fitness,
            self.phenotypes,
            self.Xoh,
            self.Xohi,
            self.cvgroups,
            self.coefficients,
            self.coefficients_userspec]
        return sum([1 for i in not_none if i is None]) == 0

    def get_mask_tensor(
        self,
        fold = 1):
        """
        Build the fold-specific coefficient mask tensor used by the model.

        :param fold: Cross-validation fold (default:1).
        :returns: Torch tensor.
        """
        mask_tensor = torch.tensor(
            np.stack(
                [self.coefficients["phenotype_"+str(i+1)][:, fold-1] for i in range(len(self.coefficients))],
                axis = 1),
            dtype = torch.float32)
        mask_tensor = torch.transpose(mask_tensor, 0, 1)
        mask_tensor = torch.reshape(mask_tensor, (1, mask_tensor.shape[0], mask_tensor.shape[1]))
        mask_tensor = mask_tensor.expand(len(self.additive_trait_names), mask_tensor.shape[1], mask_tensor.shape[2])
        mask_us = torch.transpose(torch.tensor(self.coefficients_userspec, dtype=torch.float32), 0, 1)
        mask_us = torch.reshape(mask_us, (mask_us.shape[0],1,mask_us.shape[1]))
        return mask_tensor * mask_us

    def get_split_observation_data(
        self,
        fold = 1,
        seed = 1,
        training_resample = True):
        """
        Get fold split metadata without materializing the full feature matrix.

        :param fold: Cross-validation fold (default:1).
        :param seed: Random seed for training target resampling (default:1).
        :param training_resample: Whether or not to add noise to training targets (default:True).
        :returns: Dictionary keyed by split name.
        """
        if not self.is_valid_instance():
            print("Error: Invalid MochiData instance.")
            raise ValueError
        fold_name = "fold_"+str(fold)
        data_dict = {}
        mask_tensor = self.get_mask_tensor(fold = fold)
        for g in list(set(self.cvgroups[fold_name])):
            group_mask = np.asarray(self.cvgroups[fold_name] == g)
            row_indices = np.flatnonzero(group_mask)
            data_dict[g] = {
                'row_indices': row_indices,
                'mask': mask_tensor,
                'select': torch.tensor(
                    self.phenotypes.loc[group_mask,:].to_numpy(dtype = np.float32, copy = True),
                    dtype = torch.float32)}
            y_values = self.fitness.loc[group_mask,'fitness'].to_numpy(dtype = np.float32, copy = True)
            if g == "training" and training_resample:
                np.random.seed(seed+fold*1000000)
                y_values = y_values + np.asarray(
                    [np.random.normal(scale = i) for i in list(self.fitness.loc[group_mask,'sigma'])],
                    dtype = np.float32)
            data_dict[g]['y'] = torch.reshape(torch.tensor(y_values, dtype = torch.float32), (-1, 1))
            data_dict[g]['y_wt'] = torch.reshape(
                torch.tensor(
                    self.fitness.loc[group_mask,'weight'].to_numpy(dtype = np.float32, copy = True),
                    dtype = torch.float32),
                (-1, 1))
        return data_dict

    def get_data(
        self, 
        fold = 1,
        seed = 1,
        training_resample = True):
        """
        Get data for a specified cross-validation fold.

        :param fold: Cross-validation fold (default:1).
        :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
        :returns: Dictionary of dictionaries of tensors.
        """
        #Check for fitness data
        if not self.is_valid_instance():
            print("Error: Invalid MochiData instance.")
            raise ValueError
        fold_name = "fold_"+str(fold)
        split_data = self.get_split_observation_data(
            fold = fold,
            seed = seed,
            training_resample = training_resample)
        data_dict = {}
        feature_numpy_dtype = np.uint8 if compact_feature_tensors() else np.float32
        feature_tensor_dtype = torch.uint8 if compact_feature_tensors() else torch.float32
        for g in list(set(self.cvgroups[fold_name])):
            row_indices = split_data[g]['row_indices']
            sind = list(range(len(row_indices)))
            if g=="training":
                random.seed(seed+fold*1000000)
                random.shuffle(sind)
            data_dict[g] = {}
            data_dict[g]['select'] = split_data[g]['select'][sind,:]
            data_dict[g]['mask'] = split_data[g]['mask']
            #Feature tensor
            feature_values = self.materialize_feature_matrix(
                row_indices = row_indices,
                dtype = feature_numpy_dtype)
            data_dict[g]['X'] = torch.tensor(feature_values[sind,:], dtype = feature_tensor_dtype)
            data_dict[g]['y'] = split_data[g]['y'][sind,:]
            data_dict[g]['y_wt'] = split_data[g]['y_wt'][sind,:]
        return data_dict

    def get_data_index(
        self, 
        indices = []):
        """
        Get data corresponding to specific variants (by index).

        :param indices: Variant/observation indices (default:None i.e. all indices).
        :returns: Dictionary of dictionary of tensors (select, feature and target tensors only).
        """
        #Set to all indices if empty list
        if indices == []:
            indices = list(self.phenotypes.index)

        data_dict = {}
        feature_numpy_dtype = np.uint8 if compact_feature_tensors() else np.float32
        feature_tensor_dtype = torch.uint8 if compact_feature_tensors() else torch.float32
        #Select tensor
        data_dict['select'] = torch.tensor(
            self.phenotypes.iloc[indices,:].to_numpy(dtype = np.float32, copy = True),
            dtype = torch.float32)
        #Feature tensor
        data_dict['X'] = torch.tensor(
            self.materialize_feature_matrix(
                row_indices = indices,
                dtype = feature_numpy_dtype),
            dtype = feature_tensor_dtype)
        #Target tensor
        data_dict['y'] = torch.reshape(
            torch.tensor(
                self.fitness.iloc[indices,:]['fitness'].to_numpy(dtype = np.float32, copy = True),
                dtype = torch.float32),
            (-1, 1))
        return data_dict

    def __len__(self):
        """
        Length of object.
        :returns: Number of variants (length of fdata).
        """
        return len(self.fdata)

# class MochiDataset(Dataset):
#     def __init__(
#         self, 
#         root, 
#         data,
#         dataset_type = 'training', 
#         fold = 1, 
#         seed = 1,
#         training_resample = True,
#         transform = None):
#         """
#         Initialize a MochiDataset object.

#         :param root: Path to data directory (required).
#         :param data: An instance of the MochiData class (required).
#         :param dataset_type: One of 'training', 'validation', 'test' (default:'training').
#         :param fold: Cross-validation fold (default:1).
#         :param seed: Random seed for both training target data resampling and shuffling training data (default:1).
#         :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
#         :param transform: Transform to apply to feature data (default:None).
#         :returns: MochiTask object.
#         """ 
#         self.root = root
#         self.data = data
#         self.dataset_type = dataset_type
#         self.fold_name = "fold_"+str(fold)
#         self.seed = seed
#         self.training_resample = training_resample
#         self.transform = transform
#         idx_list = list(self.data.cvgroups.loc[self.data.cvgroups[self.fold_name]==self.dataset_type,:].index)
#         self.idx_dict = {i:idx_list[i] for i in range(len(idx_list))}

#         #Target data
#         self.y = pd.DataFrame(self.data.fitness.loc[self.data.cvgroups[self.fold_name]==self.dataset_type,'fitness'])
#         #Add random noise to training target data proportional to target error (if specified)
#         if self.dataset_type=="training" and self.training_resample:
#             np.random.seed(self.seed)
#             self.y['noise'] = [np.random.normal(scale = i) for i in list(self.data.fitness.loc[self.data.cvgroups[self.fold_name]==self.dataset_type,'sigma'])]
#             self.y['y'] = pd.DataFrame(self.y.sum(axis = 1))

#     def __getitem__(self, idx):
#         #Observation
#         obs_number = self.idx_dict[idx]

#         #Feature tensor
#         file_name = "data_chunk"+str(obs_number - obs_number % 100)+".csv"
#         line_number = obs_number % 100
#         line = linecache.getline(os.path.join(self.root, file_name), line_number)
#         csv_line = csv.reader([line])
#         X = torch.tensor(np.asarray([int(x) for x in next(csv_line)]), dtype=torch.float32)

#         #Target tensor
#         y = self.y.loc[obs_number,'fitness']
#         y = torch.reshape(torch.tensor(np.asarray(y), dtype=torch.float32), (-1, 1))

#         return X,y

#     def __len__(self):
#         return len(self.idx_dict)

# class CustomImageDataset(Dataset):
#     def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
#         self.img_labels = pd.read_csv(annotations_file)
#         self.img_dir = img_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label
        
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source: https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
    """
    def __init__(
        self, 
        *tensors, 
        batch_size = 32, 
        shuffle = False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :returns: FastTensorDataLoader object.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
    def __iter__(self):
        if self.shuffle:
            self.indices = torch.randperm(self.dataset_len, device = self.tensors[0].device)
        else:
            self.indices = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        if self.indices is None:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        else:
            batch_indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(t[batch_indices] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class MaterializingRowDataLoader:
    """
    A DataLoader-like object with cached, prefetched feature blocks.
    """
    def __init__(
        self,
        data,
        row_indices,
        select,
        y,
        y_wt,
        batch_size = 32,
        shuffle = False):
        """
        Initialize an on-demand row loader.

        :param data: MochiData instance (required).
        :param row_indices: Row indices for this split (required).
        :param select: Select tensor (required).
        :param y: Target tensor (required).
        :param y_wt: Target-weight tensor (required).
        :param batch_size: Batch size to load (default:32).
        :param shuffle: Whether to shuffle batches each epoch (default:False).
        :returns: MaterializingRowDataLoader object.
        """
        self.data = data
        self.row_indices = np.asarray(row_indices, dtype = np.int64)
        self.select = select
        self.y = y
        self.y_wt = y_wt
        assert self.select.shape[0] == len(self.row_indices)
        assert self.y.shape[0] == len(self.row_indices)
        assert self.y_wt.shape[0] == len(self.row_indices)
        self.dataset_len = len(self.row_indices)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.feature_numpy_dtype = np.uint8 if compact_feature_tensors() else np.float32
        self.feature_tensor_dtype = torch.uint8 if compact_feature_tensors() else torch.float32
        self.prefetch_blocks = max(1, int(os.environ.get("MOCHI_PREFETCH_BLOCKS", "2")))
        self.prefetch_batches = max(1, int(os.environ.get("MOCHI_PREFETCH_BATCHES", "8")))
        self.block_rows = max(
            self.batch_size,
            int(os.environ.get("MOCHI_FEATURE_BLOCK_ROWS", str(self.batch_size * self.prefetch_batches))))
        self.max_cached_blocks = max(1, int(os.environ.get("MOCHI_MAX_CACHED_BLOCKS", "3")))
        self.pin_memory = torch.cuda.is_available()
        self.block_slices = [
            (start, min(start + self.block_rows, self.dataset_len))
            for start in range(0, self.dataset_len, self.block_rows)]
        self.block_cache = collections.OrderedDict()
        self._prefetch_thread = None
        self._prefetch_queue = None
        self._current_block = None
        self._current_block_pos = 0
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def _maybe_pin(
        self,
        tensor):
        """
        Pin CPU tensor memory when CUDA is available.

        :param tensor: Tensor to pin (required).
        :returns: Tensor.
        """
        if self.pin_memory and tensor.device.type == "cpu":
            return tensor.pin_memory()
        return tensor

    def _get_cached_block(
        self,
        block_id):
        """
        Get or materialize a canonical split-local feature block.

        :param block_id: Canonical block identifier (required).
        :returns: Tuple of tensors.
        """
        if block_id in self.block_cache:
            self.block_cache.move_to_end(block_id)
            return self.block_cache[block_id]
        start, stop = self.block_slices[block_id]
        block = (
            self._maybe_pin(self.select[start:stop].contiguous()),
            self._maybe_pin(torch.tensor(
                self.data.materialize_feature_matrix(
                    row_indices = self.row_indices[start:stop],
                    dtype = self.feature_numpy_dtype),
                dtype = self.feature_tensor_dtype)),
            self._maybe_pin(self.y[start:stop].contiguous()),
            self._maybe_pin(self.y_wt[start:stop].contiguous()))
        self.block_cache[block_id] = block
        self.block_cache.move_to_end(block_id)
        while len(self.block_cache) > self.max_cached_blocks:
            self.block_cache.popitem(last = False)
        return block

    def _prefetch_worker(
        self):
        """
        Materialize upcoming blocks in the background.

        :returns: Nothing.
        """
        for block_id, block_order in self._epoch_blocks:
            block = self._get_cached_block(block_id)
            self._prefetch_queue.put((block, block_order))
        self._prefetch_queue.put(None)

    def __iter__(self):
        block_ids = list(range(len(self.block_slices)))
        if self.shuffle:
            block_ids = torch.randperm(len(block_ids)).tolist()
        self._epoch_blocks = []
        for block_id in block_ids:
            start, stop = self.block_slices[block_id]
            block_len = stop - start
            if self.shuffle:
                block_order = torch.randperm(block_len)
            else:
                block_order = None
            self._epoch_blocks.append((block_id, block_order))
        self._prefetch_queue = queue.Queue(maxsize = self.prefetch_blocks)
        self._prefetch_thread = threading.Thread(
            target = self._prefetch_worker,
            daemon = True)
        self._prefetch_thread.start()
        self._current_block = None
        self._current_block_pos = 0
        self._batches_yielded = 0
        return self

    def __next__(self):
        while True:
            if self._current_block is None or self._current_block_pos >= self._current_block[0].shape[0]:
                next_item = self._prefetch_queue.get()
                if next_item is None:
                    raise StopIteration
                self._current_block, self._current_order = next_item
                self._current_block_pos = 0
            select_block, X_block, y_block, y_wt_block = self._current_block
            stop = min(self._current_block_pos + self.batch_size, select_block.shape[0])
            if self._current_order is None:
                batch_index = slice(self._current_block_pos, stop)
            else:
                batch_index = self._current_order[self._current_block_pos:stop]
            self._current_block_pos = stop
            self._batches_yielded += 1
            return (
                select_block[batch_index],
                X_block[batch_index],
                y_block[batch_index],
                y_wt_block[batch_index])

    def __len__(self):
        return self.n_batches

