
"""
MoCHI data module
"""

import re
import copy
import random
import math
from os.path import exists
import pyreadr
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
import itertools
import collections, functools, operator

import time

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
        #Check file exists
        if not exists(file_path):
            print("Error: RData file not found.")
            return vtable
        #Read file
        try:
            vtable = pyreadr.read_r(file_path)
        except:
            print("Error: Invalid RData file: cannot read file.")
            return vtable
        #Check variant fitness object exists
        if 'all_variants' not in vtable.keys():
            print("Error: Invalid RData file: variant data not found.")
            return vtable
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
        #Check file exists
        if not exists(file_path):
            print("Error: Plain text file not found.")
            return vtable
        #Read file
        try:
            vtable = pd.read_csv(file_path, sep = None, engine='python')
        except:
            print("Error: Invalid plain text file: cannot read file.")
            return vtable
        return vtable

    def read_fitness(
        self, 
        file_path, 
        name = None, 
        order_subset = None,
        downsample_observations = None,
        seed = 1):
        """
        Read fitness from DiMSum RData file.

        :param file_path: path to RData file (required).
        :param name: name of fitness dataset (optional).
        :param order_subset: list of mutation orders corresponding to retained variants (optional).
        :param downsample_observations: number (if integer) or proportion (if float) of observations to retain including WT (optional).
        :param seed: Random seed for downsampling observations (default:1).
        :returns: Nothing.
        """

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
            print("Error: Invalid RData object: required columns not found.")
            self.vtable = pd.DataFrame()
            return

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
            print("Error: Invalid RData object: WT variant missing or ambiguous.")
            self.vtable = pd.DataFrame()
            return

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
        features = [],
        ensemble = False):
        """
        Initialize a MochiData object.

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
        :param features: list of feature names to filter (default:[] i.e. all features retained).
        :param ensemble: Ensemble encode features. (default:False).
        :returns: MochiData object.
        """
        #Save attributes
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
        self.ensemble = ensemble
        #Initialize attributes
        self.fdata = None
        self.additive_trait_names = None
        self.phenotype_names = None
        self.fitness = None
        self.phenotypes = None
        self.X = None
        self.Xoh = None
        self.Xohi = None
        self.cvgroups = None
        self.coefficients = None

        #Check and save model design
        self.model_design = self.check_model_design(self.model_design)
        if len(self.model_design)==0:
            return
        #Load all datasets
        print("Loading fitness data")
        filepath_list = list(self.model_design['file'])
        fdatalist = [FitnessData(
            file_path = filepath_list[i], 
            name = str(i+1),
            order_subset = order_subset,
            downsample_observations = downsample_observations,
            seed = self.seed) for i in range(len(filepath_list))]        
        #Merge all datasets
        self.fdata = self.merge_datasets(fdatalist)
        if self.fdata==None:
            return
        #Fitness
        self.fitness = self.fdata.vtable.loc[:,['fitness', 'sigma']]
        #Phenotypes
        self.phenotypes = self.one_hot_encode_phenotypes()
        #Fitness weights
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
            max_order = max_interaction_order,
            min_observed = min_observed,
            features = features,
            downsample_interactions = downsample_interactions,
            seed = self.seed)
        #Split into training, validation and test sets
        print("Defining cross-validation groups")
        self.define_cross_validation_groups(
            k_folds = k_folds,
            seed = self.seed,
            validation_factor = validation_factor, 
            holdout_minobs = holdout_minobs, 
            holdout_orders = holdout_orders, 
            holdout_WT = holdout_WT)
        #Ensemble encode features
        if ensemble:
            print("Ensemble encoding features")
            self.Xohi = self.ensemble_encode_features()
        print("Done!")

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
            return pd.DataFrame()
        #Add unity weights if not supplied
        if 'weight' not in model_design.keys():
            model_design['weight'] = 1
        #Check if all keys present
        if sum([i not in model_design.keys() for i in model_design_keys])!=0:
            print("Error: Model design missing required keys.")
            return pd.DataFrame()
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
            return pd.DataFrame()
        #Translate phenotypes to integers
        self.phenotype_names = all_phenotypes_unique
        model_design['phenotype'] = range(1, len(model_design)+1)
        #Check files not duplicated
        all_files = list(model_design['file'])
        if len(all_files)!=len(list(set(all_files))):
            print("Error: Duplicated input files.")
            return pd.DataFrame()
        return model_design

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
            return
        #Check if same WT and sequence type
        if len(set([i.wildtype for i in data_list]))==1 & len(set([i.sequenceType for i in data_list]))==1:
            fdata = copy.deepcopy(data_list[0])
            fdata.vtable = pd.concat([i.vtable for i in data_list])
            fdata.vtable.reset_index(drop = True, inplace = True)
            return fdata
        else:
            print("Error: Fitness datasets cannot be merged: WT variants do not match.")
            return

    def one_hot_encode_phenotypes(self):
        """
        1-hot encode phenotypes.

        :returns: A DataFrame with 1-hot phenotypes.
        """
        all_phenotypes = [str(i) for i in list(self.model_design['phenotype'])]
        phenotypes_df = pd.DataFrame()
        for i in all_phenotypes:
            phenotypes_df['phenotype_'+i] = (self.fdata.vtable['phenotype']==i).astype(int)
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
            dtype = int)
        enc.fit(self.X)
        one_hot_names = [self.fdata.wildtype_split[int(i[1:-2])]+str(int(i[1:-2])+1)+i[-1] for i in enc.get_feature_names_out()]
        one_hot_df = pd.DataFrame(enc.transform(self.X).toarray(), columns = one_hot_names)
        if include_WT:
            one_hot_df = pd.concat([pd.DataFrame({'WT': [1]*len(one_hot_df)}), one_hot_df], axis=1)
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
                dict1[k] = list(set(dict1[k] + dict2[k]))
            else:
                dict1[k] = dict2[k]
        return(dict1)

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
        return((all_features, int_order_dict))

    def one_hot_encode_interactions(
        self, 
        max_order = 2,
        exclude = [0],
        max_cells = 1e9,
        min_observed = 2,
        features = [],
        downsample_interactions = None,
        seed = 1):
        """
        Add interaction terms to 1-hot encoding DataFrame.

        :param max_order: Maximum interaction order (default:2).
        :param exclude: Column indices to exclude from interactions (default:0 i.e. WT).
        :param max_cells: Maximum matrix cells permitted (default:1billion).
        :param min_observed: Minimum number of observations required to include interaction term (default:2).
        :param features: list of feature names to filter (default:[] i.e. all  features retained).
        :param downsample_interactions: number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (optional).
        :param seed: Random seed for downsampling interactions (default:1).
        :returns: Nothing.
        """

        #Check downsample_interactions argument valid
        if downsample_interactions!=None:
            if type(downsample_interactions) == float:
                #Downsample observations by proportion
                if downsample_interactions >= 1 or downsample_interactions <= 0:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
            elif type(downsample_interactions) == int:
                #Downsample observations by number
                if downsample_interactions < 1:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
            elif type(downsample_interactions) == str:
                try:
                    downsample_interactions = {(i+2):int(d) for i,d in enumerate(str(downsample_interactions).split(","))}
                except:
                    print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")
            else:
                print("Error: downsample_interactions argument invalid: only proportions in range (0,1) or positive integer numbers allowed.")

        #Check if no interactions to add
        if max_order<2:
            self.Xohi = copy.deepcopy(self.Xoh)
            #Filter features
            if features!=[]:
                print("Filtering features")
                self.filter_features(features = features)
            return
        #Columns to consider
        int_columns = list(self.Xoh.columns)
        int_columns = [int_columns[i] for i in range(len(int_columns)) if i not in exclude]

        #Get all theoretical interactions
        all_features,int_order_dict = self.get_theoretical_interactions(max_order = max_order)
        print("... Total theoretical features (order:count): "+", ".join([str(i)+":"+str(int_order_dict[i]) for i in sorted(int_order_dict.keys())]))
        #Flatten
        all_features_flat = list(itertools.chain(*list(all_features.values())))

        #Check if all interaction features exist (i.e. with mutation order>1)
        if len([i for i in features if (i not in all_features_flat) and (len(i.split('_'))>1)]) != 0:
            print(f"Error: Invalid feature names.")
            return

        #Select interactions
        int_list = []
        int_order_dict_retained = {}
        int_list_names = []
        #No shuffle if not downsampling
        if downsample_interactions == None:
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
                    int_col = (self.Xoh.loc[:,c_split].sum(axis = 1)==len(c_split)).astype(int)
                    #Check if minimum number of observations satisfied
                    if sum(int_col) >= min_observed:
                        int_list += [int_col]
                        int_list_names += [c]
                        if len(c_split) not in int_order_dict_retained.keys():
                            int_order_dict_retained[len(c_split)] = 1
                        else:
                            int_order_dict_retained[len(c_split)] += 1
                    # else:
                    #     if len(c_split)==3 and sum(int_col)==1:
                    #         print(c)
                    #Check memory footprint
                    if len(int_list)*len(self.Xoh) > max_cells:
                        print(f"Error: Too many interaction terms: number of feature matrix cells >{max_cells:>.0e}")
                        return
                    #Check if sufficient features obtained
                    if type(downsample_interactions) == float:
                        if len(int_list) == int(len(all_features_flat)*downsample_interactions):
                            break
                    elif type(downsample_interactions) == int:
                        if len(int_list) == downsample_interactions:
                            break
                    elif type(downsample_interactions) == dict:
                        if len(c_split) in int_order_dict_retained.keys():
                            if int_order_dict_retained[len(c_split)] > downsample_interactions[len(c_split)] and downsample_interactions[len(c_split)]!=(-1):
                                int_list.pop()
                                int_list_names.pop()
                                int_order_dict_retained[len(c_split)] -= 1
                                break
                            elif int_order_dict_retained == downsample_interactions:
                                break

        print("... Total retained features (order:count): "+", ".join([str(i)+":"+str(int_order_dict_retained[i])+" ("+str(round(int_order_dict_retained[i]/int_order_dict[i]*100, 1))+"%)" for i in sorted(int_order_dict_retained.keys())]))

        #Concatenate into dataframe
        if len(int_list)>0:
            self.Xohi = pd.concat(int_list, axis=1)
            self.Xohi.columns = int_list_names
            #Reorder
            self.Xohi = self.Xohi.loc[:,[i for i in all_features_flat if i in self.Xohi.columns]]
            self.Xohi = pd.concat([self.Xoh, self.Xohi], axis=1)
        else:
            self.Xohi = copy.deepcopy(self.Xoh)

        #Filter features
        if features!=[]:
            print("Filtering features")
            self.filter_features(
                features = features)

    def filter_features(
        self, 
        features):
        """
        Filter features by name.

        :param features: list of feature names to filter (default:[] i.e. all features retained).
        :returns: Nothing.
        """
        #Check if all features exist 
        if len([i for i in features if i not in self.Xohi.columns]) != 0:
            print(f"Error: Invalid feature names.")
            return
        #Filter features
        features_order = [i for i in self.Xohi.columns if i in features]
        self.Xohi = self.Xohi.loc[:,features_order]

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
        #Initialise V matrix
        V = np.array([[0.0]*len(str_coef)]*len(str_geno))
        #Fill matrix
        for i in range(len(str_geno)):
            factor1 = int(np.prod([c for a,b,c in zip(str_coef_[i], str_geno[i], num_states) if ord(a) != ord(b)]))
            factor2 = sum([1 for a,b in zip(str_coef_[i], str_geno[i]) if ord(a) == ord(b)])
            if invert:
                V[i,i] = factor1 * np.power(-1, factor2)
            else:
                V[i,i] = 1/(factor1 * np.power(-1, factor2))
        return(V)

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
        #Initialise sequence string
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
        ceof_list = [self.coefficient_to_sequence(coef, len(self.fdata.wildtype)) for coef in self.Xohi.columns]
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
        return pd.DataFrame(np.matmul(hmat_inv, vmat_inv), columns = self.Xohi.columns)

    def define_cross_validation_groups(
        self, 
        k_folds = 10,
        seed = 1,
        validation_factor = 2,
        holdout_minobs = 0,
        holdout_orders = [],
        holdout_WT = False):
        """
        Define cross-validation groups.

        :param k_folds: Number of cross-validation folds (default:10).
        :param seed: Random seed for defining cross-validation groups (default:1).
        :param validation_factor: Relative size of validation set with respect to test set (default:2).
        :param holdout_minobs: Minimum number of observations of additive trait weights to be held out (default:0).
        :param holdout_orders: list of mutation orders corresponding to retained variants (default:[] i.e. variants of all mutation orders can be held out).
        :param holdout_WT: list of mutation orders corresponding to retained variants (default:False).
        :returns: Nothing.
        """
        #Default: hold out all orders
        if holdout_orders == []:
            holdout_orders = list(set(self.fdata.vtable[self.fdata.mutationOrderCol]))

        #Variants that can be held out (determined separately for each additive trait)
        all_traits_unique = list(set([item for sublist in list(self.model_design['trait']) for item in sublist]))
        #Initialize holdout status (all variants can be held out)
        self.cvgroups = pd.DataFrame({
            "holdout" : np.array([1]*len(self.Xohi))
            })
        #Consider each additive trait separately
        for t in all_traits_unique:
            #Phenotypes reporting on this trait
            relevant_phenotype_columns = ["phenotype_"+str(self.model_design.loc[i,'phenotype']) for i in range(len(self.model_design)) if t in self.model_design.loc[i,'trait']]
            #Number of observations per coefficient
            Xohp_colsum = pd.DataFrame(self.Xohi.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,:].sum(axis=0))
            #Indices of coefficients that do not meet required threshold
            Xohp_noholdout = list(Xohp_colsum.loc[Xohp_colsum.iloc[:,0]<holdout_minobs,:].index)
            #Observations of coefficients that do not meet the required threshold
            Xohp_noholdout_rowsum = np.array(self.Xohi.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,Xohp_noholdout].sum(axis=1))
            #WT variants for these phenotypes
            Xohp_WT = np.array(self.fdata.vtable.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'WT'])
            #Mutation orders for these phenotypes
            Xohp_mutationOrder = np.array(self.fdata.vtable.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,self.fdata.mutationOrderCol])
            #Current holdout status
            current_status = list(self.cvgroups.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'holdout'])
            #Holdout status for this additive trait
            noholdout_minobs = [Xohp_noholdout_rowsum[i]!=0 for i in range(len(Xohp_noholdout_rowsum))]
            noholdout_orders = [Xohp_mutationOrder[i] not in holdout_orders for i in range(len(Xohp_noholdout_rowsum))]
            noholdout_WT = [((Xohp_WT[i]==True) & (holdout_WT==False)) for i in range(len(Xohp_noholdout_rowsum))]
            noholdout = [(noholdout_minobs[i] | noholdout_orders[i] | noholdout_WT[i]) for i in range(len(Xohp_noholdout_rowsum))]
            #New holdout status
            self.cvgroups.loc[self.phenotypes.loc[:,relevant_phenotype_columns].sum(axis=1)==1,'holdout'] = np.asarray([int((current_status[i]==1) & (noholdout[i]==False)) for i in range(len(Xohp_noholdout_rowsum))])
        
        #Total number of variants that can be held out
        n_holdout = sum(self.cvgroups.holdout)

        #Hold out folds
        holdout_fold = (list(range(1, k_folds+1))*(int(n_holdout/k_folds)+1))[:n_holdout]
        random.seed(seed)
        random.shuffle(holdout_fold)

        #Add to cvgroups DataFrame
        self.cvgroups['fold'] = None
        self.cvgroups.loc[self.cvgroups.holdout==1,'fold'] = holdout_fold

        #Add cross validation groups
        for i in range(k_folds):
            self.cvgroups['fold_'+str(i+1)] = "training"
            self.cvgroups.loc[self.cvgroups['fold']==(i+1),'fold_'+str(i+1)] = "test"
            val_groups = [(j%k_folds)+1 for j in list(range(i+1, i+1+validation_factor))]
            for j in val_groups:
                self.cvgroups.loc[self.cvgroups['fold']==j,'fold_'+str(i+1)] = "validation"

        #Cefficients that can be fit
        self.coefficients = {}
        for p in self.phenotypes.columns:
            # self.coefficients[p] = pd.DataFrame({'id': list(self.Xohi.columns)})
            self.coefficients[p] = pd.DataFrame()
            for i in range(k_folds):
                Xohp_colsum = pd.DataFrame(self.Xohi.loc[(self.phenotypes[p]==1) & (self.cvgroups['fold_'+str(i+1)]=="training"),:].sum(axis=0))
                self.coefficients[p]['fold_'+str(i+1)] = np.asarray([int(j!=0) for j in list(Xohp_colsum.iloc[:,0])])

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
        if self.fdata == None:
            return None
        #Loop over training, validation and test sets
        fold_name = "fold_"+str(fold)
        data_dict = {}
        for g in list(set(self.cvgroups[fold_name])):
            #Shuffle indices for training data
            sind = list(range(sum(self.cvgroups[fold_name]==g)))
            if g=="training":
                random.seed(seed)
                random.shuffle(sind)
            data_dict[g] = {}
            #Select tensor
            data_dict[g]['select'] = pd.DataFrame(self.phenotypes.loc[self.cvgroups[fold_name]==g,:])
            data_dict[g]['select'].reset_index(drop = True, inplace = True)
            data_dict[g]['select'] = torch.tensor(np.asarray(data_dict[g]['select'].loc[sind,:]), dtype=torch.float32)
            #Mask tensor
            data_dict[g]['mask'] = torch.tensor(np.asarray(pd.concat([self.coefficients["phenotype_"+str(i+1)].loc[:,fold_name] for i in range(len(self.coefficients))], axis = 1)), dtype=torch.float32)
            data_dict[g]['mask'] = torch.transpose(data_dict[g]['mask'], 0, 1)
            #Feature tensor
            data_dict[g]['X'] = pd.DataFrame(self.Xohi.loc[self.cvgroups[fold_name]==g,:])
            data_dict[g]['X'].reset_index(drop = True, inplace = True)
            data_dict[g]['X'] = torch.tensor(np.asarray(data_dict[g]['X'].loc[sind,:]), dtype=torch.float32)
            #Target tensor
            data_dict[g]['y'] = pd.DataFrame(self.fitness.loc[self.cvgroups[fold_name]==g,'fitness'])
            #Add random noise to training target data proportional to target error (if specified)
            if g=="training" and training_resample:
                np.random.seed(seed)
                data_dict[g]['y']['noise'] = [np.random.normal(scale = i) for i in list(self.fitness.loc[self.cvgroups[fold_name]==g,'sigma'])]
                data_dict[g]['y'] = pd.DataFrame(data_dict[g]['y'].sum(axis = 1))
            data_dict[g]['y'].reset_index(drop = True, inplace = True)
            data_dict[g]['y'] = torch.reshape(torch.tensor(np.asarray(data_dict[g]['y'].loc[sind,:]), dtype=torch.float32), (-1, 1))
            #Target weight tensor
            data_dict[g]['y_wt'] = pd.DataFrame(self.fitness.loc[self.cvgroups[fold_name]==g,'weight'])
            data_dict[g]['y_wt'].reset_index(drop = True, inplace = True)
            data_dict[g]['y_wt'] = torch.reshape(torch.tensor(np.asarray(data_dict[g]['y_wt'].loc[sind,:]), dtype=torch.float32), (-1, 1))
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
        #Select tensor
        data_dict['select'] = pd.DataFrame(self.phenotypes.iloc[indices,:])
        data_dict['select'].reset_index(drop = True, inplace = True)
        data_dict['select'] = torch.tensor(np.asarray(data_dict['select']), dtype=torch.float32)
        #Feature tensor
        data_dict['X'] = pd.DataFrame(self.Xohi.iloc[indices,:])
        data_dict['X'].reset_index(drop = True, inplace = True)
        data_dict['X'] = torch.tensor(np.asarray(data_dict['X']), dtype=torch.float32)
        #Target tensor
        data_dict['y'] = pd.DataFrame(self.fitness.iloc[indices,:]['fitness'])
        data_dict['y'].reset_index(drop = True, inplace = True)
        data_dict['y'] = torch.reshape(torch.tensor(np.asarray(data_dict['y']), dtype=torch.float32), (-1, 1))
        return data_dict

    def __len__(self):
        """
        Length of object.
        :returns: Number of variants (length of fdata).
        """
        return len(self.fdata)

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
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

