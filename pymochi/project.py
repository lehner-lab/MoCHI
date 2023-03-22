
"""
MoCHI project module
"""

import os
import torch
import pathlib
import numpy as np
import copy
from pathlib import Path
from pymochi.data import *
from pymochi.models import *
from pymochi.report import *

class MochiProject():
    """
    A class for the management of an inference project/campaign (one or more inference tasks).
    """
    def __init__(
        self, 
        directory,
        seed = 1,
        RT = None,
        seq_position_offset = 0,        
        #MochiData arguments
        model_design = pd.DataFrame(), #pandas DataFrame, Path or str
        order_subset = None,
        downsample_observations = None,
        downsample_interactions = None,
        max_interaction_order = 1,
        min_observed = 2,
        k_folds = 10,
        validation_factor = 2, 
        holdout_minobs = 0, 
        holdout_orders = [], 
        holdout_WT = False,
        features = {}, #list, dict, Path or str
        ensemble = False,
        custom_transformations = None, #Path or str
        #MochiTask arguments
        batch_size = "512,1024,2048",
        learn_rate = 0.05,
        num_epochs = 1000,
        num_epochs_grid = 100,
        l1_regularization_factor = 0,
        l2_regularization_factor = 0.000001,
        scheduler_gamma = 0.98,
        init_weights_directory = None,
        init_weights_task_id = 1,
        fix_weights = {},
        sparse_method = None):
        """
        Initialize a MochiProject object.

        :param directory: Path to project directory where tasks should be stored (required).
        :param seed: Random seed for downsampling (observations and interactions) and defining cross-validation groups (default:1).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param model_design: Model design DataFrame (or path to file) with phenotype, transformation, trait and file columns (required unless 'directory' contains saved tasks).
        :param order_subset: list of mutation orders corresponding to retained variants (optional).
        :param downsample_observations: number (if integer) or proportion (if float) of observations to retain including WT (optional).
        :param downsample_interactions: number (if integer) or proportion (if float) of interaction terms to retain (optional).
        :param max_interaction_order: Maximum interaction order (default:1).
        :param min_observed: Minimum number of observations required to include interaction term (default:2).
        :param k_folds: Numbef of cross-validation folds (default:10).
        :param validation_factor: Relative size of validation set with respect to test set (default:2).
        :param holdout_minobs: Minimum number of observations of additive trait weights to be held out (default:0).
        :param holdout_orders: list of mutation orders corresponding to retained variants (default:[] i.e. variants of all mutation orders can be held out).
        :param holdout_WT: Whether or not to WT variant can be held out (default:False).
        :param features: List, dictionary (or path to file) of trait-specific feature names to fit (default:empty dict i.e. all features fit).
        :param ensemble: Ensemble encode features. (default:False).
        :param custom_transformations: Path to custom transformations file (optional).
        :param batch_size: Minibatch size (default:512).
        :param learn_rate: Learning rate (default:0.05).
        :param num_epochs: Number of training epochs (default:300).
        :param num_epochs_grid: Number of grid search epochs (default:100).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (default:0).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (default:0.000001).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :param init_weights_directory: Path to project directory for model weight initialization (optional).
        :param init_weights_task_id: Task identifier to use for model weight initialization (default:1).
        :param fix_weights: Dictionary (or path to file) of layer names to fix weights (default:empty dict i.e. no layers fixed).
        :param sparse_method: Sparse model inference method: one of 'sig_highestorder_step' (optional).
        :returns: MochiProject object.
        """ 

        #Save attributes
        self.directory = directory
        self.seed = seed
        self.RT = RT
        self.seq_position_offset = seq_position_offset      
        #MochiData arguments
        self.model_design = model_design
        self.order_subset = order_subset
        self.downsample_observations = downsample_observations
        self.downsample_interactions = downsample_interactions
        self.max_interaction_order = max_interaction_order
        self.min_observed = min_observed
        self.k_folds = k_folds
        self.validation_factor = validation_factor
        self.holdout_minobs = holdout_minobs
        self.holdout_orders = holdout_orders
        self.holdout_WT = holdout_WT
        self.features = features
        self.ensemble = ensemble
        self.custom_transformations = custom_transformations
        #MochiTask arguments
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.num_epochs_grid = num_epochs_grid
        self.l1_regularization_factor = l1_regularization_factor
        self.l2_regularization_factor = l2_regularization_factor
        self.scheduler_gamma = scheduler_gamma
        self.init_weights_directory = init_weights_directory
        self.init_weights_task_id = init_weights_task_id
        self.fix_weights = fix_weights
        self.sparse_method = sparse_method

        #Load model_design from file if necessary
        self.model_design = self.load_model_design(self.model_design)
        if type(self.model_design) != pd.DataFrame:
            print("Error: Invalid model_design file path: does not exist.")
            return

        #Load features from file if necessary
        self.features = self.load_features(self.features)
        if type(self.features) != dict:
            print("Error: Invalid features file path: does not exist.")
            return

        #Load project and task for model weight initialization if necessary
        init_weights = None
        if not self.init_weights_directory is None:
            init_weights = MochiProject(
                directory = self.init_weights_directory).tasks[self.init_weights_task_id]

        #Load layer names to fix from file if necessary
        self.fix_weights = self.load_fix_weights(self.fix_weights)
        if type(self.fix_weights) != dict:
            print("Error: Could not load fix_weights file.")
            return

        self.tasks = {}
        if not self.model_design.empty:
            #Create project directory
            try:
                os.mkdir(self.directory)
            except FileExistsError:
                print("Warning: Project directory already exists.")

            if sparse_method is None:
                #Run CV tasks for all seeds
                self.run_cv_tasks(init_weights = init_weights)
            elif sparse_method == "sig_highestorder_step":
                #Check that only one starting seed supplied
                if len(str(self.seed).split(",")) != 1:
                    print("Error: Sparse model inference method 'sig_highestorder_step' cannot be run with multiple starting seeds.")
                    return                
                #Run sparse model inference method 'sig_highestorder_step'
                self.run_sparse_sig_highestorder_step(init_weights = init_weights)                
            else:
                print("Error: Invalid sparse model inference method.")
                return
        else:
            #Check if model directory exists
            if not os.path.exists(self.directory):
                print("Error: Project directory does not exist.")
                return

            #Reformat task folders (for backwards compatibility)
            #Check if no tasks saved
            dirlist = os.listdir(self.directory)
            if len([i for i in dirlist if i.startswith("task_")])==0:
                #Check if legacy project
                if "saved_models" in dirlist:
                    print("Reformating legacy task.")
                    #Load legacy project task
                    legacy_task = MochiTask(directory = self.directory)
                    #Add seed to data and model metadata
                    legacy_task.data.seed = 1
                    for mi in legacy_task.models:
                        mi.metadata.seed = 1
                    #Create task directory
                    task_directory = os.path.join(self.directory, "task_1")
                    os.mkdir(task_directory)
                    for fi in dirlist:
                        if not fi.startswith("."):
                            os.rename(os.path.join(self.directory, fi), os.path.join(task_directory, fi))
                    #Re-save data and models
                    legacy_task.directory = task_directory
                    legacy_task.save(overwrite = True)

            #Load saved tasks
            for seedi in [int(i) for i in str(self.seed).split(",")]:
                task_directory = os.path.join(self.directory, "task_"+str(seedi))
                print("Loading task "+str(seedi))
                #Check if task directory exists
                if not os.path.exists(task_directory):
                    print("Error: Task directory does not exist.")
                else:                
                    self.tasks[seedi] = MochiTask(directory = task_directory)

    def load_model_design(
        self,
        input_obj):
        """
        Load model design from file.

        :param input_obj: Input DataFrame, string path or Path object (required).
        :returns: A model design DataFrame.
        """ 
        #Object already a DataFrame
        if type(input_obj) == pd.DataFrame:
            return input_obj
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            print("Error: Invalid fix_weights file path: does not exist.")
            return
            return None
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            return None
        #Return model_design
        return pd.read_csv(input_obj, sep = "\t", index_col = False)

    def detect_delimiter(
        self,
        input_path):
        """
        Detect file delimiter.

        :param input_path: Path object (required).
        :returns: delimiter string.
        """ 
        reader = pd.read_csv(input_path, sep = None, iterator = True, engine='python')
        return reader._engine.data.dialect.delimiter

    def load_features(
        self,
        input_obj):
        """
        Load features from file.

        :param input_obj: Input list, dict, string path or Path object (required).
        :returns: A features dict.
        """ 
        #Object already a list
        if type(input_obj) == list:
            return {None: input_obj}
        #Object already a dict
        if type(input_obj) == dict:
            return input_obj
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            return None
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            return None
        #Detect delimiter
        delimiter = self.detect_delimiter(input_obj)
        if delimiter not in [",", ";", " ", "\t"]:
            delimiter = "\t"
        #Return features dict
        return pd.read_csv(input_obj, sep = delimiter, engine='python').to_dict("list")

    def load_fix_weights(
        self,
        input_obj):
        """
        Load fixed weights from file.

        :param input_obj: Input dict, string path or Path object (required).
        :returns: A dict.
        """ 
        #Object already a dict
        if type(input_obj) == dict:
            return input_obj
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            print("Error: Invalid features file path: does not exist.")
            return None
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            print("Error: Invalid features file path: does not exist.")
            return None
        #Read file
        fix_df = pd.read_csv(input_obj, sep = "\t", engine='python', header = None)
        #Check if entries separated by colons
        if sum([i for i in list(fix_df.iloc[:,0]) if len(i.split(':'))<2])!=0:
            print("Error: Invalid features file path: entries must be colon-separated.")
            return None
        #Construct dictionary
        fix_phenotype = [i.split(':')[1] for i in list(fix_df.iloc[:,0]) if i.split(':')[0]=='phenotype']
        fix_trait = [i.split(':')[1] for i in list(fix_df.iloc[:,0]) if i.split(':')[0]=='trait']
        fix_global = [i.split(':')[1] for i in list(fix_df.iloc[:,0]) if i.split(':')[0]=='global']
        fix_weights = {
            'phenotype' : fix_phenotype,
            'trait' : fix_trait,
            'global' : fix_global}
        return fix_weights

    def run_sparse_sig_highestorder_step(
        self,
        init_weights = None):
        """
        Run sparse model inference method 'sig_highestorder_step'.
        This method iteratively removes nominally non-significant model weights/terms/coefficients from highest to lowest order (minimum order 1)

        :param init_weights: Task to use for model weight initialization (optional).
        :returns: nothing.
        """

        #Run sparse model inference method 'sig_highestorder_step'
        taski = 1
        save_task_data = False
        l1_regularization_factori = self.l1_regularization_factor
        for orderi in range(self.max_interaction_order, -2, -1):
            #Check if task directory exists
            if os.path.exists(os.path.join(self.directory, 'task_'+str(taski))):
                print("Error: Task directory already exists.")
                break
            #First model features
            features = self.features
            #Restrict features based on previous task in the path
            if orderi < self.max_interaction_order:
                #Get additive trait weights from previous model
                at_list = self.tasks[taski-1].get_additive_trait_weights(save = False)
                #Restrict features
                features = {}
                for i in range(len(at_list)):
                    #Add mutant order
                    at_list[i]['mut_order'] = [len(str(j).split('_')) for j in list(at_list[i]['Pos'])]
                    at_list[i].loc[at_list[i]['id']=='WT','mut_order'] = 0
                    #Additive trait name
                    at_name = self.tasks[taski-1].data.additive_trait_names[i]
                    if orderi > -1:
                        features[at_name] = list(at_list[i].loc[(at_list[i]['mut_order'] <= orderi) | ((np.abs(at_list[i]['mean']) - at_list[i]['ci95']/2)>0),'id'])
                    else:
                        #Final model
                        features[at_name] = list(at_list[i]['id'])
                #Reformat features
                features = self.load_features(features)
                if type(features) != dict:
                    print("Error: Invalid features file path: does not exist.")
                    return
            #Fit final models without regularization
            if orderi <= -1:
                l1_regularization_factori = 0
                save_task_data = True
            #Run
            try:
                self.tasks[taski] = self.run_cv_task(
                    mochi_data_args = copy.deepcopy({
                        'model_design' : self.model_design,
                        'order_subset' : self.order_subset,
                        'max_interaction_order' : self.max_interaction_order,
                        'downsample_observations' : self.downsample_observations,
                        'downsample_interactions' : self.downsample_interactions,
                        'k_folds' : self.k_folds,
                        'seed' : self.seed,
                        'validation_factor' : self.validation_factor, 
                        'holdout_minobs' : self.holdout_minobs, 
                        'holdout_orders' : self.holdout_orders, 
                        'holdout_WT' : self.holdout_WT,
                        'features' : features,
                        'ensemble' : self.ensemble,
                        'custom_transformations' : self.custom_transformations}),
                    mochi_task_args = copy.deepcopy({
                        'directory' : os.path.join(self.directory, 'task_'+str(taski)),
                        'batch_size' : self.batch_size,
                        'learn_rate' : self.learn_rate,
                        'num_epochs' : self.num_epochs,
                        'num_epochs_grid' : self.num_epochs_grid,
                        'l1_regularization_factor' : l1_regularization_factori,
                        'l2_regularization_factor' : self.l2_regularization_factor,
                        'scheduler_gamma' : self.scheduler_gamma}),
                    RT = self.RT,
                    seq_position_offset = self.seq_position_offset,
                    init_weights = init_weights,
                    fix_weights = self.fix_weights,
                    save_model = save_task_data,
                    save_report = save_task_data,
                    save_weights = save_task_data)
            except ValueError:
                print("Error: Failed to create MochiTask.")
                break
            #Increment seed
            taski += 1

    def run_cv_tasks(
        self,
        init_weights = None):
        """
        Run independent CV tasks for all supplied seeds.

        :param init_weights: Task to use for model weight initialization (optional).
        :returns: nothing.
        """

        #Run CV tasks for all seeds
        for seedi in [int(i) for i in str(self.seed).split(",")]:
            #Check if task directory exists
            if os.path.exists(os.path.join(self.directory, 'task_'+str(seedi))):
                print("Error: Task directory already exists.")
                break
            #Run
            try:
                self.tasks[seedi] = self.run_cv_task(
                    mochi_data_args = copy.deepcopy({
                        'model_design' : self.model_design,
                        'order_subset' : self.order_subset,
                        'max_interaction_order' : self.max_interaction_order,
                        'downsample_observations' : self.downsample_observations,
                        'downsample_interactions' : self.downsample_interactions,
                        'k_folds' : self.k_folds,
                        'seed' : seedi,
                        'validation_factor' : self.validation_factor, 
                        'holdout_minobs' : self.holdout_minobs, 
                        'holdout_orders' : self.holdout_orders, 
                        'holdout_WT' : self.holdout_WT,
                        'features' : self.features,
                        'ensemble' : self.ensemble,
                        'custom_transformations' : self.custom_transformations}),
                    mochi_task_args = copy.deepcopy({
                        'directory' : os.path.join(self.directory, 'task_'+str(seedi)),
                        'batch_size' : self.batch_size,
                        'learn_rate' : self.learn_rate,
                        'num_epochs' : self.num_epochs,
                        'num_epochs_grid' : self.num_epochs_grid,
                        'l1_regularization_factor' : self.l1_regularization_factor,
                        'l2_regularization_factor' : self.l2_regularization_factor,
                        'scheduler_gamma' : self.scheduler_gamma}),
                    RT = self.RT,
                    seq_position_offset = self.seq_position_offset,
                    init_weights = init_weights,
                    fix_weights = self.fix_weights)
            except ValueError:
                print("Error: Failed to create MochiTask.")
                break

    def run_cv_task(
        self,
        mochi_data_args,
        mochi_task_args,
        RT = None,
        seq_position_offset = 0,
        init_weights = None,
        fix_weights = {},
        save_model = True,
        save_report = True,
        save_weights = True):
        """
        Run MochiTask and save to disk.

        :param mochi_data_args: Dictionary of arguments for MochiData constructor (required).
        :param mochi_task_args: Dictionary of arguments for MochiTask constructor (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param save_model: Whether or not to save all models (default:True).
        :param save_report: Whether or not to save task report (default:True).
        :param save_weights: Whether or not to save model weights (default:True).
        :returns: MochiTask object.
        """ 

        #Load mochi data
        mochi_data = MochiData(**mochi_data_args)

        #Create mochi project
        mochi_task = MochiTask(
            data = mochi_data,
            **mochi_task_args)

        #Grid search
        mochi_task.grid_search(
            seed = mochi_data_args['seed'],
            init_weights = init_weights,
            fix_weights = fix_weights)

        #Fit model using best hyperparameters
        for i in range(mochi_data_args['k_folds']):
            mochi_task.fit_best(
                fold = i+1, 
                seed = mochi_data_args['seed'],
                init_weights = init_weights,
                fix_weights = fix_weights)
            
        #Save all models
        if save_model:
            mochi_task.save(overwrite = True)

        #Save task report
        if save_report:
            mochi_report = MochiReport(
                task = mochi_task,
                RT = RT)

        #Save model weights
        if save_weights:
            #Get model weights
            energies = mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT)

            #Aggregate energies per sequence position
            energies_agg = mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT,
                aggregate = True,
                aggregate_absolute_value = False)

            #Aggregate absolute value of energies per sequence position
            energies_agg_abs = mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT,
                aggregate = True,
                aggregate_absolute_value = True)

        return mochi_task

    def predict(
        self, 
        input_obj,
        task_id = 1,
        RT = None,
        seq_position_offset = 0,
        order_subset = None,
        output_filename = "predicted_phenotypes_supp.txt"):
        """
        Predict phenotype for arbitrary genotypes.

        :param input_obj: Input string path or Path object (required).
        :param task_id: Task identifier to use for prediction (default:1).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param order_subset: List of mutation orders corresponding to retained variants (optional).
        :param output_filename: Filename string for saved results (default:'predicted_phenotypes_misc.txt').
        :returns: nothing.
        """ 

        ### TODO: make this work for already-existing project

        #Task to use for prediction
        if not task_id in self.tasks.keys():
            print("Error: Invalid task identifier.")
            return
        mochi_task = self.tasks[task_id]

        #Model design
        model_design = copy.deepcopy(mochi_task.data.model_design)
        if not type(input_obj) in [pathlib.PosixPath, str]:
            print("Error: Invalid string path or Path object 'input_obj'.")
            return
        #Set file
        model_design.file = input_obj.split(",")
        #Set phenotype names
        model_design.phenotype = [mochi_task.data.phenotype_names[i-1] for i in list(mochi_task.data.model_design.phenotype)]

        #Load mochi data
        mochi_data = MochiData(
            model_design = model_design,
            max_interaction_order = mochi_task.data.max_interaction_order,
            min_observed = 0,
            k_folds = mochi_task.data.k_folds,
            seed = mochi_task.data.seed,
            validation_factor = mochi_task.data.validation_factor, 
            holdout_minobs = mochi_task.data.holdout_minobs, 
            holdout_orders = mochi_task.data.holdout_orders, 
            holdout_WT = mochi_task.data.holdout_WT,
            features = {None: list(mochi_task.data.Xohi.columns)},
            ensemble = mochi_task.data.ensemble)

        #Reorder feature matrix columns
        mochi_data.Xohi = mochi_data.Xohi.loc[:,list(mochi_task.data.Xohi.columns)]
        mochi_data.feature_names = mochi_data.Xohi.columns
        #Split into training, validation and test sets
        mochi_data.define_cross_validation_groups()
        #Define coefficients to fit (for each phenotype and trait)
        mochi_data.define_coefficient_groups(
            k_folds = mochi_task.data.k_folds)
        #Ensemble encode features
        if mochi_task.data.ensemble:
            mochi_data.Xohi = mochi_data.ensemble_encode_features()

        #Predictions on all variants for all models
        result_df = mochi_task.predict_all(
            data = mochi_data,
            save = False)
        #Remove Fold column
        result_df = result_df[[i for i in result_df.columns if i!="Fold"]]

        #Output predictions directory
        directory = os.path.join(mochi_task.directory, 'predictions')

        #Save
        result_df.to_csv(os.path.join(directory, output_filename), sep = "\t", index = False)




