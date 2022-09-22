
"""
MoCHI project module
"""

import os
import torch
import pathlib
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
        features = [], #list, Path or str
        ensemble = False,
        #MochiTask arguments
        batch_size = 512,
        learn_rate = 0.05,
        num_epochs = 300,
        num_epochs_grid = 100,
        l1_regularization_factor = 0,
        l2_regularization_factor = 0,
        scheduler_gamma = 0.98):
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
        :param features: list (or path to file) of feature names to filter (default:[] i.e. all features retained).
        :param ensemble: Ensemble encode features. (default:False).
        :param batch_size: Minibatch size (default:512).
        :param learn_rate: Learning rate (default:0.05).
        :param num_epochs: Number of training epochs (default:300).
        :param num_epochs_grid: Number of grid search epochs (default:100).
        :param l1_regularization_factor: Lambda factor applied to L1 norm (default:0).
        :param l2_regularization_factor: Lambda factor applied to L2 norm (default:0).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
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
        #MochiTask arguments
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.num_epochs = num_epochs
        self.num_epochs_grid = num_epochs_grid
        self.l1_regularization_factor = l1_regularization_factor
        self.l2_regularization_factor = l2_regularization_factor
        self.scheduler_gamma = scheduler_gamma

        #Load model_design from file if necessary
        model_design = self.load_model_design(model_design)
        if type(model_design) != pd.DataFrame:
            print("Error: Invalid model_design file path: does not exist.")
            return

        #Load features from file if necessary
        features = self.load_features(features)
        if type(features) != list:
            print("Error: Invalid features file path: does not exist.")
            return

        #Initialise remaining attributes
        self.directory = directory
        self.tasks = {}
        if not model_design.empty:
            #Create project directory
            try:
                os.mkdir(self.directory)
            except FileExistsError:
                print("Warning: Project directory already exists.")

            #Run CV task
            for seedi in [int(i) for i in str(seed).split(",")]:
                #Check if task directory exists
                if os.path.exists(os.path.join(self.directory, 'task_'+str(seedi))):
                    print("Error: Task directory already exists.")
                    break
                #Run
                self.tasks[seedi] = self.run_cv_task(
                    mochi_data_args = {
                        'model_design' : model_design,
                        'order_subset' : order_subset,
                        'max_interaction_order' : max_interaction_order,
                        'downsample_observations' : downsample_observations,
                        'downsample_interactions' : downsample_interactions,
                        'k_folds' : k_folds,
                        'seed' : seedi,
                        'validation_factor' : validation_factor, 
                        'holdout_minobs' : holdout_minobs, 
                        'holdout_orders' : holdout_orders, 
                        'holdout_WT' : holdout_WT,
                        'features' : features,
                        'ensemble' : ensemble},
                    mochi_task_args = {
                        'directory' : os.path.join(self.directory, 'task_'+str(seedi)),
                        'batch_size' : batch_size,
                        'learn_rate' : learn_rate,
                        'num_epochs' : num_epochs,
                        'num_epochs_grid' : num_epochs_grid,
                        'l1_regularization_factor' : l1_regularization_factor,
                        'l2_regularization_factor' : l2_regularization_factor},
                    RT = RT,
                    seq_position_offset = seq_position_offset)

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
            for seedi in [int(i) for i in str(seed).split(",")]:
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
            return(input_obj)
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            return(None)
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            return(None)
        #Return model_design
        return(pd.read_csv(input_obj, sep = "\t", index_col = False))

    def load_features(
        self,
        input_obj):
        """
        Load features from file.

        :param input_obj: Input list, string path or Path object (required).
        :returns: A features list.
        """ 
        #Object already a list
        if type(input_obj) == list:
            return(input_obj)
        #Object a string path
        elif type(input_obj) == str:
            input_obj = pathlib.Path(input_obj)
        #Object not a path
        elif type(input_obj) != pathlib.PosixPath:
            return(None)
        #Object does not exist or not a file
        if not (input_obj.exists() and input_obj.is_file()):
            return(None)
        #Return features list
        return(list(pd.read_csv(input_obj, sep = "\t", engine='python', header = None)[0]))

    def run_cv_task(
        self,
        mochi_data_args,
        mochi_task_args,
        RT = None,
        seq_position_offset = 0):
        """
        Run MochiTask and save to disk.

        :param mochi_data_args: Dictionary of arguments for MochiData constructor (required).
        :param mochi_task_args: Dictionary of arguments for MochiTask constructor (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :returns: MochiTask object.
        """ 

        #Load mochi data
        mochi_data = MochiData(**mochi_data_args)

        #Create mochi project
        mochi_task = MochiTask(
            data = mochi_data,
            **mochi_task_args)

        #Grid search
        mochi_task.grid_search(seed = mochi_data_args['seed'])

        #Fit model using best hyperparameters
        for i in range(mochi_data_args['k_folds']):
            mochi_task.fit_best(fold = i+1, seed = mochi_data_args['seed'])
            
        #Save all models
        mochi_task.save(overwrite = True)

        #Get model weights
        energies = mochi_task.get_additive_trait_weights(
            seq_position_offset = seq_position_offset,
            RT = RT)

        #Generate project report
        mochi_report = MochiReport(
            task = mochi_task,
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
        return(mochi_task)

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

        #Task to use for prediction
        if not task_id in self.tasks.keys():
            print("Error: Invalid task identifier.")
            return()
        mochi_task = self.tasks[task_id]

        #Model design
        model_design = copy.deepcopy(mochi_task.data.model_design)
        if not type(input_obj) in [pathlib.PosixPath, str]:
            print("Error: Invalid string path or Path object 'input_obj'.")
            return()
        #Set file
        model_design.file = str(input_obj)
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
            features = list(mochi_task.data.Xohi.columns),
            ensemble = mochi_task.data.ensemble)

        #Predictions on all variants for all models
        result_df = mochi_task.predict_all(
            data = mochi_data)
        #Remove Fold column
        result_df = result_df[[i for i in result_df.columns if i!="Fold"]]

        #Output predictions directory
        directory = os.path.join(mochi_task.directory, 'predictions')

        #Save
        result_df.to_csv(os.path.join(directory, output_filename), sep = "\t", index = False)




