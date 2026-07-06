
"""
MoCHI project module
"""

import os
import pickle
import shutil
import torch
import pathlib
import numpy as np
import copy
from pathlib import Path
from pymochi.data import *
from pymochi.models import *
from pymochi.report import *

def running_in_parallel_mode():
    """Return True when invoked from the phase-split Nextflow workflow."""
    return os.environ.get("MOCHI_PARALLEL_MODE", "").lower() in {"1", "true", "yes"}

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
        training_resample = True,
        early_stopping = True,
        scheduler_gamma = 0.98,
        loss_function_name = 'WeightedL1',
        sos_architecture = [20],
        sos_outputlinear = False,
        init_weights_directory = None,
        init_weights_task_id = 1,
        fix_weights = {},
        sparse_method = None,
        auto_run = True):
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
        :param training_resample: Whether or not to add random noise to training target data proportional to target error (default:True).
        :param early_stopping: Whether or not to stop training early if validation loss not decreasing (default:True).
        :param scheduler_gamma: Multiplicative factor of learning rate decay (default:0.98).
        :param loss_function_name: Loss function name: one of 'WeightedL1', 'GaussianNLL' (default:'WeightedL1').
        :param sos_architecture: list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (default:[20]).
        :param sos_outputlinear: boolean indicating whether final sumOfSigmoids should be linear rather than sigmoidal (default:False).
        :param init_weights_directory: Path to project directory for model weight initialization (optional).
        :param init_weights_task_id: Task identifier to use for model weight initialization (default:1).
        :param fix_weights: Dictionary (or path to file) of layer names to fix weights (default:empty dict i.e. no layers fixed).
        :param sparse_method: Sparse model inference method: one of 'sig_highestorder_step' (optional).
        :param auto_run: Whether to immediately execute training when model_design is supplied (default:True).
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
        self.training_resample = training_resample
        self.early_stopping = early_stopping
        self.scheduler_gamma = scheduler_gamma
        self.loss_function_name = loss_function_name
        self.sos_architecture = sos_architecture
        self.sos_outputlinear = sos_outputlinear
        self.init_weights_directory = init_weights_directory
        self.init_weights_task_id = init_weights_task_id
        self.fix_weights = fix_weights
        self.sparse_method = sparse_method
        self.auto_run = auto_run

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
                if not running_in_parallel_mode():
                    print("Warning: Project directory already exists.")

            if not self.auto_run:
                return
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

    def get_task_directory(
        self,
        seed):
        """
        Return the canonical task directory for a seed.

        :param seed: Task seed identifier (required).
        :returns: Task directory path string.
        """
        return os.path.join(self.directory, "task_"+str(seed))

    def get_fold_directory(
        self,
        seed,
        fold):
        """
        Return the per-fold task directory for a seed/fold pair.

        :param seed: Task seed identifier (required).
        :param fold: Cross-validation fold identifier (required).
        :returns: Fold directory path string.
        """
        return os.path.join(self.get_task_directory(seed), "fold_"+str(fold))

    def get_grid_condition_directory(
        self,
        seed,
        condition_index):
        """
        Return the per-condition grid-search directory for a task seed.

        :param seed: Task seed identifier (required).
        :param condition_index: One-based grid-condition identifier (required).
        :returns: Condition directory path string.
        """
        return os.path.join(self.get_task_directory(seed), "grid_condition_"+str(condition_index))

    def list_grid_condition_directories(
        self,
        seed):
        """
        Return sorted grid-condition directories for a task seed.

        :param seed: Task seed identifier (required).
        :returns: List of condition directory path strings.
        """
        task_directory = self.get_task_directory(seed)
        if not os.path.exists(task_directory):
            return []
        condition_dirs = []
        for entry in os.listdir(task_directory):
            if not entry.startswith("grid_condition_"):
                continue
            suffix = entry.split("grid_condition_", 1)[1]
            if suffix.isdigit():
                condition_dirs.append((int(suffix), os.path.join(task_directory, entry)))
        return [path for _, path in sorted(condition_dirs)]

    def list_grid_condition_task_directories(
        self,
        seed,
        stage_index = None):
        """
        Return sorted task directories containing per-condition grid-search outputs.

        :param seed: Task seed identifier (required).
        :param stage_index: Optional sparse stage identifier (default:None).
        :returns: List of task directory path strings.
        """
        condition_task_dirs = []

        # First support the canonical in-task layout.
        canonical_condition_dirs = self.list_grid_condition_directories(seed)
        for condition_dir in canonical_condition_dirs:
            if os.path.exists(os.path.join(condition_dir, "saved_models")):
                condition_task_dirs.append(condition_dir)
                continue
            condition_task_dir = os.path.join(condition_dir, "task_"+str(seed))
            if os.path.exists(os.path.join(condition_task_dir, "saved_models")):
                condition_task_dirs.append(condition_task_dir)
        if len(condition_task_dirs) != 0:
            return condition_task_dirs

        # Fall back to the Nextflow per-condition run layout.
        run_directory = os.path.dirname(self.directory)
        project_name = os.path.basename(self.directory)
        if stage_index is None:
            grid_root = os.path.join(run_directory, "grid_search")
        else:
            grid_root = os.path.join(run_directory, "stage_"+str(stage_index), "grid_search")

        if not os.path.exists(grid_root):
            return []

        discovered_dirs = []
        for entry in os.listdir(grid_root):
            if not entry.startswith("condition_"):
                continue
            suffix = entry.split("condition_", 1)[1]
            if not suffix.isdigit():
                continue
            condition_task_dir = os.path.join(
                grid_root,
                entry,
                project_name,
                "task_"+str(seed))
            if os.path.exists(os.path.join(condition_task_dir, "saved_models")):
                discovered_dirs.append((int(suffix), condition_task_dir))
        return [path for _, path in sorted(discovered_dirs)]

    def get_sparse_stage_count(self):
        """
        Return the number of sparse stages for the configured interaction order.

        :returns: Integer sparse stage count.
        """
        return self.max_interaction_order + 2

    def get_sparse_stage_order(
        self,
        stage_index):
        """
        Return the pruning order represented by a sparse stage.

        :param stage_index: One-based sparse stage index (required).
        :returns: Integer interaction order for the stage.
        """
        return self.max_interaction_order - (stage_index - 1)

    def get_parallel_canonical_project_directory(self):
        """
        Return the canonical project directory for phase-split Nextflow runs.

        :returns: Project directory path string.
        """
        project_directory = Path(self.directory)
        for ancestor in project_directory.parents:
            if ancestor.name != "grid_search":
                continue
            run_directory = ancestor.parent
            if run_directory.name.startswith("stage_"):
                run_directory = run_directory.parent
            return str(run_directory / project_directory.name)
        return self.directory

    def get_sparse_stage_settings(
        self,
        stage_index):
        """
        Return persistence and regularization settings for one sparse stage.

        :param stage_index: One-based sparse stage index (required).
        :returns: Dictionary of sparse stage settings.
        """
        orderi = self.get_sparse_stage_order(stage_index)
        return {
            'order' : orderi,
            'l1_regularization_factor' : 0 if orderi <= -1 else self.l1_regularization_factor,
            'save_model' : True,
            'save_report' : orderi <= -1,
            'save_weights' : stage_index == 1 or orderi <= -1}

    def build_sparse_stage_inputs(
        self,
        stage_index):
        """
        Build MochiData and MochiTask constructor arguments for one sparse stage.

        :param stage_index: One-based sparse stage index (required).
        :returns: Tuple of MochiData args dict, MochiTask args dict and stage settings dict.
        """
        stage_count = self.get_sparse_stage_count()
        if stage_index < 1 or stage_index > stage_count:
            print("Error: Invalid sparse stage index.")
            raise ValueError

        stage_settings = self.get_sparse_stage_settings(stage_index)
        features = self.features
        if stage_index > 1:
            prev_task_directory = self.get_task_directory(stage_index - 1)
            if not os.path.exists(os.path.join(prev_task_directory, "saved_models")) and running_in_parallel_mode():
                prev_task_directory = os.path.join(
                    self.get_parallel_canonical_project_directory(),
                    "task_"+str(stage_index - 1))
            prev_task = MochiTask(directory = prev_task_directory)
            at_list = prev_task.get_additive_trait_weights(save = False)
            features = {}
            for i in range(len(at_list)):
                at_list[i]['mut_order'] = [len(str(j).split('_')) for j in list(at_list[i]['Pos'])]
                at_list[i].loc[at_list[i]['id']=='WT','mut_order'] = 0
                at_name = prev_task.data.additive_trait_names[i]
                if stage_settings['order'] > -1:
                    features[at_name] = list(at_list[i].loc[
                        (at_list[i]['mut_order'] <= stage_settings['order']) |
                        ((np.abs(at_list[i]['mean']) - at_list[i]['ci95']/2)>0),
                        'id'])
                else:
                    features[at_name] = list(at_list[i]['id'])
            features = self.load_features(features)
            if type(features) != dict:
                print("Error: Invalid features file path: does not exist.")
                raise ValueError

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
            'custom_transformations' : self.custom_transformations})
        mochi_task_args = copy.deepcopy({
            'directory' : self.get_task_directory(stage_index),
            'batch_size' : self.batch_size,
            'learn_rate' : self.learn_rate,
            'num_epochs' : self.num_epochs,
            'num_epochs_grid' : self.num_epochs_grid,
            'l1_regularization_factor' : stage_settings['l1_regularization_factor'],
            'l2_regularization_factor' : self.l2_regularization_factor,
            'training_resample' : self.training_resample,
            'early_stopping' : self.early_stopping,
            'scheduler_gamma' : self.scheduler_gamma,
            'loss_function_name' : self.loss_function_name,
            'sos_architecture' : self.sos_architecture,
            'sos_outputlinear' : self.sos_outputlinear})
        return mochi_data_args, mochi_task_args, stage_settings

    def build_cv_task_inputs(
        self,
        seed):
        """
        Build MochiData and MochiTask constructor arguments for one seed.

        :param seed: Task seed identifier (required).
        :returns: Tuple of MochiData args dict and MochiTask args dict.
        """
        mochi_data_args = copy.deepcopy({
            'model_design' : self.model_design,
            'order_subset' : self.order_subset,
            'max_interaction_order' : self.max_interaction_order,
            'downsample_observations' : self.downsample_observations,
            'downsample_interactions' : self.downsample_interactions,
            'k_folds' : self.k_folds,
            'seed' : seed,
            'validation_factor' : self.validation_factor,
            'holdout_minobs' : self.holdout_minobs,
            'holdout_orders' : self.holdout_orders,
            'holdout_WT' : self.holdout_WT,
            'features' : self.features,
            'ensemble' : self.ensemble,
            'custom_transformations' : self.custom_transformations})
        mochi_task_args = copy.deepcopy({
            'directory' : self.get_task_directory(seed),
            'batch_size' : self.batch_size,
            'learn_rate' : self.learn_rate,
            'num_epochs' : self.num_epochs,
            'num_epochs_grid' : self.num_epochs_grid,
            'l1_regularization_factor' : self.l1_regularization_factor,
            'l2_regularization_factor' : self.l2_regularization_factor,
            'training_resample' : self.training_resample,
            'early_stopping' : self.early_stopping,
            'scheduler_gamma' : self.scheduler_gamma,
            'loss_function_name' : self.loss_function_name,
            'sos_architecture' : self.sos_architecture,
            'sos_outputlinear' : self.sos_outputlinear})
        return mochi_data_args, mochi_task_args

    def build_task(
        self,
        mochi_data_args,
        mochi_task_args):
        """
        Build MochiData and MochiTask instances for one execution phase.

        :param mochi_data_args: Dictionary of arguments for MochiData constructor (required).
        :param mochi_task_args: Dictionary of arguments for MochiTask constructor (required).
        :returns: Tuple of MochiData object and MochiTask object.
        """
        mochi_data = MochiData(**mochi_data_args)
        print("build_task: Initializing MochiTask")
        mochi_task = MochiTask(
            data = mochi_data,
            **mochi_task_args)
        return mochi_data, mochi_task

    def finalize_task_outputs(
        self,
        mochi_task,
        RT = None,
        seq_position_offset = 0,
        save_model = True,
        save_report = True,
        save_weights = True):
        """
        Persist saved models, reports, and weight summaries for a task.

        :param mochi_task: MochiTask object (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param save_model: Whether or not to save all models (default:True).
        :param save_report: Whether or not to save task report (default:True).
        :param save_weights: Whether or not to save model weights (default:True).
        :returns: MochiTask object.
        """
        if save_model:
            mochi_task.save(overwrite = True)

        if save_report:
            MochiReport(
                task = mochi_task,
                RT = RT)

        if save_weights:
            mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT)
            mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT,
                aggregate = True,
                aggregate_absolute_value = False)
            mochi_task.get_additive_trait_weights(
                seq_position_offset = seq_position_offset,
                RT = RT,
                aggregate = True,
                aggregate_absolute_value = True)

        return mochi_task

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

        for stage_index in range(1, self.get_sparse_stage_count() + 1):
            if os.path.exists(self.get_task_directory(stage_index)):
                print("Error: Task directory already exists.")
                break
            try:
                mochi_data_args, mochi_task_args, stage_settings = self.build_sparse_stage_inputs(stage_index)
                self.tasks[stage_index] = self.run_cv_task(
                    mochi_data_args = mochi_data_args,
                    mochi_task_args = mochi_task_args,
                    RT = self.RT,
                    seq_position_offset = self.seq_position_offset,
                    init_weights = init_weights,
                    fix_weights = self.fix_weights,
                    save_model = stage_settings['save_model'],
                    save_report = stage_settings['save_report'],
                    save_weights = stage_settings['save_weights'])
            except ValueError:
                print("Error: Failed to create MochiTask.")
                break

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
            if os.path.exists(self.get_task_directory(seedi)):
                print("Error: Task directory already exists.")
                break
            #Run
            try:
                self.tasks[seedi] = self.run_full_task(
                    seed = seedi,
                    init_weights = init_weights,
                    fix_weights = self.fix_weights)
            except ValueError:
                print("Error: Failed to create MochiTask.")
                break

    def run_full_task(
        self,
        seed,
        init_weights = None,
        fix_weights = {},
        save_model = True,
        save_report = True,
        save_weights = True):
        """
        Run the legacy grid-search-plus-all-folds execution for one seed.

        :param seed: Random seed for task execution (required).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param save_model: Whether or not to save all models (default:True).
        :param save_report: Whether or not to save task report (default:True).
        :param save_weights: Whether or not to save model weights (default:True).
        :returns: MochiTask object.
        """
        mochi_data_args, mochi_task_args = self.build_cv_task_inputs(seed)
        mochi_task = self.run_cv_task(
            mochi_data_args = mochi_data_args,
            mochi_task_args = mochi_task_args,
            RT = self.RT,
            seq_position_offset = self.seq_position_offset,
            init_weights = init_weights,
            fix_weights = fix_weights,
            save_model = save_model,
            save_report = save_report,
            save_weights = save_weights)
        self.tasks[seed] = mochi_task
        return mochi_task

    def run_grid_search_task(
        self,
        seed,
        init_weights = None,
        fix_weights = {},
        overwrite = False):
        """
        Build a task for one seed, run grid search only, and persist the result.

        :param seed: Random seed for task execution (required).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param overwrite: Whether or not to overwrite an existing saved task (default:False).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(seed)
        if os.path.exists(os.path.join(task_directory, "saved_models")) and not overwrite:
            mochi_task = MochiTask(directory = task_directory)
            grid_search_models = [i for i in mochi_task.models if i.metadata.grid_search == True]
            if len(grid_search_models) == 0:
                print("Error: Saved task directory does not contain grid search models.")
                raise ValueError
            print("run_grid_search_task: Reusing existing grid search artifacts")
            self.tasks[seed] = mochi_task
            return mochi_task

        mochi_data_args, mochi_task_args = self.build_cv_task_inputs(seed)
        _, mochi_task = self.build_task(
            mochi_data_args = mochi_data_args,
            mochi_task_args = mochi_task_args)

        print("run_grid_search_task: Starting grid search")
        mochi_task.grid_search(
            seed = seed,
            init_weights = init_weights,
            fix_weights = fix_weights)
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            save_model = True,
            save_report = False,
            save_weights = False)
        self.tasks[seed] = mochi_task
        return mochi_task

    def run_sparse_stage_grid_search(
        self,
        stage_index,
        init_weights = None,
        fix_weights = {},
        overwrite = False):
        """
        Build and run grid search for one sparse stage.

        :param stage_index: One-based sparse stage index (required).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param overwrite: Whether or not to overwrite existing stage artifacts (default:False).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(stage_index)
        if os.path.exists(os.path.join(task_directory, "saved_models")) and not overwrite:
            mochi_task = MochiTask(directory = task_directory)
            grid_search_models = [i for i in mochi_task.models if i.metadata.grid_search == True]
            if len(grid_search_models) == 0:
                print("Error: Saved sparse stage directory does not contain grid search models.")
                raise ValueError
            print(f"run_sparse_stage_grid_search: Reusing existing grid search artifacts for stage {stage_index}")
            self.tasks[stage_index] = mochi_task
            return mochi_task

        mochi_data_args, mochi_task_args, _ = self.build_sparse_stage_inputs(stage_index)
        _, mochi_task = self.build_task(
            mochi_data_args = mochi_data_args,
            mochi_task_args = mochi_task_args)

        print(f"run_sparse_stage_grid_search: Starting grid search for stage {stage_index}")
        mochi_task.grid_search(
            seed = mochi_data_args['seed'],
            init_weights = init_weights,
            fix_weights = fix_weights)
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            save_model = True,
            save_report = False,
            save_weights = False)
        self.tasks[stage_index] = mochi_task
        return mochi_task

    def run_fit_fold_task(
        self,
        seed,
        fold,
        grid_search_fold = 1,
        init_weights = None,
        fix_weights = {},
        overwrite = False):
        """
        Load saved grid-search artifacts and fit one fold in an isolated directory.

        :param seed: Random seed for task execution (required).
        :param fold: Cross-validation fold to fit (required).
        :param grid_search_fold: Cross-validation fold of grid search models (default:1).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param overwrite: Whether or not to overwrite an existing fold directory (default:False).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(seed)
        fold_directory = self.get_fold_directory(seed, fold)
        if os.path.exists(os.path.join(fold_directory, "saved_models")) and not overwrite:
            try:
                mochi_task = MochiTask(directory = fold_directory)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                print(f"run_fit_fold_task: Saved fold artifacts for fold {fold} are corrupted; rebuilding fold output")
                shutil.rmtree(fold_directory, ignore_errors = True)
            else:
                fold_models = [
                    i for i in mochi_task.models
                    if (i.metadata.grid_search == False) and (i.metadata.fold == fold)]
                if len(fold_models) == 0:
                    print("Error: Saved fold directory does not contain fit_best models.")
                    raise ValueError
                print("run_fit_fold_task: Reusing existing fold artifacts")
                return mochi_task

        mochi_task = MochiTask(directory = task_directory)
        os.makedirs(fold_directory, exist_ok = True)
        mochi_task.directory = fold_directory
        print("run_fit_fold_task: Starting fit_best")
        mochi_task.fit_best(
            fold = fold,
            grid_search_fold = grid_search_fold,
            seed = seed,
            init_weights = init_weights,
            fix_weights = fix_weights)
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            save_model = True,
            save_report = False,
            save_weights = False)
        return mochi_task

    def run_sparse_stage_fit_fold(
        self,
        stage_index,
        fold,
        grid_search_fold = 1,
        init_weights = None,
        fix_weights = {},
        overwrite = False):
        """
        Fit one fold for a sparse stage using saved stage grid-search artifacts.

        :param stage_index: One-based sparse stage index (required).
        :param fold: Cross-validation fold to fit (required).
        :param grid_search_fold: Cross-validation fold of grid search models (default:1).
        :param init_weights: Task to use for model weight initialization (optional).
        :param fix_weights: Dictionary of layer names to fix weights (required).
        :param overwrite: Whether or not to overwrite an existing fold directory (default:False).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(stage_index)
        fold_directory = self.get_fold_directory(stage_index, fold)
        if os.path.exists(os.path.join(fold_directory, "saved_models")) and not overwrite:
            try:
                mochi_task = MochiTask(directory = fold_directory)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                print(
                    f"run_sparse_stage_fit_fold: Saved fold artifacts for stage {stage_index}, fold {fold} are corrupted; rebuilding fold output")
                shutil.rmtree(fold_directory, ignore_errors = True)
            else:
                fold_models = [
                    i for i in mochi_task.models
                    if (i.metadata.grid_search == False) and (i.metadata.fold == fold)]
                if len(fold_models) == 0:
                    print("Error: Saved sparse stage fold directory does not contain fit_best models.")
                    raise ValueError
                print(f"run_sparse_stage_fit_fold: Reusing existing fold artifacts for stage {stage_index}, fold {fold}")
                return mochi_task

        mochi_task = MochiTask(directory = task_directory)
        os.makedirs(fold_directory, exist_ok = True)
        mochi_task.directory = fold_directory
        print(f"run_sparse_stage_fit_fold: Starting fit_best for stage {stage_index}, fold {fold}")
        mochi_task.fit_best(
            fold = fold,
            grid_search_fold = grid_search_fold,
            seed = self.seed,
            init_weights = init_weights,
            fix_weights = fix_weights)
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            save_model = True,
            save_report = False,
            save_weights = False)
        return mochi_task

    def merge_parallel_task(
        self,
        seed,
        RT = None,
        seq_position_offset = 0,
        save_model = True,
        save_report = True,
        save_weights = True):
        """
        Merge per-fold task directories back into the canonical task directory.

        :param seed: Random seed for task execution (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :param save_model: Whether or not to save all models (default:True).
        :param save_report: Whether or not to save task report (default:True).
        :param save_weights: Whether or not to save model weights (default:True).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(seed)
        print(f"merge_parallel_task: Starting merge for seed {seed} across {self.k_folds} folds")
        mochi_task = MochiTask(directory = task_directory)
        mochi_task.models = [i for i in mochi_task.models if i.metadata.grid_search == True]
        merged_folds = []
        skipped_folds = []
        for fold in range(1, self.k_folds+1):
            fold_directory = self.get_fold_directory(seed, fold)
            print(f"merge_parallel_task: Loading fold {fold}/{self.k_folds} from {fold_directory}")
            if not os.path.exists(os.path.join(fold_directory, "saved_models")):
                print(f"merge_parallel_task: Skipping fold {fold}/{self.k_folds} because saved_models is missing")
                skipped_folds.append(fold)
                continue
            try:
                fold_task = MochiTask(directory = fold_directory)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                print(f"merge_parallel_task: Skipping corrupted fold directory for fold {fold}/{self.k_folds}")
                skipped_folds.append(fold)
                continue
            fold_models = [
                i for i in fold_task.models
                if (i.metadata.grid_search == False) and (i.metadata.fold == fold)]
            if len(fold_models) == 0:
                print(f"merge_parallel_task: Skipping fold {fold}/{self.k_folds} because no fit_best models were found")
                skipped_folds.append(fold)
                continue
            mochi_task.models.extend(fold_models)
            print(f"merge_parallel_task: Loaded fold {fold}/{self.k_folds} with {len(fold_models)} fit_best model(s)")
            merged_folds.append(fold)

        if len(merged_folds) == 0:
            print("Error: No completed fold directories found.")
            raise ValueError
        if len(skipped_folds) != 0:
            print(f"merge_parallel_task: Merging partial fold set; merged folds {merged_folds}, skipped folds {skipped_folds}")

        mochi_task.directory = task_directory
        print(f"merge_parallel_task: Finalizing merged outputs for seed {seed}")
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            RT = RT,
            seq_position_offset = seq_position_offset,
            save_model = save_model,
            save_report = save_report,
            save_weights = save_weights)
        print(f"merge_parallel_task: Completed merge for seed {seed}")
        self.tasks[seed] = mochi_task
        return mochi_task

    def merge_sparse_stage(
        self,
        stage_index,
        RT = None,
        seq_position_offset = 0):
        """
        Merge one sparse stage's per-fold models back into its canonical task directory.

        :param stage_index: One-based sparse stage index (required).
        :param RT: R=gas constant (in kcal/K/mol) * T=Temperature (in K) (optional).
        :param seq_position_offset: Sequence position offset (default:0).
        :returns: MochiTask object.
        """
        stage_settings = self.get_sparse_stage_settings(stage_index)
        task_directory = self.get_task_directory(stage_index)
        print(f"merge_sparse_stage: Starting merge for stage {stage_index} across {self.k_folds} folds")
        mochi_task = MochiTask(directory = task_directory)
        mochi_task.models = [i for i in mochi_task.models if i.metadata.grid_search == True]
        merged_folds = []
        skipped_folds = []
        for fold in range(1, self.k_folds+1):
            fold_directory = self.get_fold_directory(stage_index, fold)
            print(f"merge_sparse_stage: Loading fold {fold}/{self.k_folds} for stage {stage_index} from {fold_directory}")
            if not os.path.exists(os.path.join(fold_directory, "saved_models")):
                print(f"merge_sparse_stage: Skipping stage {stage_index}, fold {fold}/{self.k_folds} because saved_models is missing")
                skipped_folds.append(fold)
                continue
            try:
                fold_task = MochiTask(directory = fold_directory)
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                print(f"merge_sparse_stage: Skipping corrupted fold directory for stage {stage_index}, fold {fold}/{self.k_folds}")
                skipped_folds.append(fold)
                continue
            fold_models = [
                i for i in fold_task.models
                if (i.metadata.grid_search == False) and (i.metadata.fold == fold)]
            if len(fold_models) == 0:
                print(f"merge_sparse_stage: Skipping stage {stage_index}, fold {fold}/{self.k_folds} because no fit_best models were found")
                skipped_folds.append(fold)
                continue
            mochi_task.models.extend(fold_models)
            print(f"merge_sparse_stage: Loaded stage {stage_index}, fold {fold}/{self.k_folds} with {len(fold_models)} fit_best model(s)")
            merged_folds.append(fold)

        if len(merged_folds) == 0:
            print("Error: No completed fold directories found.")
            raise ValueError
        if len(skipped_folds) != 0:
            print(f"merge_sparse_stage: Merging partial fold set for stage {stage_index}; merged folds {merged_folds}, skipped folds {skipped_folds}")

        mochi_task.directory = task_directory
        print(f"merge_sparse_stage: Finalizing merged outputs for stage {stage_index}")
        self.finalize_task_outputs(
            mochi_task = mochi_task,
            RT = RT,
            seq_position_offset = seq_position_offset,
            save_model = stage_settings['save_model'],
            save_report = stage_settings['save_report'],
            save_weights = stage_settings['save_weights'])
        print(f"merge_sparse_stage: Completed merge for stage {stage_index}")
        self.tasks[stage_index] = mochi_task
        return mochi_task

    def merge_grid_search_conditions(
        self,
        seed,
        overwrite = False,
        stage_index = None):
        """
        Merge isolated per-condition grid-search runs into the canonical task directory.

        :param seed: Task seed identifier (required).
        :param overwrite: Whether or not to overwrite existing merged artifacts (default:False).
        :param stage_index: Optional sparse stage identifier (default:None).
        :returns: MochiTask object.
        """
        task_directory = self.get_task_directory(seed)
        if os.path.exists(os.path.join(task_directory, "saved_models")) and not overwrite:
            mochi_task = MochiTask(directory = task_directory)
            grid_search_models = [i for i in mochi_task.models if i.metadata.grid_search == True]
            if len(grid_search_models) != 0:
                print("merge_grid_search_conditions: Reusing existing merged grid-search artifacts")
                self.tasks[seed] = mochi_task
                return mochi_task

        condition_task_directories = self.list_grid_condition_task_directories(
            seed = seed,
            stage_index = stage_index)
        if len(condition_task_directories) == 0:
            print("Error: No grid condition task directories found.")
            raise ValueError

        merged_task = None
        merged_models = []
        for condition_task_directory in condition_task_directories:
            if not os.path.exists(os.path.join(condition_task_directory, "saved_models")):
                print("Error: Grid condition task directory does not exist.")
                raise ValueError
            condition_task = MochiTask(directory = condition_task_directory)
            condition_models = [i for i in condition_task.models if i.metadata.grid_search == True]
            if len(condition_models) == 0:
                print("Error: Grid condition task directory does not contain grid-search models.")
                raise ValueError
            if merged_task is None:
                merged_task = condition_task
            merged_models.extend(condition_models)

        merged_task.models = merged_models
        merged_task.directory = task_directory
        self.finalize_task_outputs(
            mochi_task = merged_task,
            save_model = True,
            save_report = False,
            save_weights = False)
        self.tasks[seed] = merged_task
        return merged_task

    def merge_sparse_stage_grid_search(
        self,
        stage_index,
        overwrite = False):
        """
        Merge isolated per-condition sparse-stage grid-search runs into the stage task directory.

        :param stage_index: One-based sparse stage index (required).
        :param overwrite: Whether or not to overwrite existing merged artifacts (default:False).
        :returns: MochiTask object.
        """
        merged_task = self.merge_grid_search_conditions(
            seed = stage_index,
            overwrite = overwrite,
            stage_index = stage_index)
        self.tasks[stage_index] = merged_task
        return merged_task

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

        _, mochi_task = self.build_task(
            mochi_data_args = mochi_data_args,
            mochi_task_args = mochi_task_args)

        #Grid search
        print("run_cv_task: Starting grid search")
        mochi_task.grid_search(
            seed = mochi_data_args['seed'],
            init_weights = init_weights,
            fix_weights = fix_weights)
        #Fit model using best hyperparameters
        print("run_cv_task: Starting fit_best loop")
        for i in range(mochi_data_args['k_folds']):
            print(f"run_cv_task: Fitting best model for fold {i+1}/{mochi_data_args['k_folds']}")
            mochi_task.fit_best(
                fold = i+1, 
                seed = mochi_data_args['seed'],
                init_weights = init_weights,
                fix_weights = fix_weights)

        return self.finalize_task_outputs(
            mochi_task = mochi_task,
            RT = RT,
            seq_position_offset = seq_position_offset,
            save_model = save_model,
            save_report = save_report,
            save_weights = save_weights)

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
        model_design.file = str(input_obj).split(",")
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
            features = {None: list(mochi_task.data.get_feature_names())},
            ensemble = mochi_task.data.ensemble)

        # Align feature column order with the trained task before prediction.
        mochi_data.select_feature_columns(list(mochi_task.data.get_feature_names()))
        mochi_data.feature_names = mochi_data.get_feature_names()
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




