#!/usr/bin/env python

"""
main() module -- MoCHI Command Line tool
"""

import argparse
import builtins
import os
import pathlib
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

from pymochi.project import MochiProject

ORIGINAL_PRINT = builtins.print
BOOLEAN_OPTIONS = {
    "--holdout_WT",
    "--ensemble",
    "--training_resample",
    "--early_stopping",
    "--sos_outputlinear"}
TRUE_VALUES = {"true", "t", "yes", "y", "1"}
FALSE_VALUES = {"false", "f", "no", "n", "0"}


def normalize_boolean_option_values(arguments):
    """
    Support legacy boolean values while using argparse boolean flags.
    """
    normalized = []
    i = 0
    while i < len(arguments):
        argument = arguments[i]
        if "=" in argument:
            option, value = argument.split("=", 1)
            value_lower = value.lower()
            if option in BOOLEAN_OPTIONS and value_lower in TRUE_VALUES | FALSE_VALUES:
                normalized.append(option if value_lower in TRUE_VALUES else "--no-" + option[2:])
                i += 1
                continue
        if argument in BOOLEAN_OPTIONS and i + 1 < len(arguments):
            value_lower = arguments[i + 1].lower()
            if value_lower in TRUE_VALUES | FALSE_VALUES:
                normalized.append(argument if value_lower in TRUE_VALUES else "--no-" + argument[2:])
                i += 2
                continue
        normalized.append(argument)
        i += 1
    return normalized


def configure_logging():
    """
    Route existing CLI print output through loguru with timestamps.
    """
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
        colorize = False)

    def timestamped_print(*args, **kwargs):
        output_file = kwargs.get("file", sys.stdout)
        if output_file not in (None, sys.stdout, sys.stderr):
            ORIGINAL_PRINT(*args, **kwargs)
            return
        message = kwargs.get("sep", " ").join(str(arg) for arg in args)
        logger.info(message)

    builtins.print = timestamped_print

def init_argparse(
    demo_mode = False
    ) -> argparse.ArgumentParser:
    """
    Initialize command line argument parser.

    :param demo_mode: whether to run in demo mode using toy data (default: False).
    :returns: ArgumentParser.
    """

    parser = argparse.ArgumentParser(
        description="MoCHI Command Line tool."
    )
    if not demo_mode:
        parser.add_argument('--model_design', type = pathlib.Path, help = "path to model design file")
    parser.add_argument('--output_directory', type = pathlib.Path, default = ".", help = "output directory")
    parser.add_argument('--project_name', type = str, default = "mochi_project", help = "project name (output will be saved to output_directory/project_name) (default: 'mochi_project')")
    parser.add_argument('--seed', type = int, default = 1, help = "random seed for training target data resampling (default: 1)")
    parser.add_argument('--temperature', type = float, default = 30.0, help = "temperature in degrees celsius (default: 30.0)")
    parser.add_argument('--seq_position_offset', type = int, default = 0, help = "sequence position offset (default: 0)")
    #MochiData arguments
    parser.add_argument('--order_subset', type = str, help = "comma-separated list of integer mutation orders to consider (default: all variants considered)")
    parser.add_argument('--downsample_observations', type = float, help = "number (if integer) or proportion (if float) of observations to retain including WT (default: no downsampling)")
    parser.add_argument('--downsample_interactions', help = "number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (default: no downsampling)")
    parser.add_argument('--max_interaction_order', type = int, default = 1, help = "maximum interaction order (default: 1)")
    parser.add_argument('--min_observed', type = int, default = 2, help = "minimum number of observations required to include interaction term (default:2)")
    parser.add_argument('--k_folds', type = int, default = 10, help = "number of cross-validation folds where test set%% = 100/k_folds (default: 10)")
    parser.add_argument('--validation_factor', type = int, default = 2, help = "validation factor where validation set%% = 100/k_folds*validation_factor (default: 2 i.e. 20%%)")
    parser.add_argument('--holdout_minobs', type = int, default = 0, help = "minimum number of observations of additive trait weights to be held out (default: 0)")
    parser.add_argument('--holdout_orders', type = str, help = "comma-separated list of integer mutation orders corresponding to retained variants (default: variants of all mutation orders can be held out)")
    parser.add_argument('--holdout_WT', action = argparse.BooleanOptionalAction, default = False, help = "WT variant can be held out (default: False)")
    parser.add_argument('--features', type = pathlib.Path, default = None, help = "path to features file (default: None)")
    parser.add_argument('--ensemble', action = argparse.BooleanOptionalAction, default = False, help = "use ensemble feature encoding (default: False)")
    parser.add_argument('--custom_transformations', type = pathlib.Path, default = None, help = "path to custom transformations file (default: None)")
    #MochiTask arguments
    parser.add_argument('--batch_size', default = "512,1024,2048", help = "comma-separated list of minibatch sizes to consider during grid search (default: '512,1024,2048')")
    parser.add_argument('--learn_rate', default = 0.05, help = "comma-separated list of learning rates to consider during grid search (default: 0.05)")
    parser.add_argument('--num_epochs', type = int, default = 1000, help = "maximum number of training epochs (default: 1000)")
    parser.add_argument('--num_epochs_grid', type = int, default = 100, help = "number of grid search epochs (default: 100)")
    parser.add_argument('--l1_regularization_factor', default = 0, help = "lambda factor applied to L1 norm (default: 0)")
    parser.add_argument('--l2_regularization_factor', default = 0.000001, help = "lambda factor applied to L2 norm (default: 0.000001)")
    parser.add_argument('--training_resample', action = argparse.BooleanOptionalAction, default = True, help = "whether or not to add random noise to training target data proportional to target error (default:True)")
    parser.add_argument('--early_stopping', action = argparse.BooleanOptionalAction, default = True, help = "whether or not to stop training early if validation loss not decreasing (default:True)")
    parser.add_argument('--scheduler_gamma', type = float, default = 0.98, help = "multiplicative factor of learning rate decay (default:0.98)")
    parser.add_argument('--loss_function_name', type = str, default = 'WeightedL1', help = "loss function name: one of 'WeightedL1', 'GaussianNLL' (default:'WeightedL1')")
    parser.add_argument('--sos_architecture', type = str, default = '20', help = "comma-separated list of integers corresponding to number of neurons per fully-connected sumOfSigmoids hidden layer (default: '20')")
    parser.add_argument('--sos_outputlinear', action = argparse.BooleanOptionalAction, default = False, help = "final sumOfSigmoids should be linear rather than sigmoidal (default:False)")
    parser.add_argument('--init_weights_directory', type = pathlib.Path, default = None, help = "path to project directory for model weight initialization (default: random model weight initialization)")
    parser.add_argument('--init_weights_task_id', type = int, default = 1, help = "task identifier to use for model weight initialization (default:1)")
    parser.add_argument('--fix_weights', type = pathlib.Path, default = None, help = "path to file of layer names to fix weights (default: no layers fixed)")
    parser.add_argument('--sparse_method', type = str, default = None, help = "sparse model inference method: one of 'sig_highestorder_step' (default: no sparse inference)")
    parser.add_argument('--phase', type = str, choices = ['full', 'grid_search', 'merge_grid_search', 'fit_best', 'merge_folds', 'sparse_grid_search', 'sparse_merge_grid_search', 'sparse_fit_best', 'sparse_merge_folds'], default = 'full', help = "execution phase to run (default: 'full')")
    parser.add_argument('--fold', type = int, default = None, help = "cross-validation fold to fit when --phase fit_best")
    parser.add_argument('--grid_search_fold', type = int, default = 1, help = "cross-validation fold containing grid search models (default: 1)")
    parser.add_argument('--stage_index', type = int, default = None, help = "one-based sparse stage index for sparse split phases")
    parser.add_argument('--predict', type = pathlib.Path, default = None, help = "path to supplementary variants file for prediction (default: None)")
    return parser

def main(
    demo_mode = False
    ):
    """
    Main function.

    :param demo_mode: whether to run in demo mode using toy data (default: False).
    :returns: Nothing.
    """

    configure_logging()

    #Get command line arguments
    parser = init_argparse(demo_mode = demo_mode)
    args, unknown_args = parser.parse_known_args(
        normalize_boolean_option_values(sys.argv[1:]))
    if unknown_args:
        logger.warning("Ignoring unknown command line arguments: {}", " ".join(unknown_args))

    #Load model design
    if demo_mode:
        args.model_design = pd.read_csv(Path(__file__).parent / "data/model_design.txt", sep = "\t", index_col = False)
        args.model_design['file'] = [
            str(Path(__file__).parent / "data/fitness_abundance.txt"),
            str(Path(__file__).parent / "data/fitness_binding.txt")]
        args.downsample_observations = 0.1
        args.project_name = "mochi_project_demo"
        args.k_folds = 5
        args.batch_size = "1024"

    #Prediction only
    ### TODO: make this work for already-existing project
    if args.model_design is None and not args.predict is None:
        args.model_design = pd.DataFrame()

    #Reformat lists and dictionaries
    if args.order_subset!=None:
        args.order_subset = [int(i) for i in args.order_subset.split(',')]
    if args.holdout_orders is None:
        args.holdout_orders = []
    else:
        args.holdout_orders = [int(i) for i in args.holdout_orders.split(',')]
    if args.features is None:
        args.features = []
    if args.fix_weights is None:
        args.fix_weights = {}
    if args.sos_architecture!=None:
        args.sos_architecture = [int(i) for i in args.sos_architecture.split(',')]
    sparse_stage_phases = {'sparse_grid_search', 'sparse_merge_grid_search', 'sparse_fit_best', 'sparse_merge_folds'}
    if args.k_folds < 3:
        raise ValueError("--k_folds must be at least 3")
    if args.phase in {'fit_best', 'sparse_fit_best'} and args.fold is None:
        raise ValueError("--fold is required when --phase fit_best")
    if args.phase in sparse_stage_phases and args.stage_index is None:
        raise ValueError("--stage_index is required for sparse split phases")
    if args.phase in sparse_stage_phases and args.sparse_method != 'sig_highestorder_step':
        raise ValueError("--sparse_method sig_highestorder_step is required for sparse split phases")
    if args.sparse_method is not None and args.phase != 'full' and args.phase not in sparse_stage_phases:
        raise ValueError("--sparse_method is only supported when --phase full")

    #######################################################################
    ## CREATE PROJECT ##
    #######################################################################

    #MoCHI project
    mochi_project = MochiProject(
        directory = os.path.join(args.output_directory, args.project_name),
        seed = args.seed,
        RT = (273+args.temperature)*0.001987,
        seq_position_offset = args.seq_position_offset,
        model_design = args.model_design,
        order_subset = args.order_subset,
        downsample_observations = args.downsample_observations,
        downsample_interactions = args.downsample_interactions,
        max_interaction_order = args.max_interaction_order,
        min_observed = args.min_observed,
        k_folds = args.k_folds,
        validation_factor = args.validation_factor, 
        holdout_minobs = args.holdout_minobs, 
        holdout_orders = args.holdout_orders, 
        holdout_WT = args.holdout_WT,
        features = args.features,
        ensemble = args.ensemble,
        custom_transformations = args.custom_transformations,
        batch_size = args.batch_size,
        learn_rate = args.learn_rate,
        num_epochs = args.num_epochs,
        num_epochs_grid = args.num_epochs_grid,
        l1_regularization_factor = args.l1_regularization_factor,
        l2_regularization_factor = args.l2_regularization_factor,
        training_resample = args.training_resample,
        early_stopping = args.early_stopping,
        scheduler_gamma = args.scheduler_gamma,
        loss_function_name = args.loss_function_name,
        sos_architecture = args.sos_architecture,
        sos_outputlinear = args.sos_outputlinear,
        init_weights_directory = args.init_weights_directory,
        init_weights_task_id = args.init_weights_task_id,
        fix_weights = args.fix_weights,
        sparse_method = args.sparse_method,
        auto_run = False)

    if args.phase == 'full':
        if args.sparse_method is None:
            mochi_project.run_full_task(
                seed = args.seed,
                fix_weights = mochi_project.fix_weights)
        elif args.sparse_method == 'sig_highestorder_step':
            mochi_project.run_sparse_sig_highestorder_step()
        else:
            raise ValueError(f"Unsupported sparse_method: {args.sparse_method}")
    elif args.phase == 'grid_search':
        mochi_project.run_grid_search_task(
            seed = args.seed,
            fix_weights = mochi_project.fix_weights)
    elif args.phase == 'merge_grid_search':
        mochi_project.merge_grid_search_conditions(
            seed = args.seed)
    elif args.phase == 'fit_best':
        mochi_project.run_fit_fold_task(
            seed = args.seed,
            fold = args.fold,
            grid_search_fold = args.grid_search_fold,
            fix_weights = mochi_project.fix_weights)
    elif args.phase == 'merge_folds':
        mochi_project.merge_parallel_task(
            seed = args.seed,
            RT = mochi_project.RT,
            seq_position_offset = mochi_project.seq_position_offset)
    elif args.phase == 'sparse_grid_search':
        mochi_project.run_sparse_stage_grid_search(
            stage_index = args.stage_index,
            fix_weights = mochi_project.fix_weights)
    elif args.phase == 'sparse_merge_grid_search':
        mochi_project.merge_sparse_stage_grid_search(
            stage_index = args.stage_index)
    elif args.phase == 'sparse_fit_best':
        mochi_project.run_sparse_stage_fit_fold(
            stage_index = args.stage_index,
            fold = args.fold,
            grid_search_fold = args.grid_search_fold,
            fix_weights = mochi_project.fix_weights)
    elif args.phase == 'sparse_merge_folds':
        mochi_project.merge_sparse_stage(
            stage_index = args.stage_index,
            RT = mochi_project.RT,
            seq_position_offset = mochi_project.seq_position_offset)

    #######################################################################
    ## PREDICT PHENOTYPES ##
    #######################################################################

    #Predict supplementary variant phenotypes (if supplied)
    ### TODO: make this work for already-existing project
    if args.predict!=None:
        mochi_project.predict(args.predict)


