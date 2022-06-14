#!/usr/bin/env python

"""
main() module -- MoCHI Command Line tool
"""

import os
import argparse
import pathlib
from pathlib import Path
import numpy as np
import pandas as pd
from pymochi.data import MochiData
from pymochi.models import MochiProject
from pymochi.report import MochiReport

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
        parser.add_argument('model_design', type = pathlib.Path, help = "path to model design file")
    parser.add_argument('--output_directory', type = pathlib.Path, default = ".", help = "output directory")
    parser.add_argument('--project_name', type = str, default = "mochi_model", help = "project name (output will be saved to output_directory/project_name) (default: 'mochi_model')")
    parser.add_argument('--order_subset', type = str, help = "comma-separated list of integer mutation orders to consider (default: all variants considered)")
    parser.add_argument('--downsample_proportion', type = float, help = "temperature in degrees celsius (default: no downsampling)")
    parser.add_argument('--max_interaction_order', type = int, default = 1, help = "maximum interaction order (default: 1)")
    parser.add_argument('--k_folds', type = int, default = 10, help = "number of cross-validation folds where test set%% = 100/k_folds (default: 10)")
    parser.add_argument('--validation_factor', type = int, default = 2, help = "validation factor where validation set%% = 100/k_folds*validation_factor (default: 2 i.e. 20%%)")
    parser.add_argument('--holdout_minobs', type = int, default = 0, help = "minimum number of observations of additive trait weights to be held out (default: 0)")
    parser.add_argument('--holdout_orders', type = str, help = "comma-separated list of integer mutation orders corresponding to retained variants (default: variants of all mutation orders can be held out)")
    parser.add_argument('--holdout_WT', action='store_true', default = False, help = "WT variant can be held out (default: False)")
    parser.add_argument('--num_epochs_grid', type = int, default = 100, help = "number of grid search epochs (default: 100)")
    parser.add_argument('--num_epochs', type = int, default = 1000, help = "maximum number of training epochs (default: 1000)")
    parser.add_argument('--batch_size', type = str, default = "512,1024,2048", help = "comma-separated list of minibatch sizes to consider during grid search (default: '512,1024,2048')")
    parser.add_argument('--learn_rate', type = str, default = "0.05", help = "comma-separated list of learning rates to consider during grid search (default: '0.05')")
    parser.add_argument('--seed', type = int, default = "1", help = "random seed for training target data resampling (default: 1)")
    parser.add_argument('--l1_regularization_factor', type = float, default = "0", help = "lambda factor applied to L1 norm (default: 0)")
    parser.add_argument('--l2_regularization_factor', type = float, default = "0", help = "lambda factor applied to L2 norm (default: 0)")
    parser.add_argument('--seq_position_offset', type = int, default = 0, help = "sequence position offset (default: 0)")
    parser.add_argument('--temperature', type = float, default = 30.0, help = "temperature in degrees celsius (default: 30.0)")
    return parser

def main(
    demo_mode = False
    ):
    """
    Main function.

    :param demo_mode: whether to run in demo mode using toy data (default: False).
    :returns: Nothing.
    """

    #Get command line arguments
    parser = init_argparse(demo_mode = demo_mode)
    args = parser.parse_args()

    #Load model design
    if demo_mode:
        model_design = pd.read_csv(Path(__file__).parent / "data/model_design.txt", sep = "\t", index_col = False)
        model_design['file'] = [
            str(Path(__file__).parent / "data/fitness_abundance.RData"),
            str(Path(__file__).parent / "data/fitness_binding.RData")]
        args.downsample_proportion = 0.1
        args.project_name = "mochi_model_demo"
        args.k_folds = 5
        args.batch_size = "1024"
    else:
        model_design = pd.read_csv(args.model_design, sep = "\t", index_col = False)
    model_design['trait'] = [i.split(',') for i in model_design['trait']]

    #Reformat comma-separated lists
    if args.order_subset==None:
        order_subset = args.order_subset
    else:
        order_subset = [int(i) for i in args.order_subset.split('')]
    if args.holdout_orders==None:
        holdout_orders = []
    else:
        holdout_orders = [int(i) for i in args.holdout_orders.split('')]

    #######################################################################
    ## PREPARE DATA ##
    #######################################################################

    #Load mochi data
    mochi_data = MochiData(
        model_design = model_design,
        order_subset = order_subset,
        downsample_proportion = args.downsample_proportion,
        max_interaction_order = args.max_interaction_order,
        k_folds = args.k_folds,
        seed = args.seed,
        validation_factor = args.validation_factor, 
        holdout_minobs = args.holdout_minobs, 
        holdout_orders = holdout_orders, 
        holdout_WT = args.holdout_WT)

    #######################################################################
    ## CREATE PROJECT ##
    #######################################################################

    #Create mochi project
    mochi_project = MochiProject(
        directory = os.path.join(args.output_directory, args.project_name),
        data = mochi_data,
        batch_size = args.batch_size,
        learn_rate = args.learn_rate,
        num_epochs = args.num_epochs,
        num_epochs_grid = args.num_epochs_grid,
        l1_regularization_factor = args.l1_regularization_factor,
        l2_regularization_factor = args.l2_regularization_factor)

    #######################################################################
    ## FIT MODEL ##
    #######################################################################

    #Grid search
    mochi_project.grid_search(seed = args.seed)

    #Fit best model
    for i in range(args.k_folds):
        mochi_project.fit_best(fold = i+1, seed = args.seed)

    #Save all models
    mochi_project.save(overwrite = True)

    #######################################################################
    ## MODEL REPORT ##
    #######################################################################

    #Get model weights
    energies = mochi_project.get_additive_trait_weights(
        seq_position_offset = args.seq_position_offset,
        RT = (273+args.temperature)*0.001987)

    #Generate project report
    mochi_report = MochiReport(
        project = mochi_project,
        RT = (273+args.temperature)*0.001987)

    #Aggregate energies per residue position
    energies_agg = mochi_project.get_additive_trait_weights(
        seq_position_offset = args.seq_position_offset,
        RT = (273+args.temperature)*0.001987,
        aggregate = True,
        aggregate_absolute_value = False)

    #Aggregate absolute value of energies per residue position
    energies_agg_abs = mochi_project.get_additive_trait_weights(
        seq_position_offset = args.seq_position_offset,
        RT = (273+args.temperature)*0.001987,
        aggregate = True,
        aggregate_absolute_value = True)



