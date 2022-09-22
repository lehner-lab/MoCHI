#!/usr/bin/env python

"""
main() module -- MoCHI Command Line tool
"""

import os
import argparse
import pathlib
from pathlib import Path
import pandas as pd
from pymochi.project import MochiProject

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
    parser.add_argument('--project_name', type = str, default = "mochi_project", help = "project name (output will be saved to output_directory/project_name) (default: 'mochi_project')")
    parser.add_argument('--order_subset', type = str, help = "comma-separated list of integer mutation orders to consider (default: all variants considered)")
    parser.add_argument('--downsample_observations', type = float, help = "number (if integer) or proportion (if float) of observations to retain including WT (default: no downsampling)")
    parser.add_argument('--downsample_interactions', help = "number (if integer) or proportion (if float) or list of integer numbers (if string) of interaction terms to retain (default: no downsampling)")
    parser.add_argument('--max_interaction_order', type = int, default = 1, help = "maximum interaction order (default: 1)")
    parser.add_argument('--k_folds', type = int, default = 10, help = "number of cross-validation folds where test set%% = 100/k_folds (default: 10)")
    parser.add_argument('--validation_factor', type = int, default = 2, help = "validation factor where validation set%% = 100/k_folds*validation_factor (default: 2 i.e. 20%%)")
    parser.add_argument('--holdout_minobs', type = int, default = 0, help = "minimum number of observations of additive trait weights to be held out (default: 0)")
    parser.add_argument('--holdout_orders', type = str, help = "comma-separated list of integer mutation orders corresponding to retained variants (default: variants of all mutation orders can be held out)")
    parser.add_argument('--holdout_WT', action='store_true', default = False, help = "WT variant can be held out (default: False)")
    parser.add_argument('--features', type = pathlib.Path, default = None, help = "path to features file (default: None)")
    parser.add_argument('--ensemble', action='store_true', default = False, help = "use ensemble feature encoding (default: False)")
    parser.add_argument('--num_epochs_grid', type = int, default = 100, help = "number of grid search epochs (default: 100)")
    parser.add_argument('--num_epochs', type = int, default = 1000, help = "maximum number of training epochs (default: 1000)")
    parser.add_argument('--batch_size', default = "512,1024,2048", help = "comma-separated list of minibatch sizes to consider during grid search (default: '512,1024,2048')")
    parser.add_argument('--learn_rate', default = 0.05, help = "comma-separated list of learning rates to consider during grid search (default: 0.05)")
    parser.add_argument('--seed', type = int, default = 1, help = "random seed for training target data resampling (default: 1)")
    parser.add_argument('--l1_regularization_factor', default = 0, help = "lambda factor applied to L1 norm (default: 0)")
    parser.add_argument('--l2_regularization_factor', default = 0, help = "lambda factor applied to L2 norm (default: 0)")
    parser.add_argument('--seq_position_offset', type = int, default = 0, help = "sequence position offset (default: 0)")
    parser.add_argument('--temperature', type = float, default = 30.0, help = "temperature in degrees celsius (default: 30.0)")
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

    #Get command line arguments
    parser = init_argparse(demo_mode = demo_mode)
    args = parser.parse_args()

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

    #Reformat lists
    if args.order_subset!=None:
        args.order_subset = [int(i) for i in args.order_subset.split(',')]
    if args.holdout_orders==None:
        args.holdout_orders = []
    else:
        args.holdout_orders = [int(i) for i in args.holdout_orders.split(',')]
    if args.features==None:
        args.features = []

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
        max_interaction_order = args.max_interaction_order,
        downsample_observations = args.downsample_observations,
        downsample_interactions = args.downsample_interactions,
        k_folds = args.k_folds,
        validation_factor = args.validation_factor, 
        holdout_minobs = args.holdout_minobs, 
        holdout_orders = args.holdout_orders, 
        holdout_WT = args.holdout_WT,
        features = args.features,
        ensemble = args.ensemble,
        batch_size = args.batch_size,
        learn_rate = args.learn_rate,
        num_epochs = args.num_epochs,
        num_epochs_grid = args.num_epochs_grid,
        l1_regularization_factor = args.l1_regularization_factor,
        l2_regularization_factor = args.l2_regularization_factor)

    #######################################################################
    ## PREDICT PHENOTYPES ##
    #######################################################################

    #Predict supplementary variant phenotypes (if supplied)
    if args.predict!=None:
        mochi_project.predict(args.predict)


