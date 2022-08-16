"""
Unit and regression test for the pymochi package.
"""

# Import package, test suite, and other packages as needed
import sys
import os
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path

import pytest

import pymochi
from pymochi.data import *
from pymochi.models import *
from pymochi.report import *

def test_pymochi_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pymochi" in sys.modules

def test_MochiData_init_model_design_not_DataFrame(capsys):
    """Test MochiData initialization when model design not a pandas DataFrame"""
    MochiData("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "Error: Model design is not a pandas DataFrame.\n"

def test_MochiData_init_all_files_nonexistant(capsys):
    """Test MochiData initialization when only non-existant files supplied"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        "Hello World!1",
        "Hello World!2"]
    MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No Fitness datasets to merge."

def test_MochiData_init_one_file_nonexistant(capsys):
    """Test MochiData initialization when one non-existant file supplied"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        "Hello World!"]
    MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Fitness datasets cannot be merged: WT variants do not match."

def test_MochiData_init_duplicate_files(capsys):
    """Test MochiData initialization with duplicated file"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        "Hello World!",
        "Hello World!"]
    MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out == "Error: Duplicated input files.\n"

def test_MochiTask_init_no_MochiData_empty_directory():
    """Test MochiTask initialization when no MochiData nor saved MochiTask in directory supplied"""
    assert MochiTask(directory = str(Path(__file__).parent))

def create_dummy_task():
    #Delete entire directory contents
    shutil.rmtree(str(Path(__file__).parent / "temp"), ignore_errors=True)
    #Create a MochiTask
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    mochi_task = MochiTask(
        directory = str(Path(__file__).parent / "temp"),
        data = MochiData(
            model_design = model_design,
            downsample_observations = 0.1))
    mochi_task.save()
    # return mochi_task

def test_fit_best_no_grid_search_models(capsys):
    """Test fit_best function when no grid search models available"""
    #Create dummy task
    create_dummy_task()
    mochi_task = MochiTask(directory = str(Path(__file__).parent / "temp"))
    #Fit best model
    mochi_task.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No grid search models available."

def test_fit_best_exploded_grid_search_models(capsys):
    """Test fit_best function when all grid search models have gradient explosion"""
    #Create dummy task
    mochi_task = MochiTask(directory = str(Path(__file__).parent / "temp"))
    #Load model data
    model_data = mochi_task.data.get_data()
    #Add 3 grid search models
    for i in range(3):
        mochi_task.models += [mochi_task.new_model(model_data)]
        model = mochi_task.models[-1]
        model.metadata = MochiModelMetadata(
            fold = 1,
            seed = 1,
            grid_search = True,
            batch_size = mochi_task.batch_size,
            learn_rate = mochi_task.learn_rate,
            num_epochs = mochi_task.num_epochs,
            num_epochs_grid = mochi_task.num_epochs_grid,
            l1_regularization_factor = mochi_task.l1_regularization_factor,
            l2_regularization_factor = mochi_task.l2_regularization_factor,
            training_resample = True,
            early_stopping = True,
            scheduler_gamma = mochi_task.scheduler_gamma,
            scheduler_epochs = 10)
        model.training_history['val_loss'] = [1.0, 1.0, np.nan]
    #Fit best model
    mochi_task.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No valid grid search models available."




