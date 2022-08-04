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

def test_MochiProject_init_no_MochiData_empty_directory():
    """Test MochiProject initialization when no MochiData nor saved MochiProject in directory supplied"""
    assert MochiProject(directory = str(Path(__file__).parent))

def create_dummy_project():
    #Delete entire directory contents
    shutil.rmtree(str(Path(__file__).parent / "temp"), ignore_errors=True)
    #Create a MochiProject
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    mochi_project = MochiProject(
        directory = str(Path(__file__).parent / "temp"),
        data = MochiData(model_design = model_design))
    mochi_project.save()
    # return mochi_project

def test_fit_best_no_grid_search_models(capsys):
    """Test fit_best function when no grid search models available"""
    #Create dummy project
    create_dummy_project()
    mochi_project = MochiProject(directory = str(Path(__file__).parent / "temp"))
    #Fit best model
    mochi_project.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No grid search models available."

def test_fit_best_exploded_grid_search_models(capsys):
    """Test fit_best function when all grid search models have gradient explosion"""
    #Create dummy project
    mochi_project = MochiProject(directory = str(Path(__file__).parent / "temp"))
    #Load model data
    model_data = mochi_project.data.get_data()
    #Add 3 grid search models
    for i in range(3):
        mochi_project.models += [mochi_project.new_model(model_data)]
        model = mochi_project.models[-1]
        model.metadata = MochiModelMetadata(
            fold = 1,
            seed = 1,
            grid_search = True,
            batch_size = mochi_project.batch_size,
            learn_rate = mochi_project.learn_rate,
            num_epochs = mochi_project.num_epochs,
            num_epochs_grid = mochi_project.num_epochs_grid,
            l1_regularization_factor = mochi_project.l1_regularization_factor,
            l2_regularization_factor = mochi_project.l2_regularization_factor,
            training_resample = True,
            early_stopping = True,
            scheduler_gamma = mochi_project.scheduler_gamma,
            scheduler_epochs = 10)
        model.training_history['val_loss'] = [1.0, 1.0, np.nan]
    #Fit best model
    mochi_project.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No valid grid search models available."




