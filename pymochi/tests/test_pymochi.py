"""
Unit and regression test for the pymochi package.
"""

# Import package, test suite, and other packages as needed
import sys
import os
from functools import lru_cache
import pandas as pd
import numpy as np
import pathlib
from pathlib import Path
from scipy import sparse as sp

import pytest

import pymochi
from pymochi.data import *
from pymochi.models import *
from pymochi.project import *
from pymochi.report import *


def make_demo_model_design():
    """Create a fresh model design pointing at the bundled toy datasets."""
    model_design = pd.read_csv(
        Path(__file__).parent.parent / "data/model_design.txt",
        sep = "\t",
        index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    return model_design


@lru_cache(maxsize = None)
def get_demo_mochi_data(
    max_interaction_order = 1,
    downsample_observations = None,
    downsample_interactions = None,
    seed = 1):
    """Build and cache read-only toy MochiData fixtures for regression tests."""
    return MochiData(
        model_design = make_demo_model_design(),
        max_interaction_order = max_interaction_order,
        downsample_observations = downsample_observations,
        downsample_interactions = downsample_interactions,
        seed = seed)

def test_pymochi_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "pymochi" in sys.modules

def test_MochiData_init_model_design_not_DataFrame(capsys):
    """Test MochiData initialization when model design not a pandas DataFrame"""
    with pytest.raises(ValueError) as e_info:
        MochiData("Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "Error: Model design is not a pandas DataFrame.\n" and e_info

def test_MochiData_init_all_files_nonexistant(capsys):
    """Test MochiData initialization when only non-existant files supplied"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        "Hello World!1",
        "Hello World!2"]
    with pytest.raises(ValueError) as e_info:
        MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Fitness file not found." and e_info

def test_MochiData_init_one_file_nonexistant(capsys):
    """Test MochiData initialization when one non-existant file supplied"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        "Hello World!"]
    with pytest.raises(ValueError) as e_info:
        MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Fitness file not found." and e_info

def test_MochiData_init_duplicate_files(capsys):
    """Test MochiData initialization with duplicated file"""
    #Create a problematic model design
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        "Hello World!",
        "Hello World!"]
    with pytest.raises(ValueError) as e_info:
        MochiData(model_design = model_design)
    captured = capsys.readouterr()
    assert captured.out == "Error: Duplicated fitness files.\n" and e_info

def test_MochiData_invalid_features_argument_type(capsys):
    """Test MochiData initialization with invalid features argument type"""
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    with pytest.raises(ValueError) as e_info:
        MochiData(
            model_design = model_design, 
            features = "Hello World!")
    captured = capsys.readouterr()
    assert captured.out == "Error: 'features' argument is not a dictionary.\n" and e_info

def test_MochiData_invalid_features_argument_trait_names(capsys):
    """Test MochiData initialization with invalid features argument trait names"""
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    #Create a problematic features dict
    features = {
        'Folding': ["WT"],
        'Hello World!': ["WT"]}
    with pytest.raises(ValueError) as e_info:
        MochiData(
            model_design = model_design, 
            features = features)
    captured = capsys.readouterr()
    assert captured.out == "Error: One or more invalid trait names in 'features' argument.\n" and e_info

# def test_MochiData_invalid_features_argument_features(capsys):
#     """Test MochiData initialization with invalid features argument features"""
#     model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
#     model_design['file'] = [
#         str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
#         str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
#     #Create a problematic features dict
#     features = {
#         'Folding': ["WT"],
#         'Binding': ["WT", "Hello World!"]}
#     with pytest.raises(ValueError) as e_info:
#         MochiData(
#             model_design = model_design, 
#             features = features)
#     captured = capsys.readouterr()
#     print(captured.out)
#     assert captured.out.split("\n")[-2] == "Warning: Invalid feature names: Hello World!" and e_info

def test_MochiData_invalid_features_argument_missingWT(capsys):
    """Test MochiData initialization with invalid features argument missing WT"""
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    #Create a problematic features dict
    features = {
        'Folding': ["WT"],
        'Binding': ["Hello"]}
    with pytest.raises(ValueError) as e_info:
        MochiData(
            model_design = model_design, 
            features = features)
    captured = capsys.readouterr()
    print(captured.out)
    assert captured.out.split("\n")[-2] == "Error: 'WT' missing for one or more traits in 'features' argument." and e_info

def test_MochiData_features_argument_Nonekey(capsys):
    """Test MochiData initialization with features argument None key"""
    model_design = pd.read_csv(Path(__file__).parent.parent / "data/model_design.txt", sep = "\t", index_col = False)
    model_design['file'] = [
        str(Path(__file__).parent.parent / "data/fitness_abundance.txt"),
        str(Path(__file__).parent.parent / "data/fitness_binding.txt")]
    #Create a problematic features dict
    features = {
        None: ["WT"]}
    mochi_data = MochiData(
        model_design = model_design, 
        features = features)
    captured = capsys.readouterr()
    print(captured.out)
    assert captured.out.split("\n")[-2] == "Done!" and mochi_data.Xohi.shape[1] == 1


def test_MochiData_max_interaction_order_1_preserves_additive_features():
    """Test max_interaction_order=1 keeps only additive one-hot features."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 1,
        downsample_observations = 0.02,
        seed = 1)
    assert list(mochi_data.Xohi.columns) == list(mochi_data.Xoh.columns)
    assert mochi_data.Xohi.shape == mochi_data.Xoh.shape
    assert [i for i in mochi_data.Xohi.columns if "_" in i] == []


def test_MochiData_max_interaction_order_2_adds_expected_pairwise_features():
    """Test max_interaction_order=2 adds the retained pairwise toy interactions."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    interaction_columns = [i for i in mochi_data.Xohi.columns if "_" in i]
    assert set(mochi_data.Xoh.columns).issubset(set(mochi_data.Xohi.columns))
    assert mochi_data.Xohi.shape[1] == mochi_data.Xoh.shape[1] + 2
    assert set(interaction_columns) == {'G46E_N53R', 'L13R_G54R'}


def test_MochiData_sparse_feature_store_materializes_expected_values():
    """Test the sparse retained-feature store materializes the same toy interaction values."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    assert mochi_data.feature_matrix_mode == "sparse"
    assert sp.issparse(mochi_data.get_xohi_values())
    feature_names = list(mochi_data.get_feature_names())
    interaction_names = [i for i in feature_names if "_" in i]
    interaction_indices = [feature_names.index(i) for i in interaction_names]
    materialized = mochi_data.materialize_feature_matrix(
        row_indices = np.arange(len(mochi_data)),
        feature_indices = interaction_indices,
        dtype = np.uint8)
    expected = np.column_stack([
        np.all(
            mochi_data.Xoh.loc[:, name.split("_")].to_numpy(dtype = np.uint8, copy = True) == 1,
            axis = 1).astype(np.uint8, copy = False)
        for name in interaction_names])
    assert np.array_equal(materialized, expected)


def test_MochiData_define_coefficient_groups_matches_materialized_training_activity():
    """Test sparse coefficient groups match direct training-row feature activity."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    feature_count = len(mochi_data.get_feature_names())
    phenotype_values = mochi_data.phenotypes.to_numpy(dtype = np.uint8, copy = False)
    for phenotype_index, phenotype_name in enumerate(mochi_data.phenotypes.columns):
        phenotype_mask = phenotype_values[:, phenotype_index] == 1
        coefficient_matrix = mochi_data.coefficients[phenotype_name]
        assert coefficient_matrix.shape == (feature_count, mochi_data.k_folds)
        for fold in range(1, mochi_data.k_folds + 1):
            fold_column = mochi_data.cvgroups[f"fold_{fold}"].to_numpy(copy = False)
            training_rows = np.flatnonzero(np.logical_and(phenotype_mask, fold_column == "training"))
            expected = np.any(
                mochi_data.materialize_feature_matrix(
                    row_indices = training_rows,
                    dtype = np.uint8) != 0,
                axis = 0).astype(np.uint8, copy = False)
            assert np.array_equal(coefficient_matrix[:, fold - 1], expected)


def test_MochiData_downsample_interactions_integer_and_string_match():
    """Test integer and string interaction limits keep the same single pairwise term."""
    full_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    int_limited_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        downsample_interactions = 1,
        seed = 1)
    str_limited_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        downsample_interactions = "1",
        seed = 1)
    full_interactions = [i for i in full_data.Xohi.columns if "_" in i]
    int_interactions = [i for i in int_limited_data.Xohi.columns if "_" in i]
    str_interactions = [i for i in str_limited_data.Xohi.columns if "_" in i]
    assert len(int_interactions) == 1
    assert int_interactions == str_interactions
    assert set(int_interactions).issubset(set(full_interactions))
    assert int_limited_data.Xohi.shape[1] == int_limited_data.Xoh.shape[1] + 1


def test_MochiData_downsample_interactions_two_matches_full_retained_terms():
    """Test retaining two interactions recovers the same pairwise toy terms as the full run."""
    full_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    limited_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        downsample_interactions = 2,
        seed = 1)
    full_interactions = [i for i in full_data.Xohi.columns if "_" in i]
    limited_interactions = [i for i in limited_data.Xohi.columns if "_" in i]
    assert set(limited_interactions) == set(full_interactions)
    assert len(limited_interactions) == 2
    assert limited_data.Xohi.shape[1] == limited_data.Xoh.shape[1] + 2


def test_MochiData_sparse_reorder_feature_columns_preserves_selected_values():
    """Test sparse feature-column reordering keeps the same values and exposed names."""
    mochi_data = MochiData(
        model_design = make_demo_model_design(),
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    reordered_columns = list(reversed(list(mochi_data.get_feature_names())[-3:]))
    mochi_data.reorder_feature_columns(reordered_columns)
    assert mochi_data.feature_matrix_mode == "sparse"
    assert list(mochi_data.get_feature_names()) == reordered_columns
    materialized = mochi_data.materialize_feature_matrix(
        row_indices = np.arange(len(mochi_data)),
        dtype = np.uint8)
    expected = np.column_stack([
        np.all(
            mochi_data.Xoh.loc[:, name.split("_")].to_numpy(dtype = np.uint8, copy = True) == 1,
            axis = 1).astype(np.uint8, copy = False)
        if "_" in name else mochi_data.Xoh.loc[:, name].to_numpy(dtype = np.uint8, copy = True)
        for name in reordered_columns])
    assert np.array_equal(materialized, expected)


def test_MaterializingRowDataLoader_cpu_batches_match_materialized_split():
    """Test row loader matches the expected CPU split tensors without shuffling."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    split_data = mochi_data.get_split_observation_data(
        fold = 1,
        seed = 1,
        training_resample = False)
    validation = split_data['validation']
    loader = MaterializingRowDataLoader(
        data = mochi_data,
        row_indices = validation['row_indices'],
        select = validation['select'],
        y = validation['y'],
        y_wt = validation['y_wt'],
        device = torch.device("cpu"),
        batch_size = 4,
        shuffle = False)
    batches = list(loader)
    select = torch.cat([batch[0] for batch in batches], dim = 0)
    X = torch.cat([batch[1] for batch in batches], dim = 0)
    y = torch.cat([batch[2] for batch in batches], dim = 0)
    y_wt = torch.cat([batch[3] for batch in batches], dim = 0)
    expected_X = torch.tensor(
        mochi_data.materialize_feature_matrix(
            row_indices = validation['row_indices'],
            dtype = loader.feature_numpy_dtype),
        dtype = loader.feature_tensor_dtype)
    assert torch.equal(select, validation['select'])
    assert torch.equal(X, expected_X)
    assert torch.equal(y, validation['y'])
    assert torch.equal(y_wt, validation['y_wt'])


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "CUDA required")
def test_DevicePrefetchLoader_cuda_prefetch_yields_device_batches(monkeypatch):
    """Test device prefetcher stages row-loader batches onto CUDA memory."""
    monkeypatch.setenv("MOCHI_GPU_PREFETCH", "1")
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    split_data = mochi_data.get_split_observation_data(
        fold = 1,
        seed = 1,
        training_resample = False)
    validation = split_data['validation']
    loader = MaterializingRowDataLoader(
        data = mochi_data,
        row_indices = validation['row_indices'],
        select = validation['select'],
        y = validation['y'],
        y_wt = validation['y_wt'],
        batch_size = 4,
        shuffle = False)
    batch = next(iter(DevicePrefetchLoader(loader, torch.device("cuda"))))
    torch.cuda.synchronize()
    assert batch[0].device.type == "cuda"
    assert batch[1].device.type == "cuda"
    assert batch[1].dtype == torch.float32
    assert batch[2].device.type == "cuda"
    assert batch[3].device.type == "cuda"


def forward_reference_loop(
    model,
    select,
    X,
    mask):
    """Reference the original forward-loop implementation for regression checks."""
    select_list = [torch.narrow(select, 1, i, 1) for i in range(select.shape[1])]
    observed_phenotypes = []
    for i in range(len(model.model_design)):
        additive_traits = [
            model.additivetraits[j-1](torch.mul(
                X,
                torch.reshape(
                    torch.narrow(torch.narrow(mask, 0, j-1, 1), 1, i, 1),
                    (1, -1))))
            for j in model.model_design.loc[i, 'trait']]
        if model.model_design.loc[i, 'transformation'] != "SumOfSigmoids":
            transformed_trait = get_transformation(
                model.model_design.loc[i, 'transformation'],
                custom = model.custom_transformations)(
                    additive_traits,
                    model.globalparams[i])
        else:
            transformed_trait = None
            for j in range(len(model.sumofsigmoids_list)):
                if j == 0:
                    transformed_trait = torch.sigmoid(
                        model.sumofsigmoids_list[j][int(model.model_design.loc[i, 'sos_index'])](
                            torch.cat(additive_traits, 1)))
                elif j != (len(model.sumofsigmoids_list)-1):
                    transformed_trait = torch.sigmoid(
                        model.sumofsigmoids_list[j][int(model.model_design.loc[i, 'sos_index'])](
                            transformed_trait))
                else:
                    if model.sos_outputlinear:
                        transformed_trait = model.sumofsigmoids_list[j][int(model.model_design.loc[i, 'sos_index'])](
                            transformed_trait)
                    else:
                        transformed_trait = torch.sigmoid(
                            model.sumofsigmoids_list[j][int(model.model_design.loc[i, 'sos_index'])](
                                transformed_trait))
        observed_phenotypes += [
            torch.mul(model.linears[i](transformed_trait), select_list[i])]
    return torch.stack(observed_phenotypes, dim = 0).sum(dim = 0)


def test_MochiModel_forward_matches_reference_loop():
    """Test forward refactor preserves the original masked additive-trait math."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    split_data = mochi_data.get_split_observation_data(
        fold = 1,
        seed = 1,
        training_resample = False)
    validation = split_data['validation']
    X = torch.tensor(
        mochi_data.materialize_feature_matrix(
            row_indices = validation['row_indices'],
            dtype = np.uint8),
        dtype = torch.float32)
    model = MochiModel(
        input_shape = X.shape[1],
        mask = validation['mask'].clone(),
        model_design = mochi_data.model_design.copy(),
        custom_transformations = mochi_data.custom_transformations,
        sos_architecture = [20],
        sos_outputlinear = False)
    custom_mask = validation['mask'].clone()
    custom_mask[0, 0, 0] = 0
    with torch.no_grad():
        expected_default = forward_reference_loop(
            model = model,
            select = validation['select'],
            X = X,
            mask = model.mask)
        actual_default = model(
            select = validation['select'],
            X = X)
        expected_override = forward_reference_loop(
            model = model,
            select = validation['select'],
            X = X,
            mask = custom_mask)
        actual_override = model(
            select = validation['select'],
            X = X,
            mask = custom_mask)
    assert torch.allclose(actual_default, expected_default)
    assert torch.allclose(actual_override, expected_override)

def test_MochiTask_init_no_MochiData_empty_directory(capsys):
    """Test MochiTask initialization when no MochiData nor saved MochiTask in directory supplied"""
    with pytest.raises(ValueError) as e_info:
        MochiTask(directory = str(Path(__file__).parent))
    captured = capsys.readouterr()
    assert captured.out == "Error: Saved models directory does not exist.\n" and e_info

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
    with pytest.raises(ValueError) as e_info:
        mochi_task.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No grid search models available." and e_info

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
            scheduler_epochs = 10,
            loss_function_name = 'WeightedL1',
            sos_architecture = [20],
            sos_outputlinear = False)
        model.training_history['val_loss'] = [1.0, 1.0, np.nan]
    #Fit best model
    with pytest.raises(ValueError) as e_info:
        mochi_task.fit_best(fold = 1, seed = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: No valid grid search models available." and e_info

def test_MochiProject_model_design_invalid_string_path(capsys):
    """Test MochiProject initialization when invalid model design string path supplied"""
    #Create invalid project
    mochi_project = MochiProject(
        directory = str(Path(__file__).parent / "temp"),
        model_design = "invalid_string_path")
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Invalid model_design file path: does not exist."

def test_MochiProject_model_design_invalid_type(capsys):
    """Test MochiProject initialization when invalid model design argument supplied"""
    #Create invalid project
    mochi_project = MochiProject(
        directory = str(Path(__file__).parent / "temp"),
        model_design = [])
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Invalid model_design file path: does not exist."

def test_MochiProject_features_invalid_string_path(capsys):
    """Test MochiProject initialization when invalid features string path supplied"""
    #Create invalid project
    mochi_project = MochiProject(
        directory = str(Path(__file__).parent / "temp"),
        model_design = str(Path(__file__).parent.parent / "data/model_design.txt"),
        features = "invalid_string_path")
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Invalid features file path: does not exist."

def test_MochiProject_features_invalid_type(capsys):
    """Test MochiProject initialization when invalid features argument supplied"""
    #Create invalid project
    mochi_project = MochiProject(
        directory = str(Path(__file__).parent / "temp"),
        model_design = str(Path(__file__).parent.parent / "data/model_design.txt"),
        features = 1)
    captured = capsys.readouterr()
    assert captured.out.split("\n")[-2] == "Error: Invalid features file path: does not exist."
