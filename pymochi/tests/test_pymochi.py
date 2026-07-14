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
import torch

import pytest

import pymochi
import pymochi.main as mochi_main
import pymochi.project as mochi_project_module
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
    assert sp.issparse(mochi_data.feature_sparse_matrix)
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


def test_MochiData_ensemble_dense_features_keep_float_values():
    """Test ensemble dense features are not coerced to uint8."""
    mochi_data = MochiData.__new__(MochiData)
    mochi_data.ensemble = True
    mochi_data.activate_dense_feature_matrix(pd.DataFrame({
        "WT": np.array([1.0, -0.5], dtype = np.float32),
        "A1B": np.array([0.25, 0.0], dtype = np.float32)}))
    mochi_data.phenotypes = pd.DataFrame({"phenotype_1": np.array([1, 1], dtype = np.uint8)})
    mochi_data.fitness = pd.DataFrame({"fitness": np.array([0.1, 0.2], dtype = np.float32)})

    materialized = mochi_data.materialize_feature_matrix(row_indices = [0, 1])
    feature_tensor = torch.tensor(materialized, dtype = mochi_data.get_feature_tensor_dtype())

    assert materialized.dtype == np.float32
    assert np.array_equal(materialized, mochi_data.Xohi.to_numpy(dtype = np.float32))
    assert feature_tensor.dtype == torch.float32


def test_MochiData_dense_float_coefficient_groups_use_nonzero_activity():
    """Test fractional ensemble-like dense values count as active features."""
    mochi_data = MochiData.__new__(MochiData)
    mochi_data.ensemble = True
    mochi_data.activate_dense_feature_matrix(pd.DataFrame({
        "WT": np.array([0.25, 0.0], dtype = np.float32),
        "A1B": np.array([0.0, -0.5], dtype = np.float32)}))
    mochi_data.phenotypes = pd.DataFrame({"phenotype_1": np.array([1, 1], dtype = np.uint8)})
    mochi_data.cvgroups = pd.DataFrame({"fold_1": np.array(["training", "training"])})
    mochi_data.additive_trait_names = ["trait"]
    mochi_data.features_trait = {}

    mochi_data.define_coefficient_groups(k_folds = 1)

    assert np.array_equal(
        mochi_data.coefficients["phenotype_1"][:, 0],
        np.array([1, 1], dtype = np.uint8))


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


def test_MochiData_sparse_select_feature_columns_preserves_selected_values():
    """Test sparse feature-column selection keeps the same values and exposed names."""
    mochi_data = MochiData(
        model_design = make_demo_model_design(),
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    reordered_columns = list(reversed(list(mochi_data.get_feature_names())[-3:]))
    mochi_data.select_feature_columns(reordered_columns)
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

def dense_from_sparse_feature_payload(
    payload):
    """Reconstruct a dense float32 matrix from a sparse-native batch payload."""
    n_rows, n_features = payload['shape']
    dense = torch.zeros((n_rows, n_features), dtype = torch.float32)
    offsets = payload['offsets'].tolist()
    indices = payload['indices']
    values = payload.get(
        'values',
        torch.ones(indices.shape[0], dtype = torch.float32))
    for row_index, start in enumerate(offsets):
        stop = offsets[row_index + 1] if row_index + 1 < len(offsets) else len(indices)
        dense[row_index, indices[start:stop]] = values[start:stop]
    return dense

def test_MaterializingRowDataLoader_sparse_native_batches_match_materialized_split():
    """Test sparse-native row loader batches preserve the materialized split values."""
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
    assert feature_tensor_is_sparse_native(batches[0][1])
    select = torch.cat([batch[0] for batch in batches], dim = 0)
    X = torch.cat([dense_from_sparse_feature_payload(batch[1]) for batch in batches], dim = 0)
    y = torch.cat([batch[2] for batch in batches], dim = 0)
    y_wt = torch.cat([batch[3] for batch in batches], dim = 0)
    expected_X = torch.tensor(
        mochi_data.materialize_feature_matrix(
            row_indices = validation['row_indices'],
            dtype = np.uint8),
        dtype = torch.float32)
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

def test_MochiModel_sparse_native_forward_matches_dense():
    """Test sparse-native additive accumulation matches the dense forward path."""
    mochi_data = get_demo_mochi_data(
        max_interaction_order = 2,
        downsample_observations = 0.02,
        seed = 1)
    split_data = mochi_data.get_split_observation_data(
        fold = 1,
        seed = 1,
        training_resample = False)
    validation = split_data['validation']
    dense_X = torch.tensor(
        mochi_data.materialize_feature_matrix(
            row_indices = validation['row_indices'],
            dtype = np.uint8),
        dtype = torch.float32)
    sparse_X = build_sparse_feature_batch(
        mochi_data.feature_sparse_matrix[validation['row_indices']])
    model = MochiModel(
        input_shape = dense_X.shape[1],
        mask = validation['mask'].clone(),
        model_design = mochi_data.model_design.copy(),
        custom_transformations = mochi_data.custom_transformations,
        sos_architecture = [20],
        sos_outputlinear = False)
    custom_mask = validation['mask'].clone()
    custom_mask[0, 0, 0] = 0
    with torch.no_grad():
        dense_default = model(
            select = validation['select'],
            X = dense_X)
        sparse_default = model(
            select = validation['select'],
            X = sparse_X)
        dense_override = model(
            select = validation['select'],
            X = dense_X,
            mask = custom_mask)
        sparse_override = model(
            select = validation['select'],
            X = sparse_X,
            mask = custom_mask)
    assert torch.allclose(sparse_default, dense_default)
    assert torch.allclose(sparse_override, dense_override)

def test_MochiTask_init_no_MochiData_empty_directory(capsys):
    """Test MochiTask initialization when no MochiData nor saved MochiTask in directory supplied"""
    with pytest.raises(ValueError) as e_info:
        MochiTask(directory = str(Path(__file__).parent))
    captured = capsys.readouterr()
    assert captured.out == "Error: Saved models directory does not exist.\n" and e_info

def test_validate_model_flattens_wt_residual_arrays(monkeypatch):
    """Regression test for WT residual tracking with 2D prediction tensors."""
    class DummyModel:
        def __init__(self):
            self.mask = None
            self.training_history = {
                "val_loss": [],
                "additivetrait1_WT": [],
                "residual1_WT": [],
                "residual2_WT": []}
            self.additivetraits = [type("Trait", (), {"weight": torch.nn.Parameter(torch.tensor([[1.5]]))})()]
            self.model_design = pd.DataFrame({"phenotype": ["p1", "p2"]})
            self._call_count = 0

        def eval(self):
            return self

        def calculate_l1l2_norm(self):
            return torch.tensor(0.0), torch.tensor(0.0)

        def __call__(self, select, X, mask = None):
            self._call_count += 1
            if self._call_count == 1:
                return torch.zeros((1, 1), dtype = torch.float32)
            return torch.tensor([[0.4, 1.2]], dtype = torch.float32)

    class DummyLoader:
        dataset_len = 1

        def __len__(self):
            return 1

    monkeypatch.setattr(
        "pymochi.models.DevicePrefetchLoader",
        lambda dataloader, device: [(
            torch.zeros((1, 1), dtype = torch.long),
            torch.zeros((1, 1), dtype = torch.float32),
            torch.zeros((1, 1), dtype = torch.float32),
            torch.ones((1, 1), dtype = torch.float32))])

    model = DummyModel()
    MochiModel.validate_model(
        model,
        dataloader = DummyLoader(),
        loss_function = lambda pred, y, y_wt: torch.zeros((len(y),), dtype = torch.float32),
        device = torch.device("cpu"),
        data_WT = {
            "select": torch.zeros((1, 1), dtype = torch.long),
            "X": torch.zeros((1, 1), dtype = torch.float32),
            "y": torch.tensor([[1.0, 2.0]], dtype = torch.float32)})

    assert model.training_history["val_loss"] == [pytest.approx(0.0)]
    assert model.training_history["additivetrait1_WT"] == [pytest.approx(1.5)]
    assert model.training_history["residual1_WT"] == [pytest.approx(0.6)]
    assert model.training_history["residual2_WT"] == [pytest.approx(0.8)]

def test_get_additive_trait_weights_aggregates_interaction_positions():
    """Aggregate weights sort single and interaction reference positions numerically."""
    task = MochiTask.__new__(MochiTask)
    task.directory = ""
    feature_names = ["WT", "A2B", "A2B_A3B", "A2B_A10B", "A13B", "A13B_A54B"]
    task.data = type("Data", (), {
        "k_folds": 1,
        "model_design": pd.DataFrame({
            "trait": [[1]],
            "transformation": ["Identity"]}),
        "additive_trait_names": ["trait"],
        "get_feature_names": lambda self: pd.Index(feature_names)})()
    trait = torch.nn.Linear(len(feature_names), 1, bias = False)
    model = type("Model", (), {
        "metadata": type("Metadata", (), {"grid_search": False, "fold": 1})(),
        "additivetraits": [trait],
        "mask": torch.ones((1, 1, len(feature_names)))})()
    task.models = [model]

    aggregated_weights = task.get_additive_trait_weights(
        aggregate = True,
        save = False)

    assert aggregated_weights[0]["Pos_ref"].tolist() == [
        "2", "2_3", "2_10", "13", "13_54"]

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
    model_mask = mochi_task.data.get_mask_tensor(fold = 1)
    #Add 3 grid search models
    for i in range(3):
        mochi_task.models += [mochi_task.new_model(model_mask)]
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

def test_fit_best_handles_small_num_epochs_grid(monkeypatch):
    """Test fit_best still selects a model when grid-search epochs are very small."""
    create_dummy_task()
    mochi_task = MochiTask(directory = str(Path(__file__).parent / "temp"))
    model_mask = mochi_task.data.get_mask_tensor(fold = 1)
    recorded_fit = {}

    for batch_size, learn_rate, val_loss in [
        (64, 0.05, [2.0, 1.8, 1.6, 1.4, 1.2]),
        (128, 0.01, [2.0, 1.7, 1.5, 1.2, 0.2])]:
        mochi_task.models += [mochi_task.new_model(model_mask)]
        model = mochi_task.models[-1]
        model.metadata = MochiModelMetadata(
            fold = 1,
            seed = 1,
            grid_search = True,
            batch_size = batch_size,
            learn_rate = learn_rate,
            num_epochs = mochi_task.num_epochs,
            num_epochs_grid = 5,
            l1_regularization_factor = mochi_task.l1_regularization_factor,
            l2_regularization_factor = mochi_task.l2_regularization_factor,
            training_resample = True,
            early_stopping = True,
            scheduler_gamma = mochi_task.scheduler_gamma,
            scheduler_epochs = 10,
            loss_function_name = 'WeightedL1',
            sos_architecture = [20],
            sos_outputlinear = False)
        model.training_history['val_loss'] = val_loss

    def fake_fit(self, **kwargs):
        recorded_fit.update(kwargs)

    monkeypatch.setattr(MochiTask, "fit", fake_fit)

    mochi_task.fit_best(fold = 2, seed = 1)

    assert recorded_fit["fold"] == 2
    assert recorded_fit["seed"] == 1
    assert recorded_fit["batch_size"] == 128
    assert recorded_fit["learn_rate"] == 0.01

def test_parallel_mode_suppresses_project_directory_warning(tmp_path, monkeypatch, capsys):
    """Test phase-split parallel runs do not warn when reusing the project directory."""
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    monkeypatch.setenv("MOCHI_PARALLEL_MODE", "1")

    MochiProject(
        directory = str(project_dir),
        model_design = make_demo_model_design(),
        auto_run = False)

    captured = capsys.readouterr()
    assert "Warning: Project directory already exists." not in captured.out

def test_parallel_mode_suppresses_saved_models_overwrite_warning(monkeypatch, capsys):
    """Test phase-split parallel runs do not warn when overwriting saved_models."""
    create_dummy_task()
    capsys.readouterr()
    mochi_task = MochiTask(directory = str(Path(__file__).parent / "temp"))
    capsys.readouterr()
    monkeypatch.setenv("MOCHI_PARALLEL_MODE", "1")

    mochi_task.save(overwrite = True)

    captured = capsys.readouterr()
    assert "Warning: Saved models directory already exists." not in captured.out

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


def test_main_grid_search_phase_dispatches_project_method(tmp_path, monkeypatch):
    """Test CLI grid_search phase dispatches to the project helper without auto-running."""
    calls = []

    class FakeProject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["auto_run"], kwargs["directory"]))
            self.fix_weights = kwargs["fix_weights"]
            self.RT = kwargs["RT"]
            self.seq_position_offset = kwargs["seq_position_offset"]

        def run_grid_search_task(self, seed, fix_weights):
            calls.append(("grid_search", seed, fix_weights))

    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(mochi_main, "MochiProject", FakeProject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "grid_search",
        ])

    mochi_main.main()

    assert calls == [
        ("init", False, str(tmp_path / "phase_test")),
        ("grid_search", 1, {}),
    ]


def test_main_merge_grid_search_phase_dispatches_project_method(tmp_path, monkeypatch):
    """Test CLI merge_grid_search phase dispatches to the project helper."""
    calls = []

    class FakeProject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["auto_run"], kwargs["directory"]))
            self.fix_weights = kwargs["fix_weights"]
            self.RT = kwargs["RT"]
            self.seq_position_offset = kwargs["seq_position_offset"]

        def merge_grid_search_conditions(self, seed):
            calls.append(("merge_grid_search", seed))

    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(mochi_main, "MochiProject", FakeProject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "merge_grid_search",
        ])

    mochi_main.main()

    assert calls == [
        ("init", False, str(tmp_path / "phase_test")),
        ("merge_grid_search", 1),
    ]


def test_main_full_sparse_phase_dispatches_sparse_method(tmp_path, monkeypatch):
    """Test CLI full phase dispatches sparse runs to the sparse helper."""
    calls = []

    class FakeProject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["auto_run"], kwargs["directory"], kwargs["sparse_method"]))
            self.fix_weights = kwargs["fix_weights"]
            self.RT = kwargs["RT"]
            self.seq_position_offset = kwargs["seq_position_offset"]

        def run_sparse_sig_highestorder_step(self):
            calls.append(("sparse",))

        def run_full_task(self, seed, fix_weights):
            calls.append(("full", seed, fix_weights))

    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(mochi_main, "MochiProject", FakeProject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--sparse_method", "sig_highestorder_step",
        ])

    mochi_main.main()

    assert calls == [
        ("init", False, str(tmp_path / "phase_test"), "sig_highestorder_step"),
        ("sparse",),
    ]


def test_main_sparse_grid_phase_dispatches_stage_helper(tmp_path, monkeypatch):
    """Test CLI sparse split phases dispatch to sparse stage helpers."""
    calls = []

    class FakeProject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["auto_run"], kwargs["directory"], kwargs["sparse_method"]))
            self.fix_weights = kwargs["fix_weights"]
            self.RT = kwargs["RT"]
            self.seq_position_offset = kwargs["seq_position_offset"]

        def run_sparse_stage_grid_search(self, stage_index, fix_weights):
            calls.append(("sparse_grid", stage_index, fix_weights))

    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(mochi_main, "MochiProject", FakeProject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "sparse_grid_search",
            "--stage_index", "2",
            "--sparse_method", "sig_highestorder_step",
        ])

    mochi_main.main()

    assert calls == [
        ("init", False, str(tmp_path / "phase_test"), "sig_highestorder_step"),
        ("sparse_grid", 2, {}),
    ]


def test_main_sparse_merge_grid_phase_dispatches_stage_helper(tmp_path, monkeypatch):
    """Test CLI sparse grid-merge phase dispatches to the stage helper."""
    calls = []

    class FakeProject:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs["auto_run"], kwargs["directory"], kwargs["sparse_method"]))
            self.fix_weights = kwargs["fix_weights"]
            self.RT = kwargs["RT"]
            self.seq_position_offset = kwargs["seq_position_offset"]

        def merge_sparse_stage_grid_search(self, stage_index):
            calls.append(("sparse_merge_grid", stage_index))

    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(mochi_main, "MochiProject", FakeProject)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "sparse_merge_grid_search",
            "--stage_index", "2",
            "--sparse_method", "sig_highestorder_step",
        ])

    mochi_main.main()

    assert calls == [
        ("init", False, str(tmp_path / "phase_test"), "sig_highestorder_step"),
        ("sparse_merge_grid", 2),
    ]


def test_main_fit_best_phase_requires_fold(tmp_path, monkeypatch):
    """Test CLI fit_best phase requires an explicit fold argument."""
    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "fit_best",
        ])

    with pytest.raises(ValueError, match = "--fold is required when --phase fit_best"):
        mochi_main.main()


def test_main_sparse_phase_requires_stage_index(tmp_path, monkeypatch):
    """Test sparse split phases require an explicit stage index."""
    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "sparse_grid_search",
            "--sparse_method", "sig_highestorder_step",
        ])

    with pytest.raises(ValueError, match = "--stage_index is required for sparse split phases"):
        mochi_main.main()


def test_main_sparse_method_requires_full_phase(tmp_path, monkeypatch):
    """Test sparse runs reject split phases that bypass sparse pruning."""
    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--phase", "grid_search",
            "--sparse_method", "sig_highestorder_step",
        ])

    with pytest.raises(ValueError, match = "--sparse_method is only supported when --phase full"):
        mochi_main.main()


def test_get_sparse_stage_settings_final_stage_disables_l1(tmp_path):
    """Test the final sparse stage runs without L1 regularization."""
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        max_interaction_order = 2,
        l1_regularization_factor = "0.01,0.001,0.0001",
        auto_run = False)

    stage_settings = project.get_sparse_stage_settings(stage_index = 4)

    assert stage_settings["order"] == -1
    assert stage_settings["l1_regularization_factor"] == 0
    assert stage_settings["save_report"] is True
    assert stage_settings["save_weights"] is True


def test_main_requires_at_least_three_folds(tmp_path, monkeypatch):
    """Test CLI rejects cross-validation runs with fewer than three folds."""
    monkeypatch.setattr(mochi_main, "configure_logging", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_mochi.py",
            "--model_design", str(Path(__file__).parent.parent / "data/model_design.txt"),
            "--output_directory", str(tmp_path),
            "--project_name", "phase_test",
            "--k_folds", "2",
        ])

    with pytest.raises(ValueError, match = "--k_folds must be at least 3"):
        mochi_main.main()


def test_MochiTask_fit_grid_search_without_explicit_test_split(tmp_path, monkeypatch):
    """Test grid-search fitting works when validation consumes all non-training folds."""
    monkeypatch.setenv("MOCHI_DEVICE", "cpu")
    mochi_data = MochiData(
        model_design = make_demo_model_design(),
        max_interaction_order = 1,
        k_folds = 2,
        validation_factor = 2,
        downsample_observations = 0.05,
        seed = 1)
    split_data = mochi_data.get_split_observation_data(
        fold = 1,
        seed = 1,
        training_resample = False)
    assert "test" not in split_data

    mochi_task = MochiTask(
        directory = str(tmp_path / "task_1"),
        data = mochi_data,
        batch_size = "32",
        learn_rate = "0.05",
        num_epochs = 1,
        num_epochs_grid = 1)
    mochi_task.fit(
        fold = 1,
        seed = 1,
        grid_search = True,
        batch_size = 32,
        learn_rate = 0.05,
        num_epochs = 1,
        num_epochs_grid = 1,
        epoch_status = 1)
    assert len(mochi_task.models) == 1


def test_run_grid_search_task_reuses_existing_saved_models(tmp_path, monkeypatch):
    """Test rerunning grid_search phase reuses saved task artifacts."""
    calls = []

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            calls.append(("init", directory, "data" in kwargs))
            self.directory = directory
            self.models = [type("ModelInfo", (), {
                "metadata": type("Metadata", (), {"grid_search": True, "fold": 1})()
            })()]

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        auto_run = False)
    task_directory = Path(project.get_task_directory(1)) / "saved_models"
    task_directory.mkdir(parents = True)

    def fail_build(*args, **kwargs):
        raise AssertionError("build_task should not be called when reusing artifacts")

    monkeypatch.setattr(project, "build_task", fail_build)

    reused_task = project.run_grid_search_task(seed = 1)

    assert reused_task.models[0].metadata.grid_search is True
    assert calls == [("init", project.get_task_directory(1), False)]


def test_run_fit_fold_task_reuses_existing_fold_models(tmp_path, monkeypatch):
    """Test rerunning fit_best phase reuses saved fold artifacts."""
    calls = []

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            calls.append(directory)
            self.directory = directory
            if str(directory).endswith("fold_2"):
                self.models = [type("ModelInfo", (), {
                    "metadata": type("Metadata", (), {"grid_search": False, "fold": 2})()
                })()]
            else:
                self.models = []

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        auto_run = False)
    fold_directory = Path(project.get_fold_directory(1, 2)) / "saved_models"
    fold_directory.mkdir(parents = True)

    reused_task = project.run_fit_fold_task(seed = 1, fold = 2)

    assert reused_task.models[0].metadata.fold == 2
    assert calls == [project.get_fold_directory(1, 2)]


def test_merge_grid_search_conditions_combines_condition_models(tmp_path, monkeypatch):
    """Test merging grid condition directories rebuilds the canonical task artifacts."""
    finalized = []

    def make_model(grid_search, fold, tag):
        return type("ModelInfo", (), {
            "metadata": type("Metadata", (), {"grid_search": grid_search, "fold": fold})(),
            "tag": tag,
        })()

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            self.directory = directory
            self.models = []
            self.data = object()
            self.custom_transformations = None
            if str(directory).endswith("grid_condition_1"):
                self.models = [make_model(True, 1, "a")]
            elif str(directory).endswith("grid_condition_2"):
                self.models = [make_model(True, 1, "b")]
            elif str(directory).endswith("task_1"):
                self.models = []

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        auto_run = False)

    Path(project.get_grid_condition_directory(1, 1), "saved_models").mkdir(parents = True)
    Path(project.get_grid_condition_directory(1, 2), "saved_models").mkdir(parents = True)

    def fake_finalize(mochi_task, **kwargs):
        finalized.append((mochi_task.directory, [model.tag for model in mochi_task.models], kwargs))
        return mochi_task

    monkeypatch.setattr(project, "finalize_task_outputs", fake_finalize)

    merged_task = project.merge_grid_search_conditions(seed = 1)

    assert [model.tag for model in merged_task.models] == ["a", "b"]
    assert finalized == [
        (
            project.get_task_directory(1),
            ["a", "b"],
            {"save_model": True, "save_report": False, "save_weights": False},
        )
    ]


def test_merge_sparse_grid_search_conditions_discovers_nextflow_condition_layout(tmp_path, monkeypatch):
    """Test sparse grid merge discovers task outputs saved under Nextflow condition directories."""
    finalized = []

    def make_model(grid_search, fold, tag):
        return type("ModelInfo", (), {
            "metadata": type("Metadata", (), {"grid_search": grid_search, "fold": fold})(),
            "tag": tag,
        })()

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            self.directory = directory
            self.models = []
            self.data = object()
            self.custom_transformations = None
            if str(directory).endswith("stage_1/grid_search/condition_1/mochi_project/task_1"):
                self.models = [make_model(True, 1, "a")]
            elif str(directory).endswith("stage_1/grid_search/condition_2/mochi_project/task_1"):
                self.models = [make_model(True, 1, "b")]
            elif str(directory).endswith("mochi_project/task_1"):
                self.models = []

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    run_root = tmp_path / "run"
    run_root.mkdir()
    project = MochiProject(
        directory = str(run_root / "mochi_project"),
        model_design = make_demo_model_design(),
        auto_run = False)

    (run_root / "stage_1" / "grid_search" / "condition_1" / "mochi_project" / "task_1" / "saved_models").mkdir(parents = True)
    (run_root / "stage_1" / "grid_search" / "condition_2" / "mochi_project" / "task_1" / "saved_models").mkdir(parents = True)

    def fake_finalize(mochi_task, **kwargs):
        finalized.append((mochi_task.directory, [model.tag for model in mochi_task.models], kwargs))
        return mochi_task

    monkeypatch.setattr(project, "finalize_task_outputs", fake_finalize)

    merged_task = project.merge_sparse_stage_grid_search(stage_index = 1)

    assert [model.tag for model in merged_task.models] == ["a", "b"]
    assert finalized == [
        (
            project.get_task_directory(1),
            ["a", "b"],
            {"save_model": True, "save_report": False, "save_weights": False},
        )
    ]


def test_merge_parallel_task_skips_missing_fold_artifacts(tmp_path, monkeypatch):
    """Test parallel fold merge succeeds when only some folds produced outputs."""
    finalized = []

    def make_model(grid_search, fold, tag):
        return type("ModelInfo", (), {
            "metadata": type("Metadata", (), {"grid_search": grid_search, "fold": fold})(),
            "tag": tag,
        })()

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            self.directory = directory
            self.models = []
            self.data = object()
            self.custom_transformations = None
            if str(directory).endswith("task_1"):
                self.models = [make_model(True, 1, "grid")]
            elif str(directory).endswith("fold_1"):
                self.models = [make_model(False, 1, "fold-1")]

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        k_folds = 3,
        auto_run = False)

    Path(project.get_task_directory(1), "saved_models").mkdir(parents = True)
    Path(project.get_fold_directory(1, 1), "saved_models").mkdir(parents = True)

    def fake_finalize(mochi_task, **kwargs):
        finalized.append((mochi_task.directory, [model.tag for model in mochi_task.models], kwargs))
        return mochi_task

    monkeypatch.setattr(project, "finalize_task_outputs", fake_finalize)

    merged_task = project.merge_parallel_task(seed = 1)

    assert [model.tag for model in merged_task.models] == ["grid", "fold-1"]
    assert finalized == [
        (
            project.get_task_directory(1),
            ["grid", "fold-1"],
            {
                "RT": None,
                "seq_position_offset": 0,
                "save_model": True,
                "save_report": True,
                "save_weights": True,
            },
        )
    ]


def test_merge_sparse_stage_skips_missing_fold_artifacts(tmp_path, monkeypatch):
    """Test sparse stage merge succeeds when only some folds produced outputs."""
    finalized = []

    def make_model(grid_search, fold, tag):
        return type("ModelInfo", (), {
            "metadata": type("Metadata", (), {"grid_search": grid_search, "fold": fold})(),
            "tag": tag,
        })()

    class FakeTask:
        def __init__(self, directory = None, **kwargs):
            self.directory = directory
            self.models = []
            self.data = object()
            self.custom_transformations = None
            if str(directory).endswith("task_1"):
                self.models = [make_model(True, 1, "grid")]
            elif str(directory).endswith("fold_1"):
                self.models = [make_model(False, 1, "fold-1")]

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakeTask)
    project = MochiProject(
        directory = str(tmp_path / "project"),
        model_design = make_demo_model_design(),
        k_folds = 3,
        sparse_method = "sig_highestorder_step",
        auto_run = False)

    Path(project.get_task_directory(1), "saved_models").mkdir(parents = True)
    Path(project.get_fold_directory(1, 1), "saved_models").mkdir(parents = True)

    def fake_finalize(mochi_task, **kwargs):
        finalized.append((mochi_task.directory, [model.tag for model in mochi_task.models], kwargs))
        return mochi_task

    monkeypatch.setattr(project, "finalize_task_outputs", fake_finalize)

    merged_task = project.merge_sparse_stage(stage_index = 1)

    assert [model.tag for model in merged_task.models] == ["grid", "fold-1"]
    assert finalized == [
        (
            project.get_task_directory(1),
            ["grid", "fold-1"],
            {
                "RT": None,
                "seq_position_offset": 0,
                "save_model": True,
                "save_report": False,
                "save_weights": True,
            },
        )
    ]


def test_build_sparse_stage_inputs_uses_canonical_previous_stage_from_nextflow_layout(tmp_path, monkeypatch):
    """Test sparse stage inputs load the previous merged task from the canonical Nextflow project root."""
    calls = []

    class FakePrevTask:
        def __init__(self, directory = None, **kwargs):
            calls.append(directory)
            self.directory = directory
            self.data = type(
                "Data",
                (),
                {"additive_trait_names": ["trait_a"]})()

        def get_additive_trait_weights(self, save = False):
            return [pd.DataFrame({
                "id": ["WT", "1"],
                "Pos": [None, "1"],
                "mean": [0.0, 1.0],
                "ci95": [0.0, 0.1],
            })]

    monkeypatch.setattr(mochi_project_module, "MochiTask", FakePrevTask)
    monkeypatch.setenv("MOCHI_PARALLEL_MODE", "1")
    run_root = tmp_path / "run"
    canonical_project = run_root / "mochi_project"
    (canonical_project / "task_1" / "saved_models").mkdir(parents = True)
    condition_project = run_root / "stage_2" / "grid_search" / "condition_2" / "mochi_project"
    condition_project.mkdir(parents = True)

    project = MochiProject(
        directory = str(condition_project),
        model_design = make_demo_model_design(),
        max_interaction_order = 2,
        auto_run = False)

    mochi_data_args, mochi_task_args, stage_settings = project.build_sparse_stage_inputs(stage_index = 2)

    assert calls == [str(canonical_project / "task_1")]
    assert mochi_task_args["directory"] == str(condition_project / "task_2")
    assert stage_settings["order"] == 1


def test_mochi_task_save_creates_missing_parent_directories(tmp_path):
    """Test saving a loaded task succeeds after retargeting to a nested new directory."""
    create_dummy_task()
    mochi_task = MochiTask(directory = str(Path(__file__).parent / "temp"))
    mochi_task.directory = str(tmp_path / "nested" / "fold_7")

    mochi_task.save(overwrite = True)

    assert (tmp_path / "nested" / "fold_7" / "saved_models").is_dir()
