import pytest
from thirdai._thirdai import dataset_internal
import random
from thirdai import dataset
import os

def make_random_2d_matrix(n_rows, n_cols):
    return [[random.random() for col in range(n_cols)] for row in range(n_rows)]

def make_string_2d_matrix(float_2d_matrix):
    return [[str(col) for col in row] for row in float_2d_matrix]

def make_csv_2d_matrix(float_2d_matrix, file):
    with open(file, 'w') as out:
        out.write('\n'.join([','.join([str(col) for col in row]) for row in float_2d_matrix]))

@pytest.mark.unit
def test_batch_processor():
    """Checks that batch processor python binding works.
    We only test one case: dense input and target vectors,
    no shuffling, since we exhaustively tested the other 
    cases in C++.
    """

    n_rows = 1000
    n_cols = 3
    batch_size = 256

    float_2d_mat = make_random_2d_matrix(n_rows, n_cols)
    str_2d_mat = make_string_2d_matrix(float_2d_mat)
    
    processor = dataset_internal.BatchProcessor(
        [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)],
        [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)],
        batch_size
    )

    processor.process_batch(str_2d_mat)
    input_ds, target_ds = processor.export_in_memory_dataset()
    
    assert dataset_internal.dense_bolt_dataset_matches_dense_matrix(input_ds, float_2d_mat)
    assert dataset_internal.dense_bolt_dataset_matches_dense_matrix(target_ds, float_2d_mat)

@pytest.mark.unit
def test_loader_no_shuffle():
    """Checks that data loader works without shuffling; 
    produces the appropriate bolt dataset given a CSV file.
    
    We only test one case: dense input and target vectors, 
    since we exhaustively tested batch processor
    in C++.
    """

    n_rows = 1000
    n_cols = 3
    batch_size = 256

    float_2d_mat = make_random_2d_matrix(n_rows, n_cols)
    csv_file = '2d_matrix_core_dataset_test.csv'
    make_csv_2d_matrix(float_2d_mat, csv_file)

    loader = dataset.Loader(
        source=dataset.sources.LocalFileSystem(csv_file), 
        parser=dataset.parsers.CsvIterable(), 
        schema=dataset.Schema(
            [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)],
            [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)]), 
        batch_size=batch_size)

    input_ds, target_ds = loader.processInMemory()

    assert dataset_internal.dense_bolt_dataset_matches_dense_matrix(input_ds, float_2d_mat)
    assert dataset_internal.dense_bolt_dataset_matches_dense_matrix(target_ds, float_2d_mat)

    os.remove(csv_file)   

@pytest.mark.unit
def test_loader_shuffle():
    """Checks that data loader works with shuffling; 
    produces the appropriate bolt dataset given a CSV file.
    
    We test two cases: seeded and unseeded.
    In both cases, our test input matrix have one-dimensional
    rows for the sake of simplicity.
    """

    n_rows = 1000
    n_cols = 1
    batch_size = 256

    float_2d_mat = make_random_2d_matrix(n_rows, n_cols)
    csv_file = '2d_matrix_core_dataset_test.csv'
    make_csv_2d_matrix(float_2d_mat, csv_file)

    loader = dataset.Loader(
        source=dataset.sources.LocalFileSystem(csv_file), 
        parser=dataset.parsers.CsvIterable(), 
        schema=dataset.Schema(
            [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)],
            [dataset_internal.MockBlock(column=i, dense=True) for i in range(n_cols)]), 
        batch_size=batch_size)

    input_ds_unshuf, target_ds_unshuf = loader.processInMemory()
    loader.shuffle()
    input_ds_shuf_unseed_1, target_ds_shuf_unseed_1 = loader.processInMemory()
    loader.shuffle()
    input_ds_shuf_unseed_2, target_ds_shuf_unseed_2 = loader.processInMemory()
    loader.shuffle(10)
    input_ds_shuf_seed_1, target_ds_shuf_seed_1 = loader.processInMemory()
    loader.shuffle(10)
    input_ds_shuf_seed_2, target_ds_shuf_seed_2 = loader.processInMemory()

    # All of the shuffled datasets must be a permutation of the input data.
    assert dataset_internal.dense_bolt_dataset_is_permutation_of_dense_matrix(input_ds_shuf_unseed_1, float_2d_mat)
    assert dataset_internal.dense_bolt_dataset_is_permutation_of_dense_matrix(input_ds_shuf_unseed_2, float_2d_mat)
    assert dataset_internal.dense_bolt_dataset_is_permutation_of_dense_matrix(input_ds_shuf_seed_1, float_2d_mat)
    assert dataset_internal.dense_bolt_dataset_is_permutation_of_dense_matrix(input_ds_shuf_seed_2, float_2d_mat)
    
    # For each of the processed datasets above, input must be the same as target; 
    # they must follow the same permutation
    assert dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_unseed_1, target_ds_shuf_unseed_1)
    assert dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_unseed_2, target_ds_shuf_unseed_2)
    assert dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_seed_1, target_ds_shuf_seed_1)
    assert dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_seed_2, target_ds_shuf_seed_2)
    
    # To check that the shuffled datasets are indeed shuffled, make sure that they are not equal to the
    # unshuffled dataset.
    assert (not dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_unseed_1, input_ds_unshuf))
    assert (not dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_unseed_2, input_ds_unshuf))
    assert (not dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_seed_1, input_ds_unshuf))
    assert (not dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_seed_2, input_ds_unshuf))
    
    # To check that seeding works as expected, that is, unseeded shuffling is expected to give different
    # shuffling orders while shuffling with the same seed should give the same shuffling order, check that 
    # unseeded_1 != unseeded_2 and seeded_1 == seeded_2
    assert (not dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_unseed_1, input_ds_shuf_unseed_2))
    assert dataset_internal.dense_bolt_datasets_are_equal(input_ds_shuf_seed_1, input_ds_shuf_seed_2)
    
    os.remove(csv_file)   

# test_batch_processor()
# test_loader_no_shuffle()
test_loader_shuffle()