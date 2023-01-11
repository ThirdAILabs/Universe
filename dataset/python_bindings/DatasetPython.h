#pragma once

#include <dataset/src/Datasets.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module);

/**
 * Checks whether the given bolt dataset and dense 2d matrix
 * have the same values. For testing purposes only.
 */
bool denseBoltDatasetMatchesDenseMatrix(
    BoltDataset& dataset, std::vector<std::vector<float>>& matrix);

/**
 * Checks whether the given bolt dataset represents a permutation of
 * the rows of the given dense 2d matrix. Assumes that each row of
 * the matrix is 1-dimensional; only has one element.
 * For testing purposes only.
 */
bool denseBoltDatasetIsPermutationOfDenseMatrix(
    BoltDataset& dataset, std::vector<std::vector<float>>& matrix);

/**
 * Checks whether the given bolt datasets have the same values.
 * For testing purposes only.
 */
bool denseBoltDatasetsAreEqual(BoltDataset& dataset1, BoltDataset& dataset2);

}  // namespace thirdai::dataset::python
