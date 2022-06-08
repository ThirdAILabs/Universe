#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/batch_types/DenseBatch.h>
#include <dataset/src/batch_types/SparseBatch.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/core/BlockBatchProcessor.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace thirdai::dataset::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

void createDatasetSubmodule(py::module_& module);

InMemoryDataset<SparseBatch> loadSVMDataset(const std::string& filename,
                                            uint32_t batch_size);

InMemoryDataset<DenseBatch> loadCSVDataset(const std::string& filename,
                                           uint32_t batch_size,
                                           std::string delimiter);

py::tuple loadBoltSvmDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size,
                                    bool softmax_for_multiclass = true);

py::tuple loadBoltCsvDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size, char delimiter);

py::tuple loadClickThroughDatasetWrapper(const std::string& filename,
                                         uint32_t batch_size,
                                         uint32_t num_dense_features,
                                         uint32_t num_categorical_features,
                                         bool sparse_labels);

// https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html?highlight=numpy#arrays
// for explanation of why we do py::array::c_style and py::array::forcecast.
// Ensures array is an array of floats in dense row major order.
SparseBatch wrapNumpyIntoSparseData(
    const std::vector<py::array_t<
        float, py::array::c_style | py::array::forcecast>>& sparse_values,
    const std::vector<
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast>>&
        sparse_indices,
    uint64_t starting_id);

DenseBatch wrapNumpyIntoDenseBatch(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& data,
    uint64_t starting_id);

InMemoryDataset<DenseBatch> denseInMemoryDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        labels,
    uint32_t batch_size, uint64_t starting_id);

InMemoryDataset<SparseBatch> sparseInMemoryDatasetFromNumpy(
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_idxs,
    const py::array_t<float, py::array::c_style | py::array::forcecast>& x_vals,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        x_offsets,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_idxs,
    const py::array_t<uint32_t, py::array::c_style | py::array::forcecast>&
        y_offsets,
    uint32_t batch_size, uint64_t starting_id);

BoltDatasetPtr denseBoltDatasetFromNumpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>&
        examples,
    uint32_t batch_size);

BoltDatasetPtr sparseBoltDatasetFromNumpy(const NumpyArray<uint32_t>& indices,
                                          const NumpyArray<float>& values,
                                          const NumpyArray<uint32_t>& offsets,
                                          uint32_t batch_size);

BoltDatasetPtr categoricalLabelsFromNumpy(const NumpyArray<uint32_t>& labels,
                                          uint32_t batch_size);

/*
 * This function takes a single sentence, and parses it into an sparse
 * vector of features. Right now it only supports the following parsing:
 * unigram tokenizer + murmurhash tokens into indices.
 * This function returns a tuple of python arrays, where the first array is
 * the indices of the features in the dataset, and the second array is the
 * values.
 */
std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>>
parseSentenceToSparseArray(const std::string& sentence, uint32_t seed,
                           uint32_t dimension);

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

class PyBlockBatchProcessor : public BlockBatchProcessor {
 public:
  PyBlockBatchProcessor(std::vector<std::shared_ptr<Block>> input_blocks,
                        std::vector<std::shared_ptr<Block>> target_blocks,
                        uint32_t output_batch_size)
      : BlockBatchProcessor(std::move(input_blocks), std::move(target_blocks),
                            output_batch_size) {}

  /**
   * Just like the original processBatch method but GIL is released
   * so we can process batches while the next input rows are
   * processed in python.
   */
  void processBatchPython(std::vector<std::vector<std::string>>& batch) {
    py::gil_scoped_release release;
    processBatch(batch);
  }
};

}  // namespace thirdai::dataset::python