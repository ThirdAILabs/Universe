#pragma once

#include <hashing/src/MurmurHash.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/bolt_datasets/StreamingDataset.h>
#include <dataset/src/bolt_datasets/batch_processors/MaskedSentenceBatchProcessor.h>
#include <dataset/src/core/BlockBatchProcessor.h>
#include <pybind11/cast.h>
#include <pybind11/gil.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <memory>

namespace py = pybind11;

namespace thirdai::dataset::python {

void createDatasetSubmodule(py::module_& module);

py::tuple loadBoltSvmDatasetWrapper(const std::string& filename,
                                    uint32_t batch_size,
                                    bool softmax_for_multiclass = true);

py::tuple loadClickThroughDatasetWrapper(const std::string& filename,
                                         uint32_t batch_size,
                                         uint32_t num_dense_features,
                                         uint32_t num_categorical_features,
                                         bool sparse_labels);

/*
 * This function takes a single sentence, and parses it into an sparse
 * vector of features. Right now it only supports the following parsing:
 * unigram tokenizer + murmurhash tokens into indices.
 * This function returns a tuple of python arrays, where the first array is
 * the indices of the features in the dataset, and the second array is the
 * values.
 */
std::tuple<py::array_t<uint32_t>, py::array_t<uint32_t>>
parseSentenceToUnigramsPython(const std::string& sentence, uint32_t dimension);

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
                        uint32_t output_batch_size, size_t est_num_elems)
      : BlockBatchProcessor(std::move(input_blocks), std::move(target_blocks),
                            output_batch_size, est_num_elems) {}

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

using MLMDatasetPtr = std::shared_ptr<InMemoryDataset<MaskedSentenceBatch>>;

class MLMDatasetLoader {
 public:
  explicit MLMDatasetLoader(uint32_t pairgram_range)
      : _batch_processor(
            std::make_shared<MaskedSentenceBatchProcessor>(pairgram_range)) {}

  py::tuple load(const std::string& filename, uint32_t batch_size) {
    auto data_loader =
        std::make_shared<dataset::SimpleFileDataLoader>(filename, batch_size);

    auto dataset = std::make_shared<dataset::StreamingDataset<
        bolt::BoltBatch, thirdai::dataset::BoltTokenBatch, bolt::BoltBatch>>(
        data_loader, _batch_processor);

    auto [data, masked_indices, labels] = dataset->loadInMemory();

    return py::make_tuple(py::cast(data), py::cast(masked_indices),
                          py::cast(labels));
  }

 private:
  std::shared_ptr<MaskedSentenceBatchProcessor> _batch_processor;
};

}  // namespace thirdai::dataset::python