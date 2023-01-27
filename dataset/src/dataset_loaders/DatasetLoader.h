#pragma once

#include <dataset/src/Datasets.h>
#include <dataset/src/ShuffleBatchBuffer.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <stdexcept>

namespace thirdai::dataset {

struct DatasetShuffleConfig {
  DatasetShuffleConfig() : n_batches(1000), seed(time(NULL)) {}

  explicit DatasetShuffleConfig(size_t n_batches_in_buffer)
      : n_batches(n_batches_in_buffer), seed(time(NULL)) {}

  DatasetShuffleConfig(size_t n_batches_in_buffer, uint32_t seed)
      : n_batches(n_batches_in_buffer), seed(seed) {}

  size_t n_batches;
  uint32_t seed;
};

using InputDatasets = std::vector<dataset::BoltDatasetPtr>;
using LabelDataset = dataset::BoltDatasetPtr;
class DatasetLoader final {
 public:
  DatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                dataset::BatchProcessorPtr batch_processor, bool shuffle,
                DatasetShuffleConfig shuffle_config = DatasetShuffleConfig());

  // TODO(Josh/Geordie/Nick/David): We should generalize these next two load
  // methods to return a vector of BoltDatasets, and figure out which are
  // inputs and which are labels in UDT

  std::pair<InputDatasets, LabelDataset> loadInMemory(bool verbose = true);

  std::optional<std::pair<InputDatasets, LabelDataset>> streamInMemory(
      size_t num_batches, bool verbose = true);

  void restart();

  uint32_t getInputDim() {
    // TODO(Josh): This is assuming we have one input and one label
    // dataset
    return _batch_processor->getDimensions().at(0);
  }

  uint32_t getLabelDim() {
    // TODO(Josh): Again, this is assuming we have one input and one label
    // dataset
    return _batch_processor->getDimensions().at(1);
  }

 private:
  // Adds batches to the buffer until the data source is finished or the buffer
  // reaches the passed in fill_size
  void fillShuffleBuffer(size_t fill_size);

  DataSourcePtr _data_source;
  std::shared_ptr<BatchProcessor> _batch_processor;

  // Even if the value of _shuffle is false, we still use a ShuffleBatchBuffer,
  // since we pass in the value of _shuffle.
  // TODO(Josh/Geordie): This is a bit confusing, if we aren't shuffling we
  // probably shouldn't use a ShuffleBatchBuffer
  bool _shuffle;
  // We try to ensure at least this many batches are in the buffer and shuffled
  // when we  return shuffled values
  size_t _batch_buffer_size;
  ShuffleBatchBuffer _buffer;
};

using DatasetLoaderPtr = std::unique_ptr<DatasetLoader>;

}  // namespace thirdai::dataset