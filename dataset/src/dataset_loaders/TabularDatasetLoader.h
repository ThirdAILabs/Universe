#pragma once

#include "DatasetLoader.h"
#include <dataset/src/Datasets.h>
#include <dataset/src/ShuffleBatchBuffer.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <stdexcept>

namespace thirdai::dataset {

class TabularDatasetLoader final : public DatasetLoader {
 public:
  using DatasetLoader::loadInMemory;

  TabularDatasetLoader(std::shared_ptr<dataset::DataSource> data_source,
                       dataset::BatchProcessorPtr batch_processor, bool shuffle,
                       DatasetShuffleConfig config = DatasetShuffleConfig());

  // TODO(Josh): Can we get rid of this constructor?
  TabularDatasetLoader(const std::shared_ptr<DataSource>& source,
                       std::vector<std::shared_ptr<Block>> input_blocks,
                       std::vector<std::shared_ptr<Block>> label_blocks,
                       bool shuffle = false,
                       DatasetShuffleConfig config = DatasetShuffleConfig(),
                       bool has_header = false, char delimiter = ',',
                       bool parallel = true)
      : TabularDatasetLoader(
            source,
            std::make_shared<GenericBatchProcessor>(
                std::move(input_blocks), std::move(label_blocks), has_header,
                delimiter, parallel),
            shuffle, config) {}

  std::optional<std::pair<InputDatasets, LabelDataset>> loadInMemory(
      uint64_t max_in_memory_batches) final;

  void restart() final;

  // TODO(Josh): This should be private, but okay for now
  std::optional<std::vector<BoltBatch>> nextBatchVector();

  uint32_t getInputDim() {
    auto dimensions = _batch_processor->getDimensions();
    if (!dimensions.has_value()) {
      throw std::runtime_error(
          "Cannot get the input dimension of this tabular dataset loader's "
          "batch processor.");
    }
    return dimensions->at(0);
  }

  uint32_t getLabelDim() {
    auto dimensions = _batch_processor->getDimensions();
    if (!dimensions.has_value()) {
      throw std::runtime_error(
          "Cannot get the input dimension of this tabular dataset loader's "
          "batch processor.");
    }
    // TODO(Josh): Again, this is assuming we have one input and one label
    // dataset
    return dimensions->at(1);
  }

 private:
  void prefillShuffleBuffer();

  // Returns whether data source is not exhausted
  bool addNextBatchToBuffer();

  std::shared_ptr<DataSource> _data_source;
  std::shared_ptr<BatchProcessor> _batch_processor;
  uint32_t _max_batch_size;

  // Even if the value of _shuffle is false, we still use a ShuffleBatchBuffer,
  // since we pass in the value of _shuffle.
  // TODO(Josh/Geordie): This is a bit confusing, if we aren't shuffling we
  // probably shouldn't use a ShuffleBatchBuffer
  bool _shuffle;
  uint32_t _batch_buffer_size;
  ShuffleBatchBuffer _buffer;
};

using TabularDatasetLoaderPtr = std::unique_ptr<TabularDatasetLoader>;

}  // namespace thirdai::dataset