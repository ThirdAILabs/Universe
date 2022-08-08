#pragma once

#include "DataLoader.h"
#include "ShuffleBatchBuffer.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <cstddef>
#include <memory>

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

class StreamingGenericDatasetLoader
    : public StreamingDataset<bolt::BoltBatch, bolt::BoltBatch> {
 public:
  /**
   * This constructor accepts a pointer to any data loader.
   */
  StreamingGenericDatasetLoader(
      std::shared_ptr<DataLoader> loader,
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig(),
      bool has_header = false, char delimiter = ',')
      : StreamingGenericDatasetLoader(
            std::move(loader),
            std::make_shared<GenericBatchProcessor>(std::move(input_blocks),
                                                    std::move(label_blocks),
                                                    has_header, delimiter),
            shuffle, config) {}

  /**
   * This constructor does not accept a data loader and
   * defaults to a simple file loader.
   */
  StreamingGenericDatasetLoader(
      std::string filename, std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, uint32_t batch_size,
      bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig(),
      bool has_header = false, char delimiter = ',')
      : StreamingGenericDatasetLoader(
            std::make_shared<SimpleFileDataLoader>(filename, batch_size),
            std::move(input_blocks), std::move(label_blocks), shuffle, config,
            has_header, delimiter) {}

  std::optional<std::tuple<bolt::BoltBatch, bolt::BoltBatch>> nextBatchTuple()
      final {
    if (_buffer.empty()) {
      prefillShuffleBuffer();
    }
    addNextBatchToBuffer();

    return _buffer.popBatch();
  }

  std::tuple<BoltDatasetPtr, BoltDatasetPtr> loadInMemory() final {
    while (addNextBatchToBuffer()) {
    }
    auto [input_batches, label_batches] = _buffer.exportBuffer();
    return {std::make_shared<BoltDataset>(std::move(input_batches)),
            std::make_shared<BoltDataset>(std::move(label_batches))};
  }

  uint32_t getInputDim() { return _processor->getInputDim(); }

  uint32_t getLabelDim() { return _processor->getLabelDim(); }

 private:
  /**
   * Private constructor that takes in a pointer to
   * GenericBatchProcessor so we can pass this pointer to both
   * the base class constructor and this class's member variable
   * initializer.
   */
  StreamingGenericDatasetLoader(
      std::shared_ptr<DataLoader> loader,
      std::shared_ptr<GenericBatchProcessor> processor, bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig())
      : StreamingDataset(loader, processor),
        _processor(std::move(processor)),
        _buffer(config.seed, loader->getMaxBatchSize()),
        _shuffle(shuffle),
        _batch_buffer_size(config.n_batches) {}

  void prefillShuffleBuffer() {
    size_t n_prefill_batches = _batch_buffer_size - 1;
    size_t n_added = 0;
    while (n_added < n_prefill_batches && addNextBatchToBuffer()) {
      n_added++;
    }
  }

  bool addNextBatchToBuffer() {
    auto batch =
        StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::nextBatchTuple();
    if (batch) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
      return true;
    }
    return false;
  }

  std::shared_ptr<GenericBatchProcessor> _processor;
  ShuffleBatchBuffer _buffer;
  bool _shuffle;
  size_t _batch_buffer_size;
};

}  // namespace thirdai::dataset