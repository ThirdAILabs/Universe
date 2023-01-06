#pragma once

#include "DataSource.h"
#include "ShuffleBatchBuffer.h"
#include "StreamingDataset.h"
#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>

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
    : public StreamingDataset {
 public:
  /**
   * This constructor accepts a pointer to any data source.
   */
  StreamingGenericDatasetLoader(
      const std::shared_ptr<DataSource>& source,
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig(),
      bool has_header = false, char delimiter = ',', bool parallel = true)
      : StreamingGenericDatasetLoader(
            source,
            std::make_shared<GenericBatchProcessor>(
                std::move(input_blocks), std::move(label_blocks), has_header,
                delimiter, parallel),
            shuffle, config) {}

  /**
   * This constructor accepts a generic batch processor instead of blocks.
   */
  StreamingGenericDatasetLoader(
      std::string filename, std::shared_ptr<GenericBatchProcessor> processor,
      uint32_t batch_size, bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig())
      : StreamingGenericDatasetLoader(
            std::make_shared<SimpleFileDataSource>(filename, batch_size),
            std::move(processor), shuffle, config) {}

  /**
   * Constructor that takes in a pointer to GenericBatchProcessor so we can pass
   * this pointer to both the base class constructor and this class's member
   * variable initializer.
   */
  StreamingGenericDatasetLoader(
      const std::shared_ptr<DataSource>& source,
      std::shared_ptr<GenericBatchProcessor> processor, bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig())
      : StreamingDataset(source, processor),
        _processor(std::move(processor)),
        _buffer(config.seed, source->getMaxBatchSize()),
        _shuffle(shuffle),
        _batch_buffer_size(config.n_batches) {}

  /**
   * This constructor does not accept a data source and
   * defaults to a simple file loader.
   */
  StreamingGenericDatasetLoader(
      std::string filename, std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, uint32_t batch_size,
      bool shuffle = false,
      DatasetShuffleConfig config = DatasetShuffleConfig(),
      bool has_header = false, char delimiter = ',', bool parallel = true)
      : StreamingGenericDatasetLoader(
            std::make_shared<SimpleFileDataSource>(filename, batch_size),
            std::move(input_blocks), std::move(label_blocks), shuffle, config,
            has_header, delimiter, parallel) {}

  std::optional<std::vector<BoltBatch>> nextBatchVector() final {
    if (_buffer.empty()) {
      prefillShuffleBuffer();
    }
    addNextBatchToBuffer();

    return _buffer.popBatch();
  }

  std::vector<BoltDatasetPtr> loadInMemory() final {
    return loadInMemoryWithMaxBatches(std::numeric_limits<uint32_t>::max())
        .value();
  }

  std::optional<std::tuple<BoltDatasetPtr, BoltDatasetPtr>>
  loadInMemoryWithMaxBatches(uint32_t max_in_memory_batches) {
#if THIRDAI_EXPOSE_ALL
    // This is useful internally but we don't want to expose it to keep the
    // output clear and simple.
    std::cout << "loading data | source '" << _data_source->resourceName()
              << "'" << std::endl;
#endif

    auto start = std::chrono::high_resolution_clock::now();

    uint32_t batch_cnt = 0;
    while (batch_cnt < max_in_memory_batches && addNextBatchToBuffer()) {
      batch_cnt++;
    }

    auto [input_batches, label_batches] = _buffer.exportBuffer();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();

    if (input_batches.empty()) {
#if THIRDAI_EXPOSE_ALL
      // This is to ensure that it always prints complete if it prints that it
      // has started loading above.
      std::cout << "loading data | source '" << _data_source->resourceName()
                << "' | vectors 0 | batches 0 | time " << duration
                << "s | complete\n"
                << std::endl;
#endif
      return std::nullopt;
    }

    auto dataset = std::make_tuple(
        std::make_shared<BoltDataset>(std::move(input_batches)),
        std::make_shared<BoltDataset>(std::move(label_batches)));

    std::cout << "loading data | source '" << _data_source->resourceName()
              << "' | vectors " << std::get<0>(dataset)->len() << " | batches "
              << std::get<0>(dataset)->numBatches() << " | time " << duration
              << "s | complete\n"
              << std::endl;

    return dataset;
  }

  uint32_t getInputDim() { return _processor->getInputDim(); }

  uint32_t getLabelDim() { return _processor->getLabelDim(); }

  void restart() final {
    _data_source->restart();
    // When we restart we need to make sure we don't reread the header. s
    if (_processor->expectsHeader()) {
      auto header = _data_source->nextLine();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
    }

    _buffer = ShuffleBatchBuffer(
        /* shuffle_seed= */ time(NULL),
        /* batch_size= */ _data_source->getMaxBatchSize());
  }

 private:
  void prefillShuffleBuffer() {
    size_t n_prefill_batches = _batch_buffer_size - 1;
    size_t n_added = 0;
    while (n_added < n_prefill_batches && addNextBatchToBuffer()) {
      n_added++;
    }
  }

  bool addNextBatchToBuffer() {
    auto batch = StreamingDataset<BoltBatch, BoltBatch>::nextBatchVector();
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