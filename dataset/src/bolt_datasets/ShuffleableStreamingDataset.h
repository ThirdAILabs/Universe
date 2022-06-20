#pragma once

#include "BatchProcessor.h"
#include "BoltDatasets.h"
#include "DataLoader.h"
#include "ShuffleBatchBuffer.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <ctime>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset {

class ShuffleableStreamingDataset {
 public:
  ShuffleableStreamingDataset(
      std::shared_ptr<DataLoader> data_loader,
      std::shared_ptr<BatchProcessor<bolt::BoltBatch>> batch_processor,
      bool shuffle = false, ShuffleBufferConfig config = ShuffleBufferConfig())
      : _data_loader(std::move(data_loader)),
        _batch_processor(std::move(batch_processor)),
        _buffer(config.seed),
        _shuffle(shuffle),
        _buffer_size(config.buffer_size) {
    /*
      Different formats of data may or may not contain headers. Thus we
      delegate to the particular batch processor to determine if a header is
      needed. The first row is interpreted as the header. The batch processor
      is responsible for checking that the header is properly formatted.
    */
    if (_batch_processor->expectsHeader()) {
      auto header = _data_loader->getHeader();
      if (!header) {
        throw std::invalid_argument("Cannot read empty file.");
      }
      _batch_processor->processHeader(*header);
    }
  }

  std::optional<std::pair<bolt::BoltBatch, bolt::BoltBatch>> nextBatch() {
    if (_buffer.empty()) {
      prefillShuffleBuffer();
    }
    addNextBatchToBuffer();

    return _buffer.popBatch();
  }

  void prefillShuffleBuffer() {
    for (size_t i = 0; i < _buffer_size - 1; i++) {
      addNextBatchToBuffer();
    }
  }

  void addNextBatchToBuffer() {
    auto rows = _data_loader->nextBatch();
    if (!rows) {
      return;
    }
    auto batch = _batch_processor->createBatch(rows.value());
    _buffer.insertBatch(std::move(batch.value()), _shuffle);
  }

  std::pair<BoltDatasetPtr, BoltDatasetPtr> loadInMemory() {
    while (auto batch = nextBatch()) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
    }

    return _buffer.exportBuffer();
  }

  uint32_t getMaxBatchSize() const { return _data_loader->getMaxBatchSize(); }

 private:
  std::shared_ptr<DataLoader> _data_loader;
  std::shared_ptr<BatchProcessor<bolt::BoltBatch>> _batch_processor;
  ShuffleBatchBuffer _buffer;
  bool _shuffle;
  size_t _buffer_size;
};

}  // namespace thirdai::dataset