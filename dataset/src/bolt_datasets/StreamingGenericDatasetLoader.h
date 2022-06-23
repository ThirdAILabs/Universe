#pragma once

#include "DataLoader.h"
#include "ShuffleBatchBuffer.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/batch_processors/GenericBatchProcessor.h>
#include <memory>

namespace thirdai::dataset {

class StreamingGenericDatasetLoader {
 public:
  StreamingGenericDatasetLoader(
      std::string filename, std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, uint32_t batch_size,
      bool shuffle = false, ShuffleBufferConfig config = ShuffleBufferConfig(),
      bool has_header = false, char delimiter = ',')
      : _processor(std::make_shared<GenericBatchProcessor>(
            std::move(input_blocks), std::move(label_blocks), has_header,
            delimiter)),
        _streamer(std::make_shared<SimpleFileDataLoader>(filename, batch_size),
                  _processor),
        _buffer(config.seed),
        _shuffle(shuffle),
        _buffer_size(config.buffer_size) {}

  std::optional<BoltDataLabelPair<bolt::BoltBatch>> nextBatch() {
    if (_buffer.empty()) {
      prefillShuffleBuffer();
    }
    addNextBatchToBuffer();

    return _buffer.popBatch();
  }

  std::pair<std::shared_ptr<InMemoryDataset<bolt::BoltBatch>>, BoltDatasetPtr>
  loadInMemory() {
    while (auto batch = _streamer.nextBatch()) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
    }

    return _buffer.exportBuffer();
  }

  uint32_t getMaxBatchSize() const { return _streamer.getMaxBatchSize(); }

  uint32_t getInputDim() { return _processor->getInputDim(); }

  uint32_t getLabelDim() { return _processor->getLabelDim(); }

 private:
  void prefillShuffleBuffer() {
    for (size_t i = 0; i < _buffer_size - 1; i++) {
      addNextBatchToBuffer();
    }
  }

  void addNextBatchToBuffer() {
    auto batch = _streamer.nextBatch();
    if (batch) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
    }
  }

  std::shared_ptr<GenericBatchProcessor> _processor;
  StreamingDataset<bolt::BoltBatch> _streamer;
  ShuffleBatchBuffer _buffer;
  bool _shuffle;
  size_t _buffer_size;
};

}  // namespace thirdai::dataset