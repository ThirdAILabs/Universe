#pragma once

#include "DataLoader.h"
#include "ShuffleBatchBuffer.h"
#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/batch_processors/GenericBatchProcessor.h>
#include <memory>

namespace thirdai::dataset {

class StreamingGenericDatasetLoader : public StreamingDataset<bolt::BoltBatch, bolt::BoltBatch> {
 public:
  // The idea is to pass the input loader and generic batch processor into the
  // primary constructor This primary constructor calls the base class
  // constructor.

  StreamingGenericDatasetLoader(
      std::shared_ptr<DataLoader> loader,
      std::shared_ptr<GenericBatchProcessor> processor, bool shuffle = false,
      ShuffleBufferConfig config = ShuffleBufferConfig())
      : StreamingDataset(std::move(loader), processor),
        _processor(std::move(processor)),
        _buffer(config.seed),
        _shuffle(shuffle),
        _buffer_size(config.buffer_size) {}

  StreamingGenericDatasetLoader(
      std::shared_ptr<DataLoader> loader,
      std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, bool shuffle = false,
      ShuffleBufferConfig config = ShuffleBufferConfig(),
      bool has_header = false, char delimiter = ',')
      : StreamingGenericDatasetLoader(
            std::move(loader),
            std::make_shared<GenericBatchProcessor>(std::move(input_blocks),
                                                    std::move(label_blocks),
                                                    has_header, delimiter),
            shuffle, config) {}

  StreamingGenericDatasetLoader(
      std::string filename, std::vector<std::shared_ptr<Block>> input_blocks,
      std::vector<std::shared_ptr<Block>> label_blocks, uint32_t batch_size,
      bool shuffle = false, ShuffleBufferConfig config = ShuffleBufferConfig(),
      bool has_header = false, char delimiter = ',')
      : StreamingGenericDatasetLoader(
            std::make_shared<SimpleFileDataLoader>(filename, batch_size),
            std::move(input_blocks), std::move(label_blocks), shuffle, config,
            has_header, delimiter) {}

  std::optional<std::tuple<bolt::BoltBatch, bolt::BoltBatch>> nextBatchTuple() final {
    if (_buffer.empty()) {
      prefillShuffleBuffer();
    }
    addNextBatchToBuffer();

    return _buffer.popBatch();
  }

  std::tuple<std::shared_ptr<InMemoryDataset<bolt::BoltBatch>>, std::shared_ptr<InMemoryDataset<bolt::BoltBatch>>>
  loadInMemory() final {
    while (auto batch = StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::nextBatchTuple()) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
    }

    return _buffer.exportBuffer();
  }

  uint32_t getInputDim() { return _processor->getInputDim(); }

  uint32_t getLabelDim() { return _processor->getLabelDim(); }

 private:
  void prefillShuffleBuffer() {
    for (size_t i = 0; i < _buffer_size - 1; i++) {
      addNextBatchToBuffer();
    }
  }

  void addNextBatchToBuffer() {
    auto batch = StreamingDataset<bolt::BoltBatch, bolt::BoltBatch>::nextBatchTuple();
    if (batch) {
      _buffer.insertBatch(std::move(batch.value()), _shuffle);
    }
  }

  std::shared_ptr<GenericBatchProcessor> _processor;
  ShuffleBatchBuffer _buffer;
  bool _shuffle;
  size_t _buffer_size;
};

}  // namespace thirdai::dataset