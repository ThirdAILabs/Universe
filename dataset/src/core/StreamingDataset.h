#pragma once

#include "BatchProcessor.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/core/InputTargetShuffleBuffer.h>
#include <dataset/src/loaders/LoaderInterface.h>
#include <optional>

namespace thirdai::dataset {

class StreamingDataset {
 public:
  StreamingDataset(
    std::shared_ptr<thirdai::dataset::Loader> loader, 
    std::vector<std::shared_ptr<Block>> input_blocks,
    std::vector<std::shared_ptr<Block>> target_blocks,
    size_t batch_size,
    size_t est_num_samples=0,
    bool shuffle=false,
    size_t shuffle_buffer_size=0);

  OptionalInputTargetBatch nextBatch();

  std::pair<BoltDatasetPtr, BoltDatasetPtr> loadInMemory();

 private:
  void fetchBatch();

  static uint32_t numBatchesInBuffer(size_t buffer_size, size_t batch_size) {
    buffer_size = std::max(buffer_size, batch_size);
    return (buffer_size + batch_size - 1) / batch_size;
  }

  std::shared_ptr<thirdai::dataset::Loader> _loader;
  bool _has_target;
  size_t _batch_size;
  size_t _est_num_samples;
  bool _shuffle;
  InputTargetShuffleBuffer _buffer;
  BatchProcessor _processor;
  std::optional<std::vector<std::string>> _next_batch_to_process;
  
};

} // namespace thirdai::dataset