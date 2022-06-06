#include "StreamingDataset.h"
#include <bolt/src/layers/BoltVector.h>
#include <thread>


namespace thirdai::dataset {

StreamingDataset::StreamingDataset(
  std::shared_ptr<thirdai::dataset::Loader> loader, 
  std::vector<std::shared_ptr<Block>> input_blocks,
  std::vector<std::shared_ptr<Block>> target_blocks,
  size_t batch_size,
  size_t est_num_samples,
  bool shuffle,
  size_t shuffle_buffer_size)
  : _loader(std::move(loader)),
    _has_target(!target_blocks.empty()),
    _batch_size(batch_size),
    _est_num_samples(est_num_samples),
    _shuffle(shuffle),
    _buffer(batch_size, numBatchesInBuffer(shuffle_buffer_size, batch_size), !target_blocks.empty()),
    _processor(
      std::move(input_blocks), 
      std::move(target_blocks), 
      [&](const std::string& row){ 
        return _loader->parse(row); 
      }, 
      _batch_size) {
  
  _loader->initialize(); // where we remove header
  _next_batch_to_process = _loader->nextBatch(_batch_size);
  // Prefetch. For shuffling.
  // We prefetch one less batch because another batch is fetched when
  // nextBatch() is called.
  uint32_t n_prefetch_batches = numBatchesInBuffer(shuffle_buffer_size, batch_size) - 1;
  for (uint32_t i = 0; i < n_prefetch_batches; i++) {
    fetchBatch();
  }
}

OptionalInputTargetBatch StreamingDataset::nextBatch() {
  fetchBatch();
  return _buffer.nextBatch();
}

std::pair<BoltDatasetPtr, BoltDatasetPtr> StreamingDataset::loadInMemory() {
  auto est_num_batches = _est_num_samples;
  std::vector<bolt::BoltBatch> inputs;
  inputs.reserve(est_num_batches);
  std::vector<bolt::BoltBatch> targets;
  if (_has_target) {
    targets.reserve(est_num_batches);
  }
  
  uint64_t len = 0;

  while (auto batch = nextBatch()) {
    len += batch->first.getBatchSize();
    inputs.push_back(std::move(batch->first));
    if (batch->second.has_value()) {
      targets.push_back(std::move(batch->second.value()));
    }
  }
  
  return {
    std::make_shared<BoltDataset>(std::move(inputs), len),
    targets.empty() ? nullptr : 
      std::make_shared<BoltDataset>(std::move(targets), len),
  };
}

void StreamingDataset::fetchBatch() {
  // If there's nothing to process, then nothing was loaded in the previous iteration.
  // Thus, nothing can be loaded in this iteration.
  // We should just return.
  if (!_next_batch_to_process.has_value()) {
    return;
  }
  
  std::vector<std::string> cur_batch_to_process = std::move(_next_batch_to_process.value());
  std::thread loader_thread([&](){
    _next_batch_to_process = _loader->nextBatch(_batch_size);
  });
  auto processed_batch = _processor.processBatch(cur_batch_to_process);
  _buffer.addBatch(std::move(processed_batch), _shuffle);
  
  loader_thread.join();
}

} // namespace thirdai::dataset