#include "BatchProcessor.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/loaders/LoaderInterface.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <sys/types.h>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace thirdai::dataset {

BatchProcessor::BatchProcessor(
    std::vector<std::shared_ptr<Block>> input_blocks,
    std::vector<std::shared_ptr<Block>> target_blocks,
    uint32_t output_batch_size)
    : _batch_size(output_batch_size),
      _input_blocks_dense(std::all_of(input_blocks.begin(), input_blocks.end(),
                                      [](const std::shared_ptr<Block>& block) {
                                        return block->isDense();
                                      })),
      _target_blocks_dense(std::all_of(target_blocks.begin(),
                                       target_blocks.end(),
                                       [](const std::shared_ptr<Block>& block) {
                                         return block->isDense();
                                       })),
      /**
       * Here we copy input_blocks and target_blocks because when we
       * accept a vector representation of a Python List created by
       * PyBind11, the vector does not persist beyond this function
       * call, which results in segfaults later down the line.
       * It is therefore safest to just copy these vectors.
       * Furthermore, these vectors are cheap to copy since they contain a
       * small number of elements and each element is a pointer.
       */
      _input_blocks(std::move(input_blocks)),
      _target_blocks(std::move(target_blocks)) {}

void BatchProcessor::processBatch(
    std::vector<std::string>&& batch, Loader& loader, InputTargetBuffer& buffer, bool shuffle) {
  buffer.initiateNewBatch();

#pragma omp parallel for default(none) shared(batch, loader, buffer, std::cout)
  for (size_t i = 0; i < batch.size(); ++i) {
    auto row = loader.parse(batch[i]);
    // std::cout << "phew" << std::endl;
    buffer.addNewBatchInputVec(i, makeVector(row, _input_blocks, _input_blocks_dense));
    if (!_target_blocks.empty()) {
      buffer.addNewBatchTargetVec(i, makeVector(row, _target_blocks, _target_blocks_dense));
    }
  }

  // std::cout << "About to finalize" << std::endl;
  buffer.finalizeNewBatch(shuffle);

  // std::cout << "Done with that too" << std::endl;
}

bolt::BoltVector BatchProcessor::makeVector(
    std::vector<std::string_view>& sample,
    std::vector<std::shared_ptr<Block>>& blocks, bool blocks_dense) {
  std::shared_ptr<SegmentedFeatureVector> vec_ptr;

  // Dense vector if all blocks produce dense features, sparse vector
  // otherwise.
  if (blocks_dense) {
    vec_ptr = std::make_shared<SegmentedDenseFeatureVector>();
  } else {
    vec_ptr = std::make_shared<SegmentedSparseFeatureVector>();
  }

  // Let each block encode the input sample and adds a new segment 
  // containing this encoding to the vector.
  for (auto& block : blocks) {
    block->addVectorSegment(sample, *vec_ptr);
  }

  return vec_ptr->toBoltVector();
}
}  // namespace thirdai::dataset
