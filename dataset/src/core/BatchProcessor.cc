#include "BatchProcessor.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <dataset/src/utils/ExtendableVectors.h>
#include <sys/types.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>

namespace thirdai::dataset {

BatchProcessor::BatchProcessor(
    std::vector<std::shared_ptr<Block>>& input_blocks,
    std::vector<std::shared_ptr<Block>>& target_blocks,
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
      _input_blocks(input_blocks),
      _target_blocks(target_blocks) {}

void BatchProcessor::processBatch(
    std::vector<std::vector<std::string>>& batch) {
  // TODO(Geordie): Make a python version of this class with a method
  // that releases GIL so we can process batches while the
  // next input rows are processed in python. Cannot
  // Cannot do it here because it wouldn't compile.

  // Preallocate space for new vectors. This prevents data races and
  // preserves the order of vectors when processing them in parallel.
  bool has_target = !_target_blocks.empty();

  uint32_t initial_num_elems = _input_vectors.size();

  _input_vectors.resize(_input_vectors.size() + batch.size());
  if (has_target) {
    _target_vectors.resize(_target_vectors.size() + batch.size());
  }

#pragma omp parallel for default(none) shared(batch, initial_num_elems, has_target)
  for (size_t i = 0; i < batch.size(); ++i) {
    _input_vectors[initial_num_elems + i] =
        makeVector(batch[i], _input_blocks, _input_blocks_dense);

    // We use a template argument so we don't check for the has_target
    // condition in each iteration.
    if (has_target) {
      _target_vectors[initial_num_elems + i] =
          makeVector(batch[i], _target_blocks, _target_blocks_dense);
    }
  }
}

std::pair<BoltDatasetPtr, BoltDatasetPtr> BatchProcessor::exportInMemoryDataset(bool shuffle, uint32_t shuffle_seed) {
  // We currently assert that we always have targets even if the target
  // vectors are empty because BOLT expects it.
  // TODO(Geordie, Nicholas): How do we represent a dataset without labels?
  assert(_input_vectors.size() == _target_vectors.size());

  // Produce final positions of vectors in the dataset according to
  // shuffle and shuffle_seed.
  uint32_t n_exported = _input_vectors.size();
  auto positions = makeFinalPositions(n_exported, shuffle, shuffle_seed);
  bool has_target = !_target_blocks.empty();
  size_t n_batches = (n_exported + _batch_size - 1) / _batch_size;

  std::vector<bolt::BoltBatch> input_batches(n_batches);
  std::vector<bolt::BoltBatch> target_batches(has_target ? n_batches : 0);

  // For each batch
#pragma omp parallel for default(none) \
    shared(n_exported, n_batches, input_batches, target_batches, positions, has_target)
  for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    uint32_t batch_start_idx = batch_idx * _batch_size;

    // Vectors that hold the batch's input and target vectors.
    size_t cur_batch_size = std::min(_batch_size, n_exported - batch_start_idx);
    std::vector<bolt::BoltVector> batch_inputs(cur_batch_size);
    size_t target_batch_size = has_target ? cur_batch_size : 0;
    std::vector<bolt::BoltVector> batch_targets(target_batch_size);

    // For each vector in the batch
    for (uint32_t vec_idx = 0; vec_idx < cur_batch_size; ++vec_idx) {
      // Move vectors to prevent copying.
      batch_inputs[vec_idx] =
          std::move(_input_vectors[positions[batch_start_idx + vec_idx]]);
      if (has_target) {
        batch_targets[vec_idx] =
            std::move(_target_vectors[positions[batch_start_idx + vec_idx]]);
      }
    }

    input_batches[batch_idx] = bolt::BoltBatch(std::move(batch_inputs));
    if (has_target) {
      target_batches[batch_idx] = bolt::BoltBatch(std::move(batch_targets));
    }
  }

  // Replenish after moves.
  _input_vectors = std::vector<bolt::BoltVector>();
  _target_vectors = std::vector<bolt::BoltVector>();

  return {
    std::make_shared<BoltDataset>(std::move(input_batches), n_exported),
    target_batches.empty()
      ? nullptr 
      : std::make_shared<BoltDataset>(std::move(target_batches), n_exported)
  };
}

std::vector<uint32_t> BatchProcessor::makeFinalPositions(
    uint32_t n_exported, bool shuffle, uint32_t shuffle_seed) {
  // Create identity mapping.
  std::vector<uint32_t> positions(n_exported);
  for (uint32_t i = 0; i < n_exported; i++) {
    positions[i] = i;
  }

  // Shuffle if necessary.
  if (shuffle) {
    auto rng = std::default_random_engine{};
    rng.seed(shuffle_seed);
    std::shuffle(positions.begin(), positions.end(), rng);
  }

  return positions;
}

bolt::BoltVector BatchProcessor::makeVector(
    std::vector<std::string>& sample,
    std::vector<std::shared_ptr<Block>>& blocks, bool blocks_dense) {
  std::shared_ptr<ExtendableVector> vec_ptr;

  // Dense vector if all blocks produce dense features, sparse vector
  // otherwise.
  if (blocks_dense) {
    vec_ptr = std::static_pointer_cast<ExtendableVector>(
        std::make_shared<DenseExtendableVector>());
  } else {
    vec_ptr = std::static_pointer_cast<ExtendableVector>(
        std::make_shared<SparseExtendableVector>());
  }

  // Let each block encode the input sample and extend the vector
  // with this encoding.
  for (auto& block : blocks) {
    block->extendVector(sample, *vec_ptr);
  }

  return vec_ptr->toBoltVector();
}
}  // namespace thirdai::dataset
