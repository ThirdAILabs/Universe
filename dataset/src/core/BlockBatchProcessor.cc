#include "BlockBatchProcessor.h"
#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
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

BlockBatchProcessor::BlockBatchProcessor(
    std::vector<std::shared_ptr<Block>> input_blocks,
    std::vector<std::shared_ptr<Block>> target_blocks,
    uint32_t output_batch_size, size_t est_num_elems)
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
      _target_blocks(std::move(target_blocks)) {
  _input_vectors.reserve(est_num_elems);
  if (!_target_blocks.empty()) {
    _target_vectors = std::vector<bolt::BoltVector>();
    _target_vectors->reserve(est_num_elems);
  }
}

void BlockBatchProcessor::processBatch(
    std::vector<std::vector<std::string>>& batch) {
  // Preallocate space for new vectors. This prevents data races and
  // preserves the order of vectors when processing them in parallel.
  uint32_t initial_num_elems = _input_vectors.size();

  // We use multiple emplace_backs instead of a resize() call to make
  // use of std::vector's exponential growth policy.
  for (uint32_t i = 0; i < batch.size(); i++) {
    _input_vectors.emplace_back();
  }
  if (_target_vectors) {
    for (uint32_t i = 0; i < batch.size(); i++) {
      _target_vectors->emplace_back();
    }
  }

  std::exception_ptr exception_ptr;
#pragma omp parallel for default(none) \
    shared(batch, initial_num_elems, exception_ptr)
  for (size_t i = 0; i < batch.size(); ++i) {
    _input_vectors[initial_num_elems + i] =
        makeVector(batch[i], _input_blocks, _input_blocks_dense, exception_ptr);

    if (_target_vectors) {
      _target_vectors->at(initial_num_elems + i) = makeVector(
          batch[i], _target_blocks, _target_blocks_dense, exception_ptr);
    }
  }
  if (exception_ptr) {
    std::rethrow_exception(exception_ptr);
  }
}

std::pair<BoltDatasetPtr, BoltDatasetPtr>
BlockBatchProcessor::exportInMemoryDataset(bool shuffle,
                                           uint32_t shuffle_seed) {
  // Produce final positions of vectors in the dataset according to
  // shuffle and shuffle_seed.
  uint32_t n_exported = _input_vectors.size();
  auto positions = makeFinalPositions(n_exported, shuffle, shuffle_seed);
  size_t n_batches = (n_exported + _batch_size - 1) / _batch_size;

  std::vector<bolt::BoltBatch> input_batches(n_batches);
  std::optional<std::vector<bolt::BoltBatch>> target_batches;
  if (_target_vectors) {
    target_batches = std::vector<bolt::BoltBatch>(n_batches);
  }

  // For each batch
#pragma omp parallel for default(none) \
    shared(n_exported, n_batches, input_batches, target_batches, positions)
  for (size_t batch_idx = 0; batch_idx < n_batches; ++batch_idx) {
    uint32_t batch_start_idx = batch_idx * _batch_size;

    // Vectors that hold the batch's input and target vectors.
    size_t cur_batch_size = std::min(_batch_size, n_exported - batch_start_idx);
    std::vector<bolt::BoltVector> batch_inputs(cur_batch_size);
    std::optional<std::vector<bolt::BoltVector>> batch_targets;
    if (_target_vectors) {
      batch_targets = std::vector<bolt::BoltVector>(cur_batch_size);
    }

    // For each vector in the batch
    for (uint32_t vec_idx = 0; vec_idx < cur_batch_size; ++vec_idx) {
      // Move vectors to prevent copying.
      batch_inputs[vec_idx] =
          std::move(_input_vectors[positions[batch_start_idx + vec_idx]]);
      if (batch_targets) {
        batch_targets->at(vec_idx) = std::move(
            _target_vectors->at(positions[batch_start_idx + vec_idx]));
      }
    }

    input_batches[batch_idx] = bolt::BoltBatch(std::move(batch_inputs));
    if (target_batches) {
      target_batches->at(batch_idx) =
          bolt::BoltBatch(std::move(batch_targets.value()));
    }
  }

  // Replenish after moves.
  _input_vectors = std::vector<bolt::BoltVector>();
  if (_target_vectors) {
    _target_vectors = std::vector<bolt::BoltVector>();
  }

  return {std::make_shared<BoltDataset>(std::move(input_batches), n_exported),
          target_batches ? std::make_shared<BoltDataset>(
                               std::move(target_batches.value()), n_exported)
                         : nullptr};
}

std::vector<uint32_t> BlockBatchProcessor::makeFinalPositions(
    uint32_t n_exported, bool shuffle, uint32_t shuffle_seed) {
  // Create identity mapping.
  std::vector<uint32_t> positions(n_exported);
  std::iota(positions.begin(), positions.end(), 0);

  // Shuffle if necessary.
  if (shuffle) {
    auto rng = std::default_random_engine{};
    rng.seed(shuffle_seed);
    std::shuffle(positions.begin(), positions.end(), rng);
  }

  return positions;
}

bolt::BoltVector BlockBatchProcessor::makeVector(
    std::vector<std::string>& sample,
    std::vector<std::shared_ptr<Block>>& blocks, bool blocks_dense,
    std::exception_ptr exception_ptr) {
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
    std::vector<std::string_view> sample_view(sample.size());
    for (uint32_t i = 0; i < sample.size(); i++) {
      sample_view[i] = std::string_view(sample[i].c_str(), sample[i].size());
    }
    block->addVectorSegment(sample_view, *vec_ptr, exception_ptr);
  }
  return vec_ptr->toBoltVector();
}
}  // namespace thirdai::dataset
