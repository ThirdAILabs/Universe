#include <bolt/src/layers/BoltVector.h>
#include <limits>
#include <memory>
#include <random>
#include <vector>
#include <dataset/src/BuilderVectors.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

struct BatchProcessor {
  BatchProcessor(std::vector<std::shared_ptr<Block>>& input_blocks,
                 std::vector<std::shared_ptr<Block>>& target_blocks,
                 uint32_t output_batch_size): 
                 _batch_size(output_batch_size),
                 _input_blocks(input_blocks),
                 _target_blocks(target_blocks) {}

  void processBatch(std::vector<std::vector<std::string>>& batch) {
    auto initial_num_elems = _input_vectors.size();
    allocateMemoryForBatch(batch.size());
#pragma omp parallel for default(none) shared(batch, initial_num_elems)
    for (size_t i = 0; i < batch.size(); ++i) {
      _input_vectors[initial_num_elems + i] = makeVector(batch[i], _input_blocks);
      _target_vectors[initial_num_elems + i] = makeVector(batch[i], _target_blocks);
    }
  }

  dataset::InMemoryDataset<dataset::BoltInputBatch> exportInMemoryDataset(bool shuffle=false, uint32_t shuffle_seed=0) {
    assert(_input_vectors.size() == _target_vectors.size()); // Assert we always have targets even if the target vectors are empty because BOLT expects it.
    uint32_t n_exported = _input_vectors.size();
    auto positions = getFinalPositions(n_exported, shuffle, shuffle_seed);

    auto batches = makeBatches(n_exported, positions);
    return { std::move(batches), n_exported };
  }

 private:
  void allocateMemoryForBatch(uint32_t size) {
    _input_vectors.resize(_input_vectors.size() + size);
    _target_vectors.resize(_target_vectors.size() + size);
  }

  std::vector<dataset::BoltInputBatch> makeBatches(uint32_t n_exported, std::vector<uint32_t>& positions) {
    std::vector<dataset::BoltInputBatch> batches;

    for (uint32_t batch_start_index = 0; batch_start_index < n_exported; batch_start_index += _batch_size) {
      std::vector<bolt::BoltVector> batch_inputs;
      std::vector<bolt::BoltVector> batch_labels;
      for (uint32_t index_in_batch = 0; index_in_batch < std::min(_batch_size, n_exported - batch_start_index); index_in_batch++) {
        batch_inputs.push_back(std::move(_input_vectors[positions[batch_start_index + index_in_batch]]));
        batch_labels.push_back(std::move(_target_vectors[positions[batch_start_index + index_in_batch]]));
      }
      batches.emplace_back(std::move(batch_inputs), std::move(batch_labels));
    }

    // Replenish after moves.
    _input_vectors = std::vector<bolt::BoltVector>();
    _target_vectors = std::vector<bolt::BoltVector>();
    
    return batches;
  }

  static std::vector<uint32_t> getFinalPositions(uint32_t n_exported, bool shuffle, uint32_t shuffle_seed) {
    std::vector<uint32_t> positions(n_exported);
    for (uint32_t i = 0; i < n_exported; i++) {
      positions[i] = i;
    }

    if (shuffle) {
      auto rng = std::default_random_engine {};
      rng.seed(shuffle_seed);
      std::shuffle(positions.begin(), positions.end(), rng);
    }

    return positions;
  }

  static bolt::BoltVector makeVector(std::vector<std::string>& row, std::vector<std::shared_ptr<Block>>& blocks) {
    SparseBuilderVector vec;
    uint32_t offset = 0;
    for (auto& block : blocks) {
      block->process(row, vec, offset);
      offset += block->featureDim();
    }
    
    return vec.toBoltVector();
  }

  uint32_t _batch_size;
  std::vector<bolt::BoltVector> _input_vectors;
  std::vector<bolt::BoltVector> _target_vectors;
  std::vector<std::shared_ptr<Block>> _input_blocks;
  std::vector<std::shared_ptr<Block>> _target_blocks;
  
};

} // namespace thirdai::dataset
