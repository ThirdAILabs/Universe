#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Encodes input samples – each represented by a sequence of strings –
 * as input and target BoltVectors according to the given blocks.
 * It processes these sequences in batches.
 */
struct BatchProcessor {
  BatchProcessor(std::vector<std::shared_ptr<Block>>& input_blocks,
                 std::vector<std::shared_ptr<Block>>& target_blocks,
                 uint32_t output_batch_size);

  /**
   * Consumes a batch of input samples and encodes them
   * as vectors.
   */
  void processBatch(std::vector<std::vector<std::string>>& batch);

  /**
   * Produces an InMemoryDataset of BoltInputBatches containing the
   * vectors processed so far.
   * This method can optionally produce a shuffled dataset.
   */
  dataset::InMemoryDataset<dataset::BoltInputBatch> exportInMemoryDataset(
      bool shuffle = false, uint32_t shuffle_seed = 0);

 private:
  /**
   * Helper function that preallocates space for new vectors.
   */
  void allocateMemoryForBatch(uint32_t size);

  /**
   * Produces an InMemoryDataset of BoltBatches containing the vectors
   * processed so far. Vectors are positioned according to the given
   * positions mapping.
   *
   * positions[i] = the original position of the vector that will be
   *                in position i in the exported dataset.
   */
  dataset::InMemoryDataset<dataset::BoltInputBatch> makeDatasetWithPositions(
      uint32_t n_exported, std::vector<uint32_t>& positions);

  /**
   * Produces a mapping from the final position of a vector in
   * the exported dataset to its original position based on the
   * shuffle and shuffle_seed arguments.
   */
  static std::vector<uint32_t> makeFinalPositions(uint32_t n_exported,
                                                  bool shuffle,
                                                  uint32_t shuffle_seed);

  /**
   * Encodes a sample as a BoltVector according to the given blocks.
   */
  static bolt::BoltVector makeVector(
      std::vector<std::string>& sample,
      std::vector<std::shared_ptr<Block>>& blocks, bool blocks_dense);

  uint32_t _batch_size;
  bool _input_blocks_dense;
  bool _target_blocks_dense;
  std::vector<bolt::BoltVector> _input_vectors;
  std::vector<bolt::BoltVector> _target_vectors;
  std::vector<std::shared_ptr<Block>> _input_blocks;
  std::vector<std::shared_ptr<Block>> _target_blocks;
};

}  // namespace thirdai::dataset
