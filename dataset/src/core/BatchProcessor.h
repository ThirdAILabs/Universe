#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <dataset/src/Dataset.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/bolt_datasets/BoltDatasets.h>
#include <cstdlib>
#include <random>

namespace thirdai::dataset {

/**
 * Encodes input samples – each represented by a sequence of strings –
 * as input and target BoltVectors according to the given blocks.
 * It processes these sequences in batches.
 */
class BatchProcessor {
 public:
  BatchProcessor(std::vector<std::shared_ptr<Block>>& input_blocks,
                 std::vector<std::shared_ptr<Block>>& target_blocks,
                 uint32_t output_batch_size);

  /**
   * Consumes a batch of input samples and encodes them
   * as vectors.
   */
  void processBatch(std::vector<std::vector<std::string>>& batch);

  /**
   * Produces an DatasetWithLabels containing the
   * vectors processed so far.
   * This method can optionally produce a shuffled dataset.
   */
  std::pair<BoltDatasetPtr, BoltDatasetPtr> exportInMemoryDataset(
      bool shuffle = false, uint32_t shuffle_seed = std::rand());

 private:
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
  /**
   * because using references will cause errors when given Python
   * lists through PyBind11. This is because while the PyBind11 creates
   * an std::vector representation of a Python list when passing it to
   * a C++ function, the vector does not persist beyond the function
   * call, so future references to the vector will cause a segfault.
   * Furthermore, these vectors are cheap to copy since they contain a
   * small number of elements and each element is a pointer.
   */
  std::vector<std::shared_ptr<Block>> _input_blocks;
  std::vector<std::shared_ptr<Block>> _target_blocks;
};

}  // namespace thirdai::dataset
